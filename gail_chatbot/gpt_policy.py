from typing import List, Tuple, Dict, Union
import os
import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from gym_loop.policies.base_policy import BasePolicy


class GptPolicy(torch.nn.Module, BasePolicy):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 512), torch.nn.ReLU(False), torch.nn.Linear(512, 1),
        )
        self.cache = {}

    def save(self, path: str):
        torch.save(self.model.state_dict(), os.path.join(path, "model.bin"))
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.bin"))

    def load(self, path: str):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.bin")))
        self.value_head.load_state_dict(
            torch.load(os.path.join(path, "value_head.bin"))
        )

    def get_device(self):
        _, p = next(self.model.named_parameters())
        return p.device

    def act(
        self, state: Tuple[str, List[str], List[str]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        cache_str = state[0] + "\n".join(state[1]) + " ".join(state[2][:-1])
        past_key_values = self.cache.pop(cache_str, None)
        outp = self([state], past_key_values)
        action = torch.distributions.Categorical(outp["action_distribution"]).sample()
        outp["action"] = action.detach().numpy()

        new_cache_str = state[0] + "\n".join(state[1]) + " ".join(state[2])
        self.cache[new_cache_str] = outp["past_key_values"]
        return outp

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(
        self,
        state_batch: List[Tuple[str, List[str], List[str]]],
        past_key_values: List[torch.Tensor] = None,
    ):
        if past_key_values is not None:
            # support caching for batch size of 1
            if len(state_batch) > 1:
                raise NotImplementedError(
                    "past_key_values not supported for batches of size > 1"
                )
            state_batch[0] = list(state_batch[0])
            state_batch[0][0] = ""
            state_batch[0][1] = []
            state_batch[0][2] = [state_batch[0][2][-1]]
        input_ids, token_type_ids_batch, seqlen, attention_mask = self._build_inputs(
            state_batch
        )

        input_ids = torch.LongTensor(input_ids).to(self.get_device())
        token_type_ids = torch.LongTensor(token_type_ids_batch).to(self.get_device())
        attention_mask = torch.LongTensor(attention_mask).to(self.get_device())

        action_dist, cache, hidden_states = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            past=past_key_values,
            output_hidden_states=True,
        )

        last_layer_hidden_states = hidden_states[-1]
        last_action_ids = (
            (torch.LongTensor(seqlen) - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([-1, -1, action_dist.shape[-1]])
        )
        action_dist = action_dist.gather(1, last_action_ids).squeeze(1)

        last_feature_ids = (
            (torch.LongTensor(seqlen) - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([-1, -1, last_layer_hidden_states.shape[-1]])
        )
        features = (
            last_layer_hidden_states.gather(1, last_feature_ids).squeeze(1).detach()
        )
        values = self.value_head(features)

        return {
            "action_distribution": torch.nn.functional.softmax(action_dist, dim=-1),
            "values": values.squeeze(-1),
            "past_key_values": cache,
        }

    def _build_inputs(self, state_batch: List[Tuple[str, List[str], List[str]]]):
        token_type_batch = []
        input_words = []
        seqlen = []

        for state in state_batch:
            persona, history, utterance = state
            input_tokens = self.tokenizer.tokenize(persona) + [self.tokenizer.eos_token]
            token_types = [1] * len(input_tokens)
            for step_id, turn in enumerate(history):
                turn = turn.replace("__SILENCE__", self.tokenizer.eos_token)
                tokenized = self.tokenizer.tokenize(turn) + [self.tokenizer.eos_token]
                input_tokens += tokenized
                if step_id % 2 == 0:
                    token_types += [0] * len(tokenized)
                else:
                    token_types += [1] * len(tokenized)
            seqlen.append(len(token_types))
            token_type_batch.append(token_types)
            input_words.append(input_tokens)
        max_len = max(seqlen)
        attention_mask = []
        for token_type, inp_seq in zip(token_type_batch, input_words):
            pad_num = max_len - len(token_type)
            attention_mask.append([1] * len(token_type) + [0] * pad_num)
            token_type.extend([self.tokenizer.eos_token] * pad_num)
            inp_seq.extend([self.tokenizer.eos_token] * pad_num)
        token_type_ids_batch = [
            self.tokenizer.convert_tokens_to_ids(token_type)
            for token_type in token_type_batch
        ]
        input_ids = [
            self.tokenizer.convert_tokens_to_ids(input_words_row)
            for input_words_row in input_words
        ]

        return input_ids, token_type_ids_batch, seqlen, attention_mask

    def reset_noise(self):
        pass

    def clear_cache(self):
        del self.cache
        self.cache = {}
