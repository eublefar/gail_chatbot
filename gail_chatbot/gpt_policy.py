from typing import List, Tuple, Dict, Union
import os
import gc
import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from gym_loop.policies.base_policy import BasePolicy
from torch.cuda.amp import autocast


class GptPolicy(torch.nn.Module, BasePolicy):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(768, 512), torch.nn.ReLU(False), torch.nn.Linear(512, 1),
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
        try:
            action = torch.distributions.Categorical(
                outp["action_distribution"].cpu()
            ).sample()
        except RuntimeError:
            print(outp["action_distribution"])
            print(torch.isnan(outp["action_distribution"]).any())
        outp["action"] = action.numpy()

        new_cache_str = state[0] + "\n".join(state[1]) + " ".join(state[2])
        self.cache[new_cache_str] = outp["past_key_values"]
        return outp

    def batch_act(self, state_batch):
        if isinstance(self.cache, dict):
            past_key_values = None
        else:
            past_key_values = self.cache

        outp = self(state_batch, past_key_values)

        try:
            action = torch.distributions.Categorical(
                outp["action_distribution"].cpu()
            ).sample()
        except RuntimeError:
            print(outp["action_distribution"])
            print(torch.isnan(outp["action_distribution"]).any())
        outp["actions"] = action.numpy()
        del self.cache
        self.cache = outp["past_key_values"]
        return outp

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(
        self,
        state_batch: List[Tuple[str, List[str], List[str]]],
        past_key_values: List[torch.Tensor] = None,
    ):
        if past_key_values is not None:
            state_batch = [("", [], [state[2][-1]]) for state in state_batch]

        input_ids, token_type_ids_batch, seqlen, attention_mask = self._build_inputs(
            state_batch
        )

        input_ids = torch.LongTensor(input_ids).to(self.get_device())
        token_type_ids = torch.LongTensor(
            token_type_ids_batch
        ).to(self.get_device())
        attention_mask = torch.LongTensor(attention_mask).to(self.get_device())
        with autocast():
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
        ).to(self.get_device())
        action_dist = action_dist.gather(1, last_action_ids).squeeze(1)

        last_feature_ids = (
            (torch.LongTensor(seqlen) - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([-1, -1, last_layer_hidden_states.shape[-1]])
        ).to(self.get_device())
        features = last_layer_hidden_states.gather(1, last_feature_ids).squeeze(1)
        values = self.value_head(features)

        return {
            "action_distribution": torch.nn.functional.softmax(action_dist, dim=-1)
            + 1e-8,
            "values": values.squeeze(-1),
            "past_key_values": cache,
        }

    def _build_inputs(self, state_batch: List[Tuple[str, List[str], List[str]]]):
        token_type_batch = []
        input_words = []
        seqlen = []
        for state in state_batch:
            persona, history, utterance = state
            if persona:
                persona_tokens = self.tokenizer.tokenize(persona) + [
                    self.tokenizer.eos_token
                ]
            else:
                persona_tokens = []
            input_tokens = []
            persona_types = [1] * len(persona_tokens)
            utterance_types = [1] * len(utterance)
            token_types = []
            turn_len = []
            for step_id, turn in enumerate(history):
                if "__SILENCE__" in turn:
                    turn = turn.replace("__SILENCE__", self.tokenizer.eos_token)
                    tokenized = self.tokenizer.tokenize(turn)
                else:
                    tokenized = self.tokenizer.tokenize(turn) + [
                        self.tokenizer.eos_token
                    ]
                turn_len.append(len(tokenized))
                input_tokens += tokenized
                if step_id % 2 == 0:
                    token_types += [0] * len(tokenized)
                else:
                    token_types += [1] * len(tokenized)
                if (len(input_tokens) + len(persona_tokens)) >= 512:
                    first_turn_len = turn_len.pop(0)
                    input_tokens = input_tokens[first_turn_len:]
                    token_types = token_types[first_turn_len:]
            seqlen.append(len(persona_types) + len(token_types) + len(utterance_types))
            token_type_batch.append(persona_types + token_types + utterance_types)
            input_words.append(persona_tokens + input_tokens + utterance)
        max_len = max(seqlen)
        attention_mask = []
        for token_type, inp_seq in zip(token_type_batch, input_words):
            pad_num = max_len - len(token_type)
            attention_mask.append([1] * len(token_type) + [0] * pad_num)
            token_type.extend([0] * pad_num)
            inp_seq.extend([self.tokenizer.eos_token] * pad_num)
        token_type_ids_batch = [token_type for token_type in token_type_batch]
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
        gc.collect()
