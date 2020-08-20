from typing import List, Tuple, Dict, Union
import os
import gc
import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from gym_loop.policies.base_policy import BasePolicy
from contextlib import suppress
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

try:
    from torch.cuda.amp import autocast

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False


class GptPolicy(torch.nn.Module, BasePolicy):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

        self.loc_transform_layer = torch.nn.Linear(768, 768)
        self.std_layer = torch.nn.Linear(768, 768)

        self.value_head = torch.nn.Linear(768, 1)
        self.cache = None
        self.use_cache = True

    def save(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, "model.bin"))

    def load(self, path: str):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.bin")))
        self.value_head.load_state_dict(
            torch.load(os.path.join(path, "value_head.bin"))
        )

    def get_device(self):
        _, p = next(self.model.named_parameters())
        return p.device

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, state_batch: List[Tuple[str, List[str], List[str]]]):
        if self.use_cache:
            past_key_values = None
        else:
            past_key_values = self.cache

        if past_key_values is not None:
            state_batch = [("", [], [state[2][-1]]) for state in state_batch]
        input_ids, token_type_ids_batch, seqlen, attention_mask = self._build_inputs(
            state_batch
        )

        input_ids = torch.LongTensor(input_ids).to(self.get_device())
        token_type_ids = torch.LongTensor(token_type_ids_batch).to(self.get_device())
        attention_mask = torch.LongTensor(attention_mask).to(self.get_device())

        with autocast() if MIXED_PREC else suppress():
            last_layer_hidden_states, cache = self.model.transformer(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                past=past_key_values,
            )

        last_feature_ids = (
            (torch.LongTensor(seqlen) - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([-1, -1, last_layer_hidden_states.shape[-1]])
        ).to(self.get_device())
        features = last_layer_hidden_states.gather(1, last_feature_ids).squeeze(1)

        values = self.value_head(features)

        means = features + self.loc_transform_layer(features)
        stds = F.relu(self.std_layer(features)) + 1e-10

        self.cache = cache
        return {
            "action_distribution": MultivariateNormal(means, stds.diag_embed()),
            "values": values.squeeze(-1),
        }

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.clear_cache()
        self.use_cache = False

    def decode(self, hidden_states):
        hidden_states = torch.FloatTensor(hidden_states)
        return torch.argmax(self.model.lm_head(hidden_states), dim=-1)

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
        self.cache = None
        gc.collect()
