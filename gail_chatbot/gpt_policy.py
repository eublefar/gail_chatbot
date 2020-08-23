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
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.cuda.amp import autocast

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False


class GptPolicy(torch.nn.Module, BasePolicy):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})

        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
        self.loc_transform_layer = torch.nn.Linear(1024, 1024)
        self.std_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 1024),
        )

        self.value_head = torch.nn.Linear(1024, 1)
        self.cache = None
        self.use_cache = True

    def save(self, path: str):
        self.model.save_pretrained(os.path.join(path, "model.bin"))
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.bin"))
        torch.save(
            self.loc_transform_layer.state_dict(), os.path.join(path, "loc_head.bin")
        )
        torch.save(self.std_layer.state_dict(), os.path.join(path, "std_head.bin"))

    def load(self, path: str):
        self.model.from_pretrained(os.path.join(path, "model.bin"))
        self.value_head.load_state_dict(
            torch.load(os.path.join(path, "value_head.bin"))
        )
        self.loc_transform_layer.load_state_dict(
            torch.load(os.path.join(path, "loc_head.bin"))
        )
        self.std_layer.load_state_dict(torch.load(os.path.join(path, "std_head.bin")))

    def get_device(self):
        _, p = next(self.model.named_parameters())
        return p.device

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, state_batch: List[Tuple[str, List[str], List[str]]]):
        if self.use_cache:
            past_key_values = self.cache
        else:
            past_key_values = None

        if past_key_values is not None:
            try:
                state_batch = [
                    ("", [], state[2][-1].unsqueeze(-1)) for state in state_batch
                ]
            except IndexError as e:
                print(self.use_cache)
                print(past_key_values)
                print(state_batch)

        input_ids, token_type_ids_batch, seqlen, attention_mask = self._build_inputs(
            state_batch
        )

        input_ids = input_ids#.to(self.get_device(), non_blocking=True)
        token_type_ids = token_type_ids_batch#.to(self.get_device(), non_blocking=True)
        attention_mask = attention_mask#.to(self.get_device(), non_blocking=True)

        with autocast() if MIXED_PREC else suppress():
            last_layer_hidden_states, past_key_values = self.model.transformer(
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
            ).to(self.get_device(), non_blocking=True)
            features = last_layer_hidden_states.gather(1, last_feature_ids).squeeze(1)
            values = self.value_head(features)
            means = features + self.loc_transform_layer(features)
            stds = F.relu(self.std_layer(features))
        stds = stds + 1e-5

        if self.use_cache:
            self.cache = past_key_values
        
        diag_emb = stds.diag_embed()
        distr = MultivariateNormal(
                means, diag_emb
            )
        return {
            "action_distribution": distr,
            "values": values.squeeze(-1),
        }

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.clear_cache()
        self.use_cache = False

    def decode(self, hidden_states):
        return torch.argmax(
            self.model.lm_head(
                hidden_states
            ).detach(), dim=-1
        )

    def _build_inputs(self, state_batch: List[Tuple[str, List[str], torch.Tensor]]):
        token_type_batch = []
        input_words = []
        seqlen = []
        utterance_batch_list = [state[2] for state in state_batch]
        persona_batch = [state[0] for state in state_batch]

        tensor_tuple = self._prepare_persona_batch(persona_batch)

        history_replies_num = [len(state[1]) for state in state_batch]
        history_batch = [turn for state in state_batch for turn in state[1]]
        tensor_tuple = self._append_history_batch(
            tensor_tuple, history_batch, history_replies_num
        )
        tensor_tuple = self._append_utterance_batch(tensor_tuple, utterance_batch_list)

        input_ids, attention_mask, token_type_ids_batch, lengths = tensor_tuple
        return input_ids, token_type_ids_batch, lengths, attention_mask

    def _prepare_persona_batch(self, persona_batch):
        if all(persona_batch):
            persona_batch_outp = self.tokenizer(
                persona_batch,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
                add_special_tokens=True,
                return_attention_mask=True,
                return_length=True,
            )
            persona_batch_ids = persona_batch_outp["input_ids"].pin_memory()
            persona_batch_mask = persona_batch_outp["attention_mask"].pin_memory()
            token_types_persona = torch.zeros_like(persona_batch_ids).pin_memory()
        else:
            persona_batch_ids = torch.empty(len(persona_batch), 0, dtype=torch.long)
            persona_batch_mask = torch.empty(len(persona_batch), 0, dtype=torch.long)
            token_types_persona = torch.empty(len(persona_batch), 0, dtype=torch.long)
        return persona_batch_ids, persona_batch_mask, token_types_persona

    def _append_history_batch(self, tensor_tuple, history_batch, history_replies_num):
        (persona_batch_ids, persona_batch_mask, token_types_persona,) = tensor_tuple
        if history_batch:
            history_batch_outp = self.tokenizer(
                history_batch,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=8,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            history_batch_ids = history_batch_outp["input_ids"].pin_memory()
            history_batch_mask = history_batch_outp["attention_mask"].pin_memory()

            (
                history_token_ids,
                history_mask,
                history_type_ids,
            ) = self._format_history_tensors(
                history_batch_ids, history_batch_mask, history_replies_num
            )

            history_token_ids = torch.cat([persona_batch_ids, history_token_ids], dim=1)
            history_mask = torch.cat([persona_batch_mask, history_mask], dim=1)
            history_type_ids = torch.cat([token_types_persona, history_type_ids], dim=1)
        else:
            history_token_ids = persona_batch_ids
            history_mask = persona_batch_mask
            history_type_ids = token_types_persona

        return history_token_ids, history_mask, history_type_ids

    def _format_history_tensors(
        self, history_batch_ids, history_batch_mask, history_replies_num
    ):
        history_batch_ids_list = []
        history_batch_mask_list = []
        history_batch_token_type_list = []
        num_sum = 0
        for num in history_replies_num:
            history_row_ids = history_batch_ids[num_sum:num, :]

            history_row_ids_flat = history_row_ids.view([-1])
            history_row_mask = history_batch_mask[num_sum:num, :].view([-1])

            history_batch_ids_list.append(history_row_ids_flat)
            history_batch_mask_list.append(history_row_mask)

            history_types_ones = torch.ones_like(history_row_ids)
            history_types_zeros = torch.zeros_like(history_row_ids)
            history_types = (
                torch.where(
                    (torch.arange(0, num) % 2 == 0)
                    .unsqueeze(-1)
                    .expand_as(history_row_ids),
                    history_types_ones,
                    history_types_zeros,
                )
                .pin_memory()
                .view(-1)
            )

            history_batch_token_type_list.append(history_types)

        history_token_ids = pad_sequence(
            history_batch_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id,
        )
        history_mask = pad_sequence(
            history_batch_mask_list, batch_first=True, padding_value=0.0
        )
        history_type_ids = pad_sequence(
            history_batch_token_type_list, batch_first=True, padding_value=0.0
        )
        return history_token_ids, history_mask, history_type_ids

    def _append_utterance_batch(self, tensor_tuple, utterance_batch_list):
        utterance_attention_list = [
            torch.ones_like(utt) for utt in utterance_batch_list
        ]

        history_token_ids, history_mask, history_type_ids = tensor_tuple
        if any([el.nelement() != 0 for el in utterance_batch_list]):
            lengths = [
                history_type_ids.shape[1] + el.shape[0] for el in utterance_batch_list
            ]
            utterance = pad_sequence(
                utterance_batch_list,
                batch_first=True,
                padding_value=self.tokenizer.eos_token_id,
            )#.pin_memory()
            utterance_attention = pad_sequence(
                utterance_attention_list, batch_first=True, padding_value=0,
            )
            token_types_utterance = torch.zeros_like(utterance)

            input_ids = torch.cat([
                history_token_ids.to(self.get_device(), non_blocking=True),
                utterance
            ], dim=1)
            token_type_ids_batch = torch.cat(
                [history_type_ids.to(self.get_device(), non_blocking=True), token_types_utterance],
                dim=1
            )
            attention_mask = torch.cat([
                history_mask.to(self.get_device(), non_blocking=True),
                utterance_attention
            ], dim=1)
        else:
            lengths = [history_type_ids.shape[1] for el in utterance_batch_list]
            input_ids = history_token_ids.to(self.get_device(), non_blocking=True)
            token_type_ids_batch = history_type_ids.to(self.get_device(), non_blocking=True)
            attention_mask = history_mask.to(self.get_device(), non_blocking=True)

        return (input_ids, attention_mask, token_type_ids_batch, lengths)

    def reset_noise(self):
        pass

    def clear_cache(self):
        del self.cache
        self.cache = None
