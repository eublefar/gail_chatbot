from typing import List, Tuple, Dict, Union
import os
import gc
import numpy as np
import torch
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    set_seed,
)
from gym_loop.policies.base_policy import BasePolicy
from contextlib import suppress
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.cuda.amp import autocast

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False


class GptPolicy(torch.nn.Module, BasePolicy):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.temp = 1
        self.block_eos = False
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.add_special_tokens(
            {
                "pad_token": self.tokenizer.eos_token,
                "sep_token": self.tokenizer.eos_token,
            }
        )

        self.model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium").eval()
        #         self.loc_transform_layer = torch.nn.Linear(768, 768)

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 256), torch.nn.ReLU(True), torch.nn.Linear(256, 1)
        )
        self.cache = None
        self.use_cache = True

    def save(self, path: str):
        self.model.save_pretrained(os.path.join(path, "model.bin"))
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.bin"))

    def load(self, path: str):
        self.model = self.model.from_pretrained(os.path.join(path, "model.bin")).eval()
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
            past_key_values = self.cache
            self.cache = None
        else:
            past_key_values = None
        if past_key_values is not None:
            try:
                state_batch = [
                    ("", [], torch.LongTensor([state[2][-1]],)) for state in state_batch
                ]
            except IndexError as e:
                # print(self.use_cache)
                print(past_key_values)
                print(state_batch)
                raise e

        (
            input_ids,
            token_type_ids_batch,
            seqlen,
            attention_mask,
            position_ids,
        ) = self._build_inputs(state_batch)
        input_ids = input_ids.to(self.get_device(), non_blocking=True)
        token_type_ids = token_type_ids_batch.to(self.get_device(), non_blocking=True)
        attention_mask = attention_mask.to(self.get_device(), non_blocking=True)

        with autocast() if MIXED_PREC else suppress():
            outp = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            logits, past_key_values, hidden_states = (
                outp["logits"],
                outp["past_key_values"],
                outp["hidden_states"],
            )
            seq_last_id = (torch.LongTensor(seqlen) - 1).view([-1, 1, 1]).cuda()
            logits = logits.gather(dim=1, index=seq_last_id.expand_as(logits))[:, 0, :]
            features = hidden_states[-1].gather(
                dim=1, index=seq_last_id.expand_as(hidden_states[-1])
            )[:, 0, :]
            values = self.value_head(features.squeeze(1))
            distr = Categorical(logits=logits)

        if self.use_cache:
            self.cache = past_key_values

        return {
            "action_distribution": distr,
            "values": values.squeeze(-1),
        }

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.clear_cache()
        self.use_cache = False

    def decode(self, hidden_states, topk=50, temp=None):
        return hidden_states.to("cpu")

    def _build_inputs(self, state_batch: List[Tuple[str, List[str], torch.Tensor]]):

        utterance_batch_list = [state[2] for state in state_batch]
        persona_batch = [state[0] for state in state_batch]

        tensor_tuple = self._prepare_persona_batch(persona_batch)

        history_replies_num = [len(state[1]) for state in state_batch]
        history_batch = [
            turn + self.tokenizer.eos_token
            for state in state_batch
            for turn in state[1]
        ]
        tensor_tuple = self._append_history_batch(
            tensor_tuple, history_batch, history_replies_num
        )
        tensor_tuple = self._append_utterance_batch(tensor_tuple, utterance_batch_list)

        (
            input_ids,
            attention_mask,
            token_type_ids_batch,
            lengths,
            position_ids,
        ) = tensor_tuple
        return input_ids, token_type_ids_batch, lengths, attention_mask, position_ids

    def _prepare_persona_batch(self, persona_batch):
        if all(persona_batch):
            persona_batch = [
                persona + self.tokenizer.eos_token for persona in persona_batch
            ]
            persona_batch_outp = self.tokenizer(
                persona_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                return_attention_mask=True,
                return_length=True,
            )
            persona_batch_ids = persona_batch_outp["input_ids"].pin_memory()
            persona_batch_mask = (
                persona_batch_outp["attention_mask"].bool().pin_memory()
            )
            token_types_persona = torch.zeros_like(persona_batch_ids).pin_memory()
        else:
            persona_batch_ids = torch.empty(len(persona_batch), 0, dtype=torch.long)
            persona_batch_mask = torch.empty(len(persona_batch), 0, dtype=torch.long)
            token_types_persona = torch.empty(len(persona_batch), 0, dtype=torch.long)

        return persona_batch_ids, persona_batch_mask, token_types_persona

    def _append_history_batch(self, tensor_tuple, history_batch, history_replies_num):
        (persona_batch_ids, persona_batch_mask, token_types_persona,) = tensor_tuple

        persona_sizes = persona_batch_mask.sum(dim=1)
        if history_batch:

            persona_batch_list = [
                persona[persona_batch_mask[i]]
                for i, persona in enumerate(persona_batch_ids)
            ]
            token_types_persona_list = [
                torch.zeros_like(persona).pin_memory() for persona in persona_batch_list
            ]

            history_batch_outp = self.tokenizer(
                history_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            history_batch_ids = history_batch_outp["input_ids"].pin_memory()
            history_batch_mask = (
                history_batch_outp["attention_mask"].bool().pin_memory()
            )

            (
                history_token_ids,
                history_mask,
                history_type_ids,
            ) = self._format_history_tensors(
                history_batch_ids,
                history_batch_mask,
                history_replies_num,
                persona_batch_list,
                persona_sizes,
                token_types_persona_list,
            )
        else:
            history_token_ids = persona_batch_ids
            history_mask = persona_batch_mask
            history_type_ids = token_types_persona

        return history_token_ids, history_mask, history_type_ids

    def _format_history_tensors(
        self,
        history_batch_ids,
        history_batch_mask,
        history_replies_num,
        persona_batch_list,
        persona_sizes,
        token_types_persona_list,
    ):
        history_batch_ids_list = []
        history_batch_mask_list = []
        history_batch_token_type_list = []
        num_sum = 0
        for i, num in enumerate(history_replies_num):
            history_row_ids = history_batch_ids[num_sum : num_sum + num, :]
            history_row_mask = history_batch_mask[num_sum : num_sum + num, :].view([-1])
            history_row_ids_flat = history_row_ids.view([-1])[history_row_mask]

            history_size = history_row_mask.sum()

            while (history_size + persona_sizes[i]) > 400:
                num_sum += 1
                num -= 1
                history_row_mask = history_batch_mask[num_sum : num_sum + num, :].view(
                    [-1]
                )
                history_row_ids = history_batch_ids[num_sum : num_sum + num, :]
                history_row_ids_flat = history_row_ids.view([-1])[history_row_mask]
                history_size = history_row_mask.sum()

            history_batch_ids_list.append(
                torch.cat([persona_batch_list[i], history_row_ids_flat])
            )
            history_batch_mask_list.append(torch.ones_like(history_batch_ids_list[i]))

            history_types_ones = torch.ones_like(history_row_ids)
            history_types_zeros = torch.zeros_like(history_row_ids)
            try:
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
                )[history_row_mask]
            except Exception as e:
                raise e

            history_batch_token_type_list.append(
                torch.cat([token_types_persona_list[i], history_types])
            )
            num_sum += num
        history_token_ids = history_batch_ids_list
        history_mask = history_batch_mask_list
        history_type_ids = history_batch_token_type_list
        return history_token_ids, history_mask, history_type_ids

    def _append_utterance_batch(self, tensor_tuple, utterance_batch_list):
        utterance_attention_list = [
            torch.ones_like(utt) for utt in utterance_batch_list
        ]
        lengths = []
        ids_list = []
        mask_list = []
        type_list = []
        for i, tensor_tuple_row in enumerate(zip(*tensor_tuple)):
            history_token_ids, history_mask, history_type_ids = tensor_tuple_row

            if utterance_batch_list[i].nelement() != 0:
                lengths.append(
                    history_type_ids.shape[0] + utterance_batch_list[i].shape[0]
                )

                ids_list.append(
                    torch.cat(
                        [
                            history_token_ids,  # .to(self.get_device(), non_blocking=True),
                            utterance_batch_list[i],
                        ],
                        dim=0,
                    ).pin_memory()
                )
                type_list.append(
                    torch.cat(
                        [
                            history_type_ids,  # .to(self.get_device(), non_blocking=True),
                            torch.zeros_like(utterance_batch_list[i]),
                        ],
                        dim=0,
                    ).pin_memory()
                )
                mask_list.append(
                    torch.cat(
                        [
                            history_mask,  # .to(self.get_device(), non_blocking=True),
                            utterance_attention_list[i],
                        ],
                        dim=0,
                    ).pin_memory()
                )
            else:
                lengths.append(history_type_ids.shape[0])
                ids_list.append(history_token_ids.pin_memory())
                type_list.append(history_type_ids.pin_memory())
                mask_list.append(history_mask.pin_memory())
        input_ids = pad_sequence(
            ids_list, batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        token_type_ids_batch = pad_sequence(
            type_list, batch_first=True, padding_value=0
        )
        attention_mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
        return (
            input_ids,
            attention_mask,
            token_type_ids_batch,
            lengths,
            None,  #             attention_mask.cumsum(dim=1) - 1,
        )

    def reset_noise(self):
        pass

    def clear_cache(self):
        del self.cache
        self.cache = None
