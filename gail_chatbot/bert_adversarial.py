from typing import List, Tuple, Dict, Union
import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import chain
from contextlib import suppress
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.cuda.amp import autocast, GradScaler

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False


class BertAdversarial(torch.nn.Module):
    def __init__(self, lr=1e-6, mixed_precision=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2"
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if MIXED_PREC:
            self.scaler = GradScaler()

    def fit_batch(
        self,
        dialogs: List[Tuple[str, List[str]]],
        labels: List[int],
        sub_batch: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.LongTensor(labels).to(self.get_device())
        loss, logits, hidden_states = self(dialogs, labels, sub_batch)

        if MIXED_PREC:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), 20)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            clip_grad_norm_(self.model.parameters(), 20)
            self.optimizer.step()
        self.optimizer.zero_grad()
        return logits, hidden_states, loss

    def get_device(self) -> Union[int, str]:
        _, p = next(self.model.named_parameters())
        return p.device

    def forward(
        self,
        dialogs: List[Tuple[str, List[str]]],
        labels: torch.LongTensor = None,
        sub_batch: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids, token_type_ids, attention_mask, position_ids = self._build_inputs(
            dialogs
        )

        loss, logits, hidden_states = [], [], []
        iters = token_ids.shape[0] // sub_batch + int(
            (token_ids.shape[0] % sub_batch) != 0
        )
        for i in range(iters):
            lower = i * sub_batch
            upper = (i + 1) * sub_batch
            with autocast() if MIXED_PREC else suppress():
                ids = token_ids[lower:upper].to(self.get_device(), non_blocking=False)
                mask = attention_mask[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                types = token_type_ids[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                positions = position_ids[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                outp = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=types,
                    position_ids=positions,
                    labels=labels[lower:upper] if labels is not None else None,
                    output_hidden_states=True,
                )
                del ids, mask, types, positions
            if labels is not None:

                (
                    (self.scaler.scale(outp[0] / iters)).backward()
                    if MIXED_PREC
                    else (outp[0] / iters).backward()
                )
                loss.append(outp[0].cpu().detach())
                logits.append(outp[1].cpu().detach())
            #                 hidden_states.append(outp[2][-1].detach())
            else:
                logits.append(outp[0].cpu().detach())
            #                 hidden_states.append(outp[1][-1].cpu().detach())
            del outp

        return (
            *([torch.stack(loss).mean()] if labels is not None else []),
            torch.cat(logits, dim=0),
            #             torch.cat(hidden_states, dim=0),
        )

    def _build_inputs(self, dialogs):
        result = []
        persona_batch = [dialog[0] for dialog in dialogs]
        persona_batch_outp = self.tokenizer(
            persona_batch,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=1,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        persona_batch_ids = persona_batch_outp["input_ids"].pin_memory()
        persona_batch_mask = persona_batch_outp["attention_mask"].pin_memory()
        token_types_persona = torch.zeros_like(persona_batch_ids).pin_memory()
        persona_sizes = persona_batch_mask.sum(dim=1)

        history_batch = [
            turn + self.tokenizer.sep_token for dialog in dialogs for turn in dialog[1]
        ]
        history_replies_num = [len(dialog[1]) for dialog in dialogs]
        history_batch_outp = self.tokenizer(
            history_batch,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=1,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        history_batch_ids = history_batch_outp["input_ids"].pin_memory()
        history_batch_mask = history_batch_outp["attention_mask"].pin_memory()

        history_batch_ids_list = []
        history_batch_mask_list = []
        history_batch_token_type_list = []
        num_sum = 0
        for i, num in enumerate(history_replies_num):
            history_row_ids = history_batch_ids[num_sum : num_sum + num, :]
            history_row_ids_flat = history_row_ids.view([-1])
            history_row_mask = history_batch_mask[num_sum : num_sum + num, :].view([-1])

            history_size = history_row_mask.sum()

            while (history_size + persona_sizes[i]) > 400:
                num_sum += 1
                num -= 1
                history_row_ids = history_batch_ids[num_sum : num_sum + num, :]
                history_row_ids_flat = history_row_ids.view([-1])
                history_row_mask = history_batch_mask[num_sum : num_sum + num, :].view(
                    [-1]
                )
                history_size = history_row_mask.sum()
            print("Bert ", history_size + persona_sizes[i])
            history_batch_ids_list.append(history_row_ids_flat)
            history_batch_mask_list.append(history_row_mask)

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
                )
            except Exception as e:
                print(num)
                print(num_sum)
                raise e

            history_batch_token_type_list.append(history_types)
            num_sum += num

        history_token_ids = pad_sequence(
            history_batch_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        history_mask = pad_sequence(
            history_batch_mask_list, batch_first=True, padding_value=0.0
        )
        history_type_ids = pad_sequence(
            history_batch_token_type_list, batch_first=True, padding_value=0.0
        )

        input_ids = torch.cat([persona_batch_ids, history_token_ids], dim=1)
        token_type_ids_batch = torch.cat([token_types_persona, history_type_ids], dim=1)
        attention_mask = torch.cat([persona_batch_mask, history_mask], dim=1)

        return (
            input_ids,
            token_type_ids_batch,
            attention_mask,
            attention_mask.cumsum(dim=1) - 1,
        )

