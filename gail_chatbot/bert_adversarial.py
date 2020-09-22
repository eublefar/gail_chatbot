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
    def __init__(self, lr=3e-5, mixed_precision=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2"
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-10)
        if MIXED_PREC:
            self.scaler = GradScaler()

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def forward_contrastive(self, dialogs_gen, dialogs_pos, sub_batch=8, backprop=True):

        dialogs_gen = [(dialog[0], dialog[1][:-1]) for dialog in dialogs_gen]
        dialogs_pos = [(dialog[0], dialog[1][:-1]) for dialog in dialogs_pos]
        if dialogs_pos != dialogs_gen:
            print("dialogs_pos", dialogs_pos)
            print("dialogs_gen", dialogs_gen)
            raise RuntimeError("Paired dialog contexts are different")

        token_ids, token_type_ids, attention_mask, position_ids = self._build_inputs(
            [*dialogs_gen, *dialogs_pos]
        )
        token_ids_gen = token_ids.narrow(0, 0, len(dialogs_gen))
        token_ids_pos = token_ids.narrow(0, len(dialogs_gen), len(dialogs_pos))

        token_type_ids_gen = token_type_ids.narrow(0, 0, len(dialogs_gen))
        token_type_ids_pos = token_type_ids.narrow(
            0, len(dialogs_gen), len(dialogs_pos)
        )

        attention_mask_gen = attention_mask.narrow(0, 0, len(dialogs_gen))
        attention_mask_pos = attention_mask.narrow(
            0, len(dialogs_gen), len(dialogs_pos)
        )

        position_ids_gen = position_ids.narrow(0, 0, len(dialogs_gen))
        position_ids_pos = position_ids.narrow(0, len(dialogs_gen), len(dialogs_pos))

        #         print("Bert ", token_ids.shape)
        loss_return, probs_return = 0, []
        iters = token_ids_gen.shape[0] // sub_batch + int(
            (token_ids_gen.shape[0] % sub_batch) != 0
        )

        for i in range(iters):
            lower = i * sub_batch
            upper = (i + 1) * sub_batch
            with autocast() if MIXED_PREC else suppress():

                ids_gen = token_ids_gen[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                mask_gen = attention_mask_gen[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                types_gen = token_type_ids_gen[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                positions_gen = position_ids_gen[lower:upper].to(
                    self.get_device(), non_blocking=False
                )

                ids_pos = token_ids_pos[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                mask_pos = attention_mask_pos[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                types_pos = token_type_ids_pos[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                positions_pos = position_ids_pos[lower:upper].to(
                    self.get_device(), non_blocking=False
                )

                outp_gen = self.model(
                    input_ids=ids_gen,
                    attention_mask=mask_gen,
                    token_type_ids=types_gen,
                    position_ids=positions_gen,
                )

                outp_pos = self.model(
                    input_ids=ids_pos,
                    attention_mask=mask_pos,
                    token_type_ids=types_pos,
                    position_ids=positions_pos,
                )
                logits = torch.stack([outp_gen[0][:, 1], outp_pos[0][:, 1]], dim=1)
                probs = torch.softmax(logits, dim=1)
                labels = torch.stack(
                    [
                        torch.zeros_like(outp_gen[0][:, 1]),
                        torch.ones_like(outp_pos[0][:, 1]),
                    ],
                    dim=1,
                )
                loss = torch.nn.functional.binary_cross_entropy(probs, labels)
                if backprop:
                    (self.scaler.scale(loss / iters)).backward()

                loss_return += (loss / iters).cpu().item()
                probs_return.append(probs)
        return loss, torch.cat(probs_return, dim=0)

    def get_device(self) -> Union[int, str]:
        _, p = next(self.model.named_parameters())
        return p.device

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
        persona_batch_mask = persona_batch_outp["attention_mask"].bool().pin_memory()
        persona_batch_list = [
            persona[persona_batch_mask[i]]
            for i, persona in enumerate(persona_batch_ids)
        ]
        token_types_persona_list = [
            torch.zeros_like(persona).pin_memory() for persona in persona_batch_list
        ]
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
        history_batch_mask = history_batch_outp["attention_mask"].bool().pin_memory()

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
                history_row_ids = history_batch_ids[num_sum : num_sum + num, :]
                history_row_mask = history_batch_mask[num_sum : num_sum + num, :].view(
                    [-1]
                )
                history_row_ids_flat = history_row_ids.view([-1])[history_row_mask]
                history_size = history_row_mask.sum()

            history_batch_ids_list.append(
                torch.cat([persona_batch_list[i], history_row_ids_flat])
            )
            history_batch_mask_list.append(torch.ones_like(history_batch_ids_list[i]))

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
            )[history_row_mask]

            history_batch_token_type_list.append(
                torch.cat([token_types_persona_list[i], history_types])
            )
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

        return (
            history_token_ids,
            history_type_ids,
            history_mask,
            history_mask.cumsum(dim=1) - 1,
        )

