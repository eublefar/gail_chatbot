from typing import List, Tuple, Dict, Union
import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from itertools import chain
from contextlib import suppress
from torch.nn.utils.rnn import pad_sequence

try:
    from torch.cuda.amp import autocast, GradScaler

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False


class BARTSimple(torch.nn.Module):
    def __init__(
        self,
        lr=1e-5,
        mixed_precision=True,
        special_tokens=None,
        self_speaker_token="<speaker_self>",
        other_speaker_token="<speaker_other>",
        emote_num=23,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        # self.tokenizer.add_special_tokens(
        #     {
        #         "pad_token": self.tokenizer.eos_token,
        #         "sep_token": self.tokenizer.eos_token,
        #     }
        # )
        self.other_speaker_token = other_speaker_token
        self.self_speaker_token = self_speaker_token
        if special_tokens is not None:
            self.tokenizer.add_tokens(
                special_tokens + [other_speaker_token, self_speaker_token]
            )

        self.model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large"
        ).train()

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.emote_head = torch.nn.Linear(self.model.config.d_model, emote_num)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-10)
        self.ignore_token_id = -100
        self.lr = lr
        if MIXED_PREC:
            self.scaler = GradScaler()

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def fit_batch(
        self,
        dialogs: List[Tuple[str, List[str]]],
        emote_labels: torch.LongTensor,
        sub_batch: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        loss, logits, labels = self(dialogs, emote_labels, sub_batch)

        if MIXED_PREC:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), 20)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            clip_grad_norm_(self.model.parameters(), 20)
            self.optimizer.step()
        self.optimizer.zero_grad()
        return (
            logits,
            labels,
            loss,
        )

    def get_device(self) -> Union[int, str]:
        _, p = next(self.model.named_parameters())
        return p.device

    def forward(
        self,
        dialogs: List[Tuple[str, List[str]]],
        emote_labels: torch.LongTensor,
        sub_batch: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            token_ids,
            attention_mask,
            position_ids,
            type_ids,
            decoder_ids,
            labels,
        ) = self._build_inputs(dialogs)

        loss, logits = [], []
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
                positions = position_ids[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                type_ids_el = type_ids[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                decoder_el = decoder_ids[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                labels_el = labels[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                emote_labels_el = emote_labels[lower:upper].to(
                    self.get_device(), non_blocking=False
                )
                outp = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=type_ids_el,
                    decoder_input_ids=decoder_el,
                    labels=labels_el,
                    output_hidden_states=True,
                    return_dict=True,
                )
                emote_outp = self.emote_head(outp["encoder_hidden_states"][-1][:, 0, :])
                emote_loss = torch.nn.functional.cross_entropy(
                    emote_outp, emote_labels_el
                )

                outp["loss"] += emote_loss

                del ids, mask, decoder_el, positions, emote_labels_el
            if labels is not None:

                (
                    (self.scaler.scale(outp["loss"] / iters)).backward()
                    if MIXED_PREC
                    else (outp["loss"] / iters).backward()
                )
                loss.append(outp["loss"].cpu().detach())
                logits = outp["logits"].cpu().detach()
            else:
                logits.append(outp["logits"].cpu().detach())
            del outp

        return (
            *([torch.stack(loss).mean()] if labels is not None else []),
            logits.float(),
            labels_el.cpu(),  # pyright: reportUnboundVariable=false
        )

    def _build_inputs(self, dialogs):
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
        persona_sizes = persona_batch_mask.sum(dim=1)

        token_types_persona_list = [
            torch.zeros_like(persona).pin_memory() for persona in persona_batch_list
        ]

        #         print(dialogs)
        history_batch = [turn for dialog in dialogs for turn in dialog[1]]
        history_replies_num = [len(dialog[1]) for dialog in dialogs]
        history_batch_outp = self.tokenizer(
            history_batch,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=1,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        history_batch_ids = (
            history_batch_outp["input_ids"].pin_memory()[:, 1:].contiguous()
        )
        history_batch_mask = (
            history_batch_outp["attention_mask"].bool().pin_memory()[:, 1:].contiguous()
        )

        history_batch_ids_list = []
        history_batch_mask_list = []
        history_batch_token_type_list = []

        labels_list = []
        decoder_ids_list = []

        num_sum = 0
        for i, num in enumerate(history_replies_num):
            history_row_ids = history_batch_ids[num_sum : num_sum + num - 1, :]
            history_row_mask = history_batch_mask[num_sum : num_sum + num - 1, :].view(
                [-1]
            )
            history_row_ids_flat = history_row_ids.view([-1])[history_row_mask]

            label_mask = history_batch_mask[num_sum + num - 1, :]
            labels = history_batch_ids[num_sum + num - 1, :][label_mask]

            decoder_ids = shift_tokens_right(
                labels.view([1, -1]),
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ).view([-1])

            history_size = history_row_mask.sum()
            while (history_size + persona_sizes[i]) > 512:
                num_sum += 1
                num -= 1

                history_row_ids = history_batch_ids[num_sum : num_sum + num - 1, :]
                history_row_mask = history_batch_mask[
                    num_sum : num_sum + num - 1, :
                ].view([-1])
                history_row_ids_flat = history_row_ids.view([-1])[history_row_mask]

                label_mask = history_batch_mask[num_sum + num - 1, :]
                labels = history_batch_ids[num_sum + num - 1, :][label_mask]

                decoder_ids = shift_tokens_right(
                    labels.view([1, -1]),
                    self.tokenizer.pad_token_id,
                    self.tokenizer.eos_token_id,
                ).view([-1])

                history_size = history_row_mask.sum()

            history_batch_ids_list.append(
                torch.cat([persona_batch_list[i], history_row_ids_flat])
            )
            history_batch_mask_list.append(torch.ones_like(history_batch_ids_list[i]))

            history_types_ones = torch.ones_like(history_row_ids)
            history_types_zeros = torch.zeros_like(history_row_ids)
            history_types = (
                torch.where(
                    (torch.arange(0, num - 1) % 2 == 0)
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

            labels_list.append(labels)
            decoder_ids_list.append(decoder_ids)
            num_sum += num

        history_token_ids = pad_sequence(
            history_batch_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        history_mask = pad_sequence(
            history_batch_mask_list, batch_first=True, padding_value=0.0
        )
        labels_batch = pad_sequence(
            labels_list, batch_first=True, padding_value=self.ignore_token_id
        )
        decoder_ids_batch = pad_sequence(
            decoder_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        history_type_ids = pad_sequence(
            history_batch_token_type_list, batch_first=True, padding_value=0.0
        )
        return (
            history_token_ids,
            history_mask,
            history_mask.cumsum(dim=1) - 1,
            history_type_ids,
            decoder_ids_batch,
            labels_batch,
        )

    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.emote_head.state_dict(), os.path.join(path, "emote_head.bin"))
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.model = self.model.from_pretrained(path).cuda().train()
        self.tokenizer = self.tokenizer.from_pretrained(path)
        self.emote_head.load_state_dict(
            torch.load(os.path.join(path, "emote_head.bin"))
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, eps=1e-10
        )
