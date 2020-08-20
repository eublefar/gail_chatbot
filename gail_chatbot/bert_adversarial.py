from typing import List, Tuple, Dict, Union
import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import chain
from contextlib import suppress

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
        preprocessed_dialogs = [
            self._build_inputs_dialogs(persona, history) for persona, history in dialogs
        ]
        token_ids, token_type_ids, attention_mask = self._prepare_tensors(
            preprocessed_dialogs
        )
        loss, logits, hidden_states = [], [], []
        for i in range(
            token_ids.shape[0] // sub_batch + int(token_ids.shape[0] % sub_batch != 0)
        ):
            lower = i * sub_batch
            upper = (i + 1) * sub_batch
            with autocast() if MIXED_PREC else suppress():
                outp = self.model(
                    input_ids=token_ids[lower:upper],
                    attention_mask=attention_mask[lower:upper],
                    token_type_ids=token_type_ids[lower:upper],
                    labels=labels[lower:upper] if labels is not None else None,
                    output_hidden_states=True,
                )
            if labels is not None:

                (
                    self.scaler.scale(outp[0]).backward()
                    if MIXED_PREC
                    else outp[0].backward()
                )
                loss.append(outp[0])
                logits.append(outp[1])
                hidden_states.append(outp[2][-1])
            else:
                logits.append(outp[0])
                hidden_states.append(outp[1][-1])

        return (
            torch.stack(loss).mean() if loss is not None else None,
            torch.cat(logits, dim=0),
            torch.cat(hidden_states, dim=0),
        )

    def _build_inputs_dialogs(self, persona, history):
        persona = self.tokenizer.tokenize(persona)
        persona = [[self.tokenizer.cls_token] + persona + [self.tokenizer.sep_token]]
        history_seq = [
            self.tokenizer.tokenize(s) + [self.tokenizer.sep_token]
            for i, s in enumerate(history)
        ]
        if len(list(chain(*persona, *history_seq))) >= 512:
            persona_len = len(persona[0])
            history_lengths = [len(seq) for seq in history_seq]
            while (sum(history_lengths) + persona_len) >= 512:
                history_lengths = history_lengths[1:]
                history_seq = history_seq[1:]
        sequence = persona + history_seq
        words = list(chain(*sequence))
        segments = [i % 2 for i, s in enumerate(sequence) for _ in s]
        return words, segments

    def _prepare_tensors(
        self, words_segments: List[Tuple[List[str], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maxlen = max([len(word_seg[0]) for word_seg in words_segments])
        token_ids = []
        token_type_ids = []
        attention_masks = []
        for words, segments in words_segments:
            pad_num = maxlen - len(words)
            attention_masks.append([1] * len(words) + [0] * pad_num)
            words.extend([self.tokenizer.pad_token] * pad_num)
            segments.extend([segments[-1]] * pad_num)
            token_type_ids.append(segments)
            token_ids.append(self.tokenizer.convert_tokens_to_ids(words))
        return (
            torch.LongTensor(token_ids).to(self.get_device()),
            torch.LongTensor(token_type_ids).to(self.get_device()),
            torch.LongTensor(attention_masks).to(self.get_device()),
        )

