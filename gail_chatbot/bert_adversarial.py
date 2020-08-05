from typing import List, Tuple, Dict, Union
import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import chain


class BertAdversarial(torch.nn.Module):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit_batch(
        self, dialogs: List[Tuple[str, List[str]]], labels: List[int], sub_batch: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.LongTensor(labels).to(self.get_device())
        loss, logits, hidden_states = self(dialogs, labels, sub_batch)

#         clip_grad_norm_(self.model.parameters(), 20)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return logits, hidden_states, torch.stack(loss).mean()

    def get_device(self) -> Union[int, str]:
        _, p = next(self.model.named_parameters())
        return p.device

    def forward(
        self, dialogs: List[Tuple[str, List[str]]], labels: torch.LongTensor, sub_batch: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preprocessed_dialogs = [
            self._build_inputs_dialogs(persona, history) for persona, history in dialogs
        ]
        token_ids, token_type_ids, attention_mask = self._prepare_tensors(
            preprocessed_dialogs
        )
        loss, logits, hidden_states = [], [], []
        for i in range(token_ids.shape[0]//sub_batch):
            lower = i*sub_batch
            upper = (i + 1)*sub_batch
            outp = self.model(
                input_ids=token_ids[lower:upper],
                attention_mask=attention_mask[lower:upper],
                token_type_ids=token_type_ids[lower:upper],
                labels=labels[lower:upper],
                output_hidden_states=True,
            )
            outp[0].backward()
            loss.append(outp[0])
            logits.append(outp[1])
            hidden_states.append(outp[2][-1])
        return (
            loss,
            torch.cat(logits, dim=0),
            torch.cat(hidden_states, dim=0)
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

