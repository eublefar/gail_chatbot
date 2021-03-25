from typing import List, Tuple
import os
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from gym_loop.policies.base_policy import BasePolicy
from contextlib import suppress
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F

try:
    from torch.cuda.amp import autocast

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = self.linear6(x2)

        return x1, x2


class BartPolicy(torch.nn.Module, BasePolicy):
    def __init__(
        self,
        path=None,
        special_tokens=None,
        self_speaker_token="<speaker_self>",
        other_speaker_token="<speaker_other>",
        emote_num=23,
        min_length=10,
        alpha=1,
        *args,
        **kwargs
    ):
        torch.nn.Module.__init__(self)
        self.other_speaker_token = other_speaker_token
        self.self_speaker_token = self_speaker_token
        self.alpha = alpha

        self.emote_head = torch.nn.Linear(1024, emote_num)

        self.q_function = QNetwork(1024, 1024, 1024)

        if path is not None:
            self.load(path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
            self.model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large"
            ).train()
            if special_tokens is not None:
                self.tokenizer.add_tokens(
                    special_tokens + [other_speaker_token, self_speaker_token]
                )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.cache = None
        self.use_cache = True
        self.min_length = min_length
        self.avg_entropy = 0
        self.cnt = 0

    def save(self, path: str):
        self.model.save_pretrained(os.path.join(path, "model.bin"))
        self.tokenizer.save_pretrained(os.path.join(path, "model.bin"))
        torch.save(self.emote_head.state_dict(), os.path.join(path, "emote_head.bin"))

    def load(self, path: str):
        self.model = BartForConditionalGeneration.from_pretrained(
            os.path.join(path, "model.bin")
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "model.bin"))
        self.emote_head.load_state_dict(
            torch.load(os.path.join(path, "emote_head.bin"))
        )

    def get_device(self):
        _, p = next(self.model.named_parameters())
        return p.device

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def forward(self, state_batch: List[Tuple[str, List[str], List[int]]], step=None):
        if self.use_cache and self.cache:
            encoder_outputs, past_key_values = self.cache
            self.cache = None
        else:
            past_key_values = None
            encoder_outputs = None
        if past_key_values is not None:
            try:
                state_batch = [
                    ("", [], torch.LongTensor([state[2][-1]])) for state in state_batch
                ]
            except IndexError as e:
                # print(self.use_cache)
                print(past_key_values)
                print(state_batch)
                raise e

        (
            input_ids,
            seqlen,
            type_ids,
            decoder_input_ids,
            history_mask,
        ) = self._build_inputs(state_batch)
        input_ids = input_ids.to(self.get_device(), non_blocking=True)
        type_ids = type_ids.to(self.get_device(), non_blocking=True)
        decoder_input_ids = decoder_input_ids.to(self.get_device(), non_blocking=True)
        history_mask = history_mask.to(self.get_device(), non_blocking=True)
        with autocast() if MIXED_PREC else suppress():
            outp = self.model(
                input_ids,
                attention_mask=history_mask if encoder_outputs is None else None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
                output_attentions=True,
            )
            logits, past_key_values, hidden_states = (
                outp["logits"],
                outp["past_key_values"],
                outp["decoder_hidden_states"],
            )
            seq_last_id = (torch.LongTensor(seqlen) - 1).view([-1, 1, 1]).cuda()
            logits = logits.gather(dim=1, index=seq_last_id.expand_as(logits))[:, 0, :]
            hidden_states = hidden_states.gather(
                dim=1, index=seq_last_id.expand_as(hidden_states)
            )[:, 0, :]

        if self.use_cache:
            self.cache = (
                (
                    outp["encoder_last_hidden_state"],
                    outp["encoder_hidden_states"],
                    outp["encoder_attentions"],
                ),
                past_key_values,
            )
        return logits, hidden_states

    def getV(self, q_value):
        v = self.alpha * torch.log(
            torch.sum(torch.exp(q_value / self.alpha), dim=1, keepdim=True)
        )
        return v

    def choose_action(self, state):
        with torch.no_grad():
            mask = torch.BoolTensor([state_el[2].numel() < 3 for state_el in state]).to(
                self.get_device()
            )
            logits = self.forward(state)
            logits[mask, self.tokenizer.eos_token_id] = float("-inf")
            c = Categorical(logits=logits)
            self.avg_entropy += c.entropy().detach().cpu().mean().item()
            self.cnt += 1
            a = c.sample()
        return a

    def compute_q(self, state_hidden, action):
        action_embs = self.model.encoder.embed_tokens(action)
        return self.q_function(state_hidden, action_embs)

    def get_entropy(self):
        if self.cnt > 0:
            avg = self.avg_entropy / self.cnt
            self.avg_entropy = 0
            self.cnt = 0
            return avg
        else:
            return -1

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False

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

        (input_ids, lengths, type_ids, decoder_ids, mask) = tensor_tuple
        return input_ids, lengths, type_ids, decoder_ids, mask

    def _prepare_persona_batch(self, persona_batch):
        if all(persona_batch):
            persona_batch_outp = self.tokenizer(
                persona_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
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
        (persona_batch_ids, persona_batch_mask, token_types_persona) = tensor_tuple

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
                add_special_tokens=True,
                return_attention_mask=True,
            )
            history_batch_ids = (
                history_batch_outp["input_ids"][:, 1:].contiguous().pin_memory()
            )
            history_batch_mask = (
                history_batch_outp["attention_mask"]
                .bool()[:, 1:]
                .contiguous()
                .pin_memory()
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

            while (history_size + persona_sizes[i]) > 512:
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
        lengths = []
        ids_list = []
        decoder_ids_list = []
        mask = []
        type_ids = tensor_tuple[2]
        for i, tensor_tuple_row in enumerate(zip(*tensor_tuple)):
            history_token_ids, history_mask, _ = tensor_tuple_row
            mask.append(history_mask)
            lengths.append(utterance_batch_list[i].shape[0] + 1)
            ids_list.append(history_token_ids.pin_memory())
            decoder_ids_list.append(
                torch.cat(
                    [
                        torch.LongTensor([self.tokenizer.eos_token_id]),
                        utterance_batch_list[i].pin_memory(),
                    ],
                    dim=0,
                )
            )
        input_ids = pad_sequence(
            ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        decoder_ids = pad_sequence(
            decoder_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        type_ids = pad_sequence(type_ids, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        return (input_ids, lengths, type_ids, decoder_ids, mask)

    def reset_noise(self):
        pass

    def clear_cache(self):
        del self.cache
        self.cache = None
