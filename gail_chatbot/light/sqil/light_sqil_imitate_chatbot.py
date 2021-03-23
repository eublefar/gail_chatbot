"""Main module."""
from typing import Iterable, Dict, Tuple, List, Any

import numpy as np
import torch
import os
import json
import yaml
import json
import pprint

from tensorboardX import SummaryWriter
from copy import deepcopy
from contextlib import suppress

from gail_chatbot.light.sqil.bart_policy import BartPolicy
from gail_chatbot.light.sqil.light_imitate_mixin import LightImitateMixin
from gail_chatbot.light.sqil.light_selfplay_base_mixin import LightSelfplayBaseMixin
from .replay_buffer import ReplayBuffer
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


try:
    from torch.cuda.amp import autocast, GradScaler

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False

torch.set_num_threads(8)
# TODO:
# - implement alpha decay, ili poka net


class LightGailChatbot(LightSelfplayBaseMixin, LightImitateMixin):
    """Chatbot that learns ConvAI task using GAIL
    """

    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        LightSelfplayBaseMixin.__init__(self, opt, shared)
        LightImitateMixin.__init__(self, opt, shared)

        self.id = "GailChatbot"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SUBPATHS = {
            "generator": "generator",
            "generator_target": "generator_target",
        }
        self.opt = opt

        # Neural nets
        self.generator = None
        self.optimizer = None
        self.update_batch_size = 8
        self.gen_sub_batch_size = 8
        if MIXED_PREC:
            self.scaler = GradScaler(enabled=True)

        # Batch sizes
        self.batch_size = opt["batchsize"]  # batch size of gradient update

        # Hyperparameters
        self.maxlen = 120
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 100
        self.episode_checkpoint_num = 100
        self.update_generator = True
        self.gamma = 0.99
        self.alpha = 1
        self.alpha_log_decay = 0.5
        self.min_alpha = 0.01
        self.imitate_reward = 5
        self.target_update_steps = 1000
        self.updates_per_step = 15
        self.gradient_clip_norm = 40
        self.min_replay_size = 15000
        self.lr = 1e-5
        # Counters
        self.gen_episode_num = 0
        self.no_update_step = 0

        # Paths
        self.dialog_dump_path = opt["model_file"] + "_dialogs"
        self.checkpoint_path = os.path.split(opt["model_file"])[0]
        self.pp = pprint.PrettyPrinter(indent=4)

        # Utility
        self.lr_updated = False
        self.metrics = {}
        self.is_eval = False

        if not os.path.isdir(self.dialog_dump_path):
            os.mkdir(self.dialog_dump_path)

        if shared:
            self._create_from_shared(shared)
        else:
            self._create_from_path(opt["model_file"])

    def _create_from_shared(self, shared: Dict[str, Any]):
        self.generator = shared["generator"]
        self.generator_target = shared["generator_target"]
        self.optimizer = shared["optimizer"]
        self.replay_buffer_sample = shared["replay_buffer_sample"]
        self.replay_buffer_expert = shared["replay_buffer_expert"]
        self.overwrite_params = shared["overwrite_params"]
        self.__dict__ = {
            k: v if k not in self.overwrite_params else self.overwrite_params[k]
            for k, v in self.__dict__.items()
        }

    def _create_from_path(self, path: str):
        path, filename = os.path.split(path)
        param_file = os.path.join(path, "parameters.yml")
        if os.path.isfile(param_file):
            with open(os.path.join(path, "parameters.yml")) as param_file:
                overwrite_params = yaml.load(param_file.read())
        else:
            with open(os.path.join(path, "parameters.yml"), "w") as param_file:
                overwrite_params = self._get_default_params()
                param_file.write(yaml.dump(overwrite_params))
        self._construct_generator(path)
        self.writer = SummaryWriter(os.path.join(path, filename) + ".tensorboard")
        self.__dict__ = {
            k: v if k not in overwrite_params else overwrite_params[k]
            for k, v in self.__dict__.items()
        }
        self.overwrite_params = overwrite_params
        self.replay_buffer_expert = ReplayBuffer(10000000, self.batch_size)
        self.replay_buffer_sample = ReplayBuffer(10000000, self.batch_size)

    def _get_default_params(self):
        return {
            "maxlen": 120,
            "gen_sub_batch_size": 32,
            "update_batch_size": 16,
            "episode_num_log": 1,
            "episode_num_dialog_dump": 100,
            "episode_checkpoint_num": 200,
            "update_generator": True,
            "gamma": 0.99,
            "alpha": 1,
            "alpha_log_decay": 0.5,
            "min_alpha": 0.01,
            "imitate_reward": 1,
            "target_update_steps": 1000,
            "updates_per_step": 15,
            "gradient_clip_norm": 40,
            "min_replay_size": 15000,
            "lr": 1e-5,
        }

    def _construct_generator(self, path: str):
        gen_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        gen_target_dir = os.path.join(path, self.MODEL_SUBPATHS["generator_target"])
        if os.path.isdir(gen_dir):
            self.generator = BartPolicy(gen_dir, special_tokens=self.ctx_tokens).eval()
            self.generator_target = BartPolicy(
                gen_target_dir, special_tokens=self.ctx_tokens
            ).eval()
        else:
            self.generator = BartPolicy(special_tokens=self.ctx_tokens).eval()
            self.generator_target = BartPolicy(special_tokens=self.ctx_tokens).eval()
        self.generator_target.to(self.device)
        self.generator.to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)

    def share(self) -> Dict[str, Any]:
        return dict(
            **super().share(),
            **{
                "generator": self.generator,
                "generator_target": self.generator_target,
                "optimizer": self.optimizer,
                "overwrite_params": self.overwrite_params,
                "replay_buffer_expert": self.replay_buffer_expert,
                "replay_buffer_sample": self.replay_buffer_sample,
            }
        )

    def batch_imitate(self, dialogs):
        done = np.zeros([len(dialogs)], dtype=bool)
        actions = [
            torch.LongTensor(
                self.generator.tokenizer.encode(dialog[1][-1], add_special_tokens=False)
                + [self.generator.tokenizer.eos_token_id]
            )
            for dialog in dialogs
        ]
        max_len = max([action.shape[0] for action in actions])
        dialogs = [
            (dialog[0], dialog[1][:-1], torch.empty(0, dtype=torch.long))
            for dialog in dialogs
        ]
        prev_dialog = [None for dialog in dialogs]
        for step in range(max_len):
            if done.all():
                break
            for i, dialog in enumerate(dialogs):
                if done[i]:
                    continue
                prev_dialog[i] = deepcopy(dialog)
                new_utterance = actions[i][: (step + 1)]
                dialogs[i] = (*dialog[:-1], new_utterance)
                if step == (len(actions[i]) - 1):
                    done[i] = True
                self.replay_buffer_expert.store(
                    prev_dialog[i],
                    actions[i][step],
                    self.imitate_reward,
                    deepcopy(dialogs[i]),
                    False,
                )

    def batch_sample(self, dialogs_to_generate):
        self.generator.enable_cache()
        run = True  # to start first run
        sub_batch_size = self.gen_sub_batch_size

        while run:
            run = False
            exc = False
            gen_dialogs_batch = []
            try:
                for i in range(
                    (self.batch_size // sub_batch_size)
                    + int((self.batch_size % sub_batch_size) > 0)
                ):
                    upper = (i + 1) * sub_batch_size
                    lower = i * sub_batch_size
                    generated_dialogs = self.generate_dialogs(
                        dialogs_to_generate[lower:upper], max_len=self.maxlen,
                    )
                    generated_dialogs = self.decode_reply(generated_dialogs)
                    gen_dialogs_batch.extend(generated_dialogs)
            except RuntimeError as e:
                print("reducing sample batch_size")
                exc = True

            if exc:
                self.generator.clear_cache()
                run = True
                torch.cuda.synchronize()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                torch.cuda.synchronize()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                sub_batch_size = sub_batch_size // 2
        return gen_dialogs_batch

    def batch_update(self):
        if (
            self.replay_buffer_expert.size < self.min_replay_size
            or self.replay_buffer_sample.size < self.min_replay_size
        ):
            print("Not enough data, skipping update")
            return None
        self.no_update_step = +1
        if self.no_update_step > 1000:
            self.no_update_step = 0
            self.generator_target.load_state_dict(self.generator.state_dict())
        total_loss = 0
        total_q = 0
        self.generator.disable_cache()
        self.generator_target.disable_cache()
        samples_expert = self.replay_buffer_expert.sample_batch()
        samples_policy = self.replay_buffer_sample.sample_batch()
        run = True  # to start first run
        sub_batch_size = self.update_batch_size // 2
        while run:
            run = False
            exc = False
            try:
                iters = len(samples_expert["obs"]) // sub_batch_size + int(
                    (len(samples_expert["obs"]) % sub_batch_size) != 0
                )
                for i in range(iters):
                    upper = (i + 1) * sub_batch_size
                    lower = i * sub_batch_size

                    loss, q = self._compute_loss(
                        {
                            sample_key: [
                                *sample[lower:upper],
                                *samples_policy[sample_key][lower:upper],
                            ]
                            for sample_key, sample in samples_expert.items()
                        },
                        iters,
                    )
                    if MIXED_PREC:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss += loss.detach().cpu().item()
                    total_q += q.detach().cpu().item()
                    del loss, q
            except RuntimeError as e:
                print("reducing update batch_size")
                exc = True
                if "loss" in locals():
                    del loss

            if exc:
                run = True
                if "loss" in locals():
                    del loss
                torch.cuda.synchronize()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                torch.cuda.synchronize()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                sub_batch_size = sub_batch_size // 2

            if MIXED_PREC:
                self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.generator.parameters(), self.gradient_clip_norm)

            if MIXED_PREC:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        self.writer.add_scalar("Loss", total_loss, global_step=self.train_step)
        self.writer.add_scalar("Q value", total_q, global_step=self.train_step)
        self.writer.add_scalar(
            "Entropy", self.generator.get_entropy(), global_step=self.train_step
        )
        self.generator.enable_cache()

    def _compute_loss(self, samples, norm_term):
        with autocast() if MIXED_PREC else suppress():
            state = samples["obs"]
            next_state = samples["next_obs"]
            action = (
                torch.stack(samples["acts"])
                .to(self.device, non_blocking=True)
                .view([-1, 1])
            )
            rewards = (
                torch.FloatTensor(samples["rews"])
                .to(self.device, non_blocking=True)
                .view([-1, 1])
            )
            with torch.no_grad():
                next_q = self.generator_target(next_state)
                next_v = self.generator_target.getV(next_q)
                y = rewards + self.gamma * next_v
            q = self.generator(state)
            q = q.gather(1, action.long())
            loss = F.mse_loss(q, y)
            q = q.mean() / norm_term
            loss /= norm_term
        return loss, q

    def generate_dialogs(
        self, dialogs: Iterable[Tuple[str, List[str]]], max_len: int = 32
    ):
        self.generator.clear_cache()
        done = np.zeros([len(dialogs)], dtype=bool)
        dialogs = [(*dialog, torch.empty(0, dtype=torch.long)) for dialog in dialogs]
        prev_dialog = [None for dialog in dialogs]
        for step in range(max_len):
            if done.all():
                break
            actions = self.generator.choose_action(dialogs)
            ids = actions.to("cpu", non_blocking=True)
            for i, dialog in enumerate(dialogs):
                if done[i]:
                    continue
                prev_dialog[i] = deepcopy(dialog)

                if dialog[2].nelement() != 0:
                    new_utterance = torch.cat([dialog[2], ids[i].unsqueeze(-1)], dim=0)
                    dialogs[i] = (*dialog[:-1], new_utterance)
                else:
                    dialogs[i] = (*dialog[:-1], ids[i].unsqueeze(-1))

                if ids[i] == self.generator.tokenizer.eos_token_id or (
                    step == (max_len - 1)
                ):
                    done[i] = True
                self.replay_buffer_sample.store(
                    prev_dialog[i], ids[i], 0, deepcopy(dialogs[i]), False,
                )
            self.batch_update()
        self.generator.clear_cache()
        return [d[2] for d in dialogs]

    def decode_reply(self, generated_dialogs):
        return [
            self.generator.tokenizer.decode(generated, skip_special_tokens=True)
            for generated in generated_dialogs
        ]

    def checkpoint(self, dialogs):
        gen_p = os.path.join(
            self.checkpoint_path, self.MODEL_SUBPATHS["generator"]
        ) + "_{}".format(0)
        gen_target_p = os.path.join(
            self.checkpoint_path, self.MODEL_SUBPATHS["generator_target"]
        ) + "_{}".format(0)
        if not os.path.isdir(gen_p):
            os.mkdir(gen_p)
            os.mkdir(gen_target_p)
        self.generator.save(gen_p)
        self.generator_target.save(gen_target_p)

        with open(
            os.path.join(
                self.dialog_dump_path, "dialogs{}.json".format(self.train_step)
            ),
            "w",
        ) as dialog_file:
            json.dump(dialogs, dialog_file)

    def __del__(self):
        self.writer.close()
        super().__del__()

    def save(self, path: str = None):
        super().save(path=path)
        path, _ = os.path.split(path)
        gen_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        gen_target_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if not os.path.isdir(gen_dir):
            os.mkdir(gen_dir)
            os.mkdir(gen_target_dir)
        self.generator.save(gen_dir)
        self.generator_target.save(gen_target_dir)
