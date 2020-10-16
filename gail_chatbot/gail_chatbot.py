"""Main module."""
import numpy as np
from typing import Iterable, Dict, Tuple, List, Any
from parlai.core.agents import Agent
from parlai.core.message import Message
from random import sample
import torch
from tensorboardX import SummaryWriter
import os
from copy import deepcopy
from torch.distributions import Categorical
from collections import deque
import json
from .gpt_policy import GptPolicy
from .bert_adversarial import BertAdversarial, MIXED_PREC
from gym_loop.agents.pytorch_ppo import PPO
from tensorboardX import SummaryWriter
import yaml
import json
import pprint

torch.set_num_threads(8)

# MAKE SURE THAT N_STEPS IS SET CORRECTLY SO THAT ALL BATCH_SIZE//SUBBATCHSIZE * 2 ELEMENTS FIT IN MEMORY


class GailChatbot(Agent):
    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "GailChatbot"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SUBPATHS = {
            "generator": "generator",
            "adversarial": "adversarial.bin",
        }
        self.maxlen = 50
        self.eval_step = 0
        self.train_step = 0
        self.gen_episode_num = 0
        self.opt = opt
        self.persona = None
        self.history = []
        self.last_label = None
        self.last_input = None
        self.lr_updated = False
        self.generator = None
        self.adversarial = None
        self.batch_size = opt["batchsize"]
        self.adv_sub_batch_size = 8
        self.gen_sub_batch_size = 8
        self.gpt_update_batch_size = 8
        self.metrics = {}
        self.is_eval = False
        self.warmup_steps = 2
        self.adversarial_lr = 1e-5
        self.rew_mean = None
        self.rew_std = None
        self.momentum = 0.1

        # Hyperparameters
        self.similarity_coef = 0.2
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 100
        self.episode_checkpoint_num = 100
        self.update_generator = True
        self.update_adversarial = True
        self.add_distractors = False
        self.dialog_dump_path = opt["model_file"] + "_dialogs"
        self.checkpoint_path = os.path.split(opt["model_file"])[0]
        self.pp = pprint.PrettyPrinter(indent=4)
        if not os.path.isdir(self.dialog_dump_path):
            os.mkdir(self.dialog_dump_path)

        if opt["task"] != "convai2":
            raise ValueError("Only works on convai task")

        if shared:
            self._create_from_shared(shared)
        else:
            self._create_from_path(opt["model_file"])

    def _create_from_shared(self, shared: Dict[str, Any]):
        self.generator = shared["generator"]
        self.adversarial = shared["adversarial"]
        self.generator_policy = self.generator.policy

    def _create_from_path(self, path: str):
        path, filename = os.path.split(path)
        param_file = os.path.join(path, "parameters.yml")
        if os.path.isfile(param_file):
            with open(os.path.join(path, "parameters.yml")) as param_file:
                overwrite_params = yaml.load(param_file.read())
        else:
            with open(os.path.join(path, "parameters.yml"), "w") as param_file:
                def_parameters = PPO.get_default_parameters()
                overwrite_params = self._get_default_params()
                param_file.write(yaml.dump(overwrite_params))
        self.__dict__.update(overwrite_params.get("gail", {}))
        self._construct_generator(overwrite_params.get("generator_agent", {}), path)
        self._construct_adversarial(path)
        self.writer = SummaryWriter(os.path.join(path, filename) + ".tensorboard")
        self.reward_norm = torch.nn.BatchNorm1d(1)

    def _get_default_params(self):
        return {
            "gail": {
                "maxlen": 50,
                "gd_frac": 0,
                "gd_frac_decay": 1 / 4000,
                "maxlen": 50,
                "adversarial_lr_warmup": 1e-5,
                "adversarial_lr": 1e-5,
                "warmup_steps": 100,
                "adv_sub_batch_size": 8,
                "gen_sub_batch_size": 8,
                "gpt_update_batch_size": 8,
                "similarity_coef": 0.2,
                "episode_num_log": 1,
                "episode_num_dialog_dump": 100,
                "episode_checkpoint_num": 200,
                "update_generator": True,
                "update_adversarial": True,
                "warmup_steps": 1,
            },
            "generator_agent": PPO.get_default_parameters(),
        }

    def _construct_generator(self, overwrite_params: Dict[str, Any], path: str):
        params = PPO.get_default_parameters()
        params.update(overwrite_params)
        params.update({"n_envs": self.gen_sub_batch_size})
        # Same batch size as adversarial (negative + positive sample x batch_size)
        params.update({"batch_size": self.batch_size})

        self.generator_policy = GptPolicy().eval()
        params["policy"] = self.generator_policy
        self.generator = PPO(**params)

        gen_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if os.path.isdir(gen_dir):
            self.generator_policy.load(gen_dir)
        else:
            os.mkdir(gen_dir)
            self.generator_policy.save(gen_dir)
        if torch.cuda.device_count() > 1:
            self.adversarial.cuda(0)
        else:
            self.generator_policy.to(self.device)

    def _construct_adversarial(self, path):
        self.adversarial = BertAdversarial()

        adv_dir = os.path.join(path, self.MODEL_SUBPATHS["adversarial"])
        if os.path.isfile(adv_dir):
            self.adversarial.load_state_dict(torch.load(adv_dir))
        else:
            torch.save(self.adversarial.state_dict(), adv_dir)
        if torch.cuda.device_count() > 1:
            self.adversarial.cuda(1)
        else:
            self.adversarial.to(self.device)
        self.adversarial.set_lr(self.adversarial_lr)

    def share(self) -> Dict[str, Any]:
        return dict(
            **super().share(),
            **{"generator": self.generator, "adversarial": self.adversarial}
        )

    def observe(self, observation: Message):
        if "text" not in observation:
            self.reset()
            return observation
        res = dict(observation)
        if not self.persona:
            res["text"] = self._extract_persona(observation["text"])
        if self.last_label is not None:
            self.history.append(self.last_label)
        self.history.append(res["text"])

        self.last_label = (
            observation["labels"][0]
            if "labels" in observation
            else observation["eval_labels"][0]
        )
        self.episode_done = observation["episode_done"]
        neg_obs = list(observation["label_candidates"])
        neg_obs.remove(self.last_label)
        neg_sample = sample(neg_obs, 2)
        res["text"] = [
            (self.persona, self.history),  # Generate sample
            (self.persona, self.history + [self.last_label]),  # Positive sample
            (self.persona, self.history + [neg_sample[0]]),  # Negative sample
        ]
        res[("labels" if "labels" in observation else "eval_labels")] = [
            0,
            1,
            0,
        ]
        res["generate_mask"] = [
            1,
            0,
            0,
        ]
        self.last_input = observation
        if self.episode_done:
            self.reset()
        return res

    def _extract_persona(self, text):
        lines = text.split("\n")
        persona = [
            line.replace("your persona: ", "")
            for line in lines
            if "your persona: " in line
        ]
        if not persona:
            raise ValueError("Tried to parse persona but none found")
        self.persona = "\n".join(persona)
        return "\n".join([line for line in lines if "your persona: " not in line])

    def act(self):
        raise NotImplementedError()

    def batch_act(self, observations: List[Message]):
        self.is_eval = any(
            ["eval_labels" in observation for observation in observations]
        )

        if self.is_eval:
            self.eval_step += 1
        else:
            self.train_step += 1

        #  Optimize generative model
        dialogs_neg, dialogs_pos, dialogs_to_generate = self.flatten(observations)
        gen_dialogs_batch = []
        if self.update_generator:
            gen_dialogs_batch = self.update_generator_(dialogs_pos, dialogs_to_generate)

        if self.update_adversarial:
            self.update_adversarial_(dialogs_neg, dialogs_pos, gen_dialogs_batch)

        self.gd_frac = self.gd_frac - self.gd_frac * self.gd_frac_decay
        self.gd_frac = max(self.gd_frac, 0)
        if self.train_step % self.episode_num_log == 0 and self.train_step:
            self.write_metrics()

        if (
            self.train_step % self.episode_num_dialog_dump == 0
        ) and self.train_step != 0:
            self.checkpoint(gen_dialogs_batch)

        return [{"id": self.id} for observation in observations]

    def update_generator_(self, dialogs_pos, dialogs_to_generate):
        #         torch.cuda.empty_cache()
        with torch.no_grad():
            gen_dialogs_batch = self.generate_dialog_batch(
                dialogs_to_generate, dialogs_pos
            )
            self.force_teacher_batch(dialogs_pos)

        self.generator_policy.disable_cache()
        #         torch.cuda.empty_cache()
        self.generator.update(self.gen_episode_num)
        self.generator_policy.enable_cache()

        return gen_dialogs_batch

    def generate_dialog_batch(self, dialogs_to_generate, dialogs_pos):
        gen_dialogs_batch = []
        for i in range(
            (self.batch_size // self.gen_sub_batch_size)
            + int((self.batch_size % self.gen_sub_batch_size) > 0)
        ):
            upper = (i + 1) * self.gen_sub_batch_size
            lower = i * self.gen_sub_batch_size
            (
                generated_dialogs,
                final_transitions,
                self.gen_episode_num,
            ) = self.generate_dialogs(
                dialogs_to_generate[lower:upper],
                self.gen_episode_num,
                temp=None,
                max_len=self.maxlen,
            )
            generated_dialogs = self.decode_reply(generated_dialogs)
            gen_dialogs_batch.extend(generated_dialogs)
            scores = self.compute_rewards(generated_dialogs, dialogs_pos[lower:upper])
            self.metrics["gen_reward"] = scores.mean()

            for i, final_transition in enumerate(final_transitions):
                if final_transition is not None:
                    final_transition[2] = scores[i]
            self.generator.batch_memorize(final_transitions)
        return gen_dialogs_batch

    def generate_dialogs(
        self,
        dialogs: Iterable[Tuple[str, List[str]]],
        episode_num: int = 0,
        max_len: int = 32,
        temp=None,
        min_step=3,
    ):
        self.generator_policy.clear_cache()
        global_step = 0
        done = np.zeros([len(dialogs)], dtype=bool)
        dialogs = [(*dialog, torch.empty(0, dtype=torch.long,),) for dialog in dialogs]
        prev_dialog = [None for dialog in dialogs]
        final_transitions = [None] * len(dialogs)
        for step in range(max_len):

            transitions = [None] * len(dialogs)
            if done.all():
                break
            actions = self.generator.batch_act(dialogs, done).detach()
            ids = self.generator_policy.decode(actions, temp=temp).detach()
            actions_cpu = actions.to("cpu", non_blocking=True)
            for i, dialog in enumerate(dialogs):
                if done[i]:
                    continue
                prev_dialog[i] = deepcopy(dialog)
                if dialog[2].nelement() != 0:
                    new_utterance = torch.cat([dialog[2], ids[i].unsqueeze(-1)], dim=0)
                    dialogs[i] = (*dialog[:-1], new_utterance)
                else:
                    dialogs[i] = (*dialog[:-1], ids[i].unsqueeze(-1))
                if ids[i] == self.generator_policy.tokenizer.eos_token_id or (
                    step == (max_len - 1)
                ):
                    episode_num += 1
                    done[i] = True
                    final_transitions[i] = [
                        prev_dialog[i],
                        actions_cpu[i],
                        0,
                        done[i],
                        deepcopy(dialogs[i]),
                    ]
                else:
                    transitions[i] = [
                        prev_dialog[i],
                        actions_cpu[i],
                        0,
                        done[i],
                        deepcopy(dialogs[i]),
                    ]
                    global_step += 1
            if not all(final_transitions):
                self.generator.batch_memorize(transitions)
            del actions, ids, actions_cpu, transitions

        self.generator_policy.clear_cache()
        return dialogs, final_transitions, episode_num

    def force_teacher_batch(self, dialogs_pos):
        gd_frac = int(self.batch_size * self.gd_frac)
        if gd_frac == 0:
            return
        dialogs_pos = dialogs_pos[:gd_frac]
        for i in range(
            (gd_frac // self.gen_sub_batch_size)
            + int((gd_frac % self.gen_sub_batch_size) > 0)
        ):
            upper = (i + 1) * self.gen_sub_batch_size
            lower = i * self.gen_sub_batch_size
            self.gen_episode_num = self.force_teacher(
                dialogs_pos[lower:upper], self.gen_episode_num, max_len=self.maxlen
            )

    def force_teacher(
        self,
        dialogs: Iterable[Tuple[str, List[str]]],
        episode_num: int = 0,
        max_len: int = 32,
        temp=None,
    ):
        self.generator_policy.clear_cache()
        global_step = 0
        done = np.zeros([len(dialogs)], dtype=bool)
        actions = [
            torch.LongTensor(
                self.generator_policy.tokenizer.encode(dialog[1][-1])
                + [self.generator_policy.tokenizer.eos_token_id]
            )
            for dialog in dialogs
        ]
        dialogs = [
            (dialog[0], dialog[1][:-1], torch.empty(0, dtype=torch.long),)
            for dialog in dialogs
        ]
        prev_dialog = [None for dialog in dialogs]
        final_transitions = [None] * len(dialogs)
        for step in range(max_len):
            transitions = [None] * len(dialogs)
            if done.all():
                break
            _ = self.generator.batch_act(
                dialogs, done, log_prob_override=torch.ones(len(dialogs)) * -0.045
            ).detach()
            for i, dialog in enumerate(dialogs):
                if done[i]:
                    continue
                prev_dialog[i] = deepcopy(dialog)
                new_utterance = actions[i][: (step + 1)]
                dialogs[i] = (*dialog[:-1], new_utterance)
                if (step == (len(actions[i]) - 1)) or (step == (max_len - 1)):
                    episode_num += 1
                    done[i] = True
                    final_transitions[i] = [
                        prev_dialog[i],
                        actions[i][step],
                        (2),
                        done[i],
                        deepcopy(dialogs[i]),
                    ]
                else:
                    transitions[i] = [
                        prev_dialog[i],
                        actions[i][step],
                        0,
                        done[i],
                        deepcopy(dialogs[i]),
                    ]
                    global_step += 1
            if not all(final_transitions):
                self.generator.batch_memorize(transitions)
            del transitions
        self.generator_policy.clear_cache()
        self.generator.batch_memorize(final_transitions)
        return episode_num

    def decode_reply(self, generated_dialogs):
        generated_dialogs_converted = []
        for persona, history, generated in generated_dialogs:
            reply_str = self.generator_policy.tokenizer.decode(
                generated, skip_special_tokens=True
            )
            if reply_str == "":
                reply_str = "__SILENCE__"
            generated_dialogs_converted.append((persona, history + [reply_str]))
        return generated_dialogs_converted

    def compute_rewards(self, dialogs_gen, dialogs_pos):
        loss, probs = self.adversarial.forward_contrastive(
            dialogs_gen, dialogs_pos, sub_batch=self.gen_sub_batch_size, backprop=False
        )
        probs = probs.cpu().float().detach()

        self.metrics["gen_logits_predict"] = probs[:, 0].mean()

        adequacy_scores = probs[:, 0]

        #         self.update_stats(adequacy_scores)
        #         adequacy_scores = (
        #             ((adequacy_scores - self.rew_mean) / self.rew_std)
        #             if self.rew_std != 0 else
        #             0
        #         )
        #         adequacy_scores = self.reward_norm(
        #             adequacy_scores.unsqueeze(-1)
        #         ).squeeze(-1)[:probs.shape[0]]

        reward_scores = adequacy_scores.cpu().detach().numpy()

        return reward_scores

    def update_stats(self, adequacy_scores_for_stats):
        if self.rew_std and (self.rew_std == self.rew_std):
            self.rew_std = (
                self.rew_std * (1 - self.momentum)
                + adequacy_scores_for_stats.std() * self.momentum
            )
        else:
            self.rew_std = adequacy_scores_for_stats.std()
        if self.rew_mean and (self.rew_mean == self.rew_mean):
            self.rew_mean = (
                self.rew_mean * (1 - self.momentum)
                + adequacy_scores_for_stats.mean() * self.momentum
            )
        else:
            self.rew_mean = adequacy_scores_for_stats.mean()

    def update_adversarial_(self, dialogs_neg, dialogs_pos, gen_dialogs_batch):
        #         torch.cuda.empty_cache()
        dialog_neg = []

        loss, probs = self.adversarial.forward_contrastive(
            gen_dialogs_batch, dialogs_pos, sub_batch=self.adv_sub_batch_size
        )

        p = probs[:, 0].mean()
        diff = min(max(0.5 - p, 0), 0.5)
        new_lr = -0.000024975 * diff + 0.00001
        new_lr = max(new_lr, 1e-8)
        self.adversarial.set_lr(new_lr)
        self.metrics["lr"] = new_lr
        self.metrics["pos_logits"] = probs[:, 1].mean()
        self.metrics["gen_logits"] = probs[:, 0].mean() if self.update_generator else -1
        self.metrics["adv_loss"] = loss

        if MIXED_PREC:
            self.adversarial.scaler.step(self.adversarial.optimizer)
            self.adversarial.scaler.update()
        else:
            self.adversarial.optimizer.step()
        self.adversarial.optimizer.zero_grad()

    def write_metrics(self):
        metrics = self.generator.metrics(
            self.train_step if not self.is_eval else self.eval_step
        )
        metrics.update(self.metrics)
        self.metrics = {}
        for key, v in metrics.items():
            self.writer.add_scalar(
                key,
                v,
                global_step=self.train_step if not self.is_eval else self.eval_step,
            )

    def checkpoint(self, dialogs):
        gen_p = os.path.join(
            self.checkpoint_path, self.MODEL_SUBPATHS["generator"]
        ) + "_{}".format(self.train_step)
        if not os.path.isdir(gen_p):
            os.mkdir(gen_p)
        self.generator_policy.save(gen_p)
        adv_p = os.path.join(
            self.checkpoint_path, self.MODEL_SUBPATHS["adversarial"]
        ) + "_{}".format(self.train_step)
        torch.save(self.adversarial.state_dict(), adv_p)
        with open(
            os.path.join(
                self.dialog_dump_path, "dialogs{}.json".format(self.train_step)
            ),
            "w",
        ) as dialog_file:
            json.dump(dialogs, dialog_file)

    def flatten(
        self, observations: List[Message]
    ) -> Tuple[
        List[Tuple[str, List[str]]],
        List[Tuple[str, List[str]]],
        List[Tuple[str, List[str]]],
    ]:
        """Split messages into dialogs"""
        dialogs_to_generate = []
        dialogs_neg = []
        dialogs_pos = []
        for observation in observations:
            for i, dialog in enumerate(observation["text"]):
                if observation["generate_mask"][i]:
                    dialogs_to_generate.append(dialog)
                elif observation["labels"][i]:
                    dialogs_pos.append(dialog)
                else:
                    dialogs_neg.append(dialog)

        return dialogs_neg, dialogs_pos, dialogs_to_generate

    def __del__(self):
        #         self.checkpoint([])
        self.writer.close()
        super().__del__()

    def reset(self):
        super().reset()
        self.history = []
        self.last_label = None
        self.persona = None

    def save(self, path: str = None):
        super().save(path=path)
        path, _ = os.path.split(path)
        gen_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if not os.path.isdir(gen_dir):
            os.mkdir(gen_dir)
        self.generator.save(gen_dir)
