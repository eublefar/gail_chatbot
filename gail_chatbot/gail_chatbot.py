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
from .bert_adversarial import BertAdversarial
from gym_loop.agents.pytorch_ppo import PPO
from tensorboardX import SummaryWriter
import yaml
import json


try:
    from apex import amp

    APEX_AVAILABLE = True
    print("********** Using mixed precision **************")
except ModuleNotFoundError:
    APEX_AVAILABLE = False
    print("********** Not using mixed precision **************")

torch.set_num_threads(8)


class GailChatbot(Agent):
    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "GailChatbot"
        self.MODEL_SUBPATHS = {"generator": "generator", "adversarial": "adversarial"}
        self.eval_step = 0
        self.train_step = 0
        self.gen_episode_num = 0
        self.opt = opt
        self.persona = None
        self.history = []
        self.last_label = None
        self.last_input = None
        self.generator = None
        self.adversarial = None
        self.batch_size = opt["batchsize"]

        # Hyperparameters
        self.similarity_coef = 0.3
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 1000
        self.dialog_dump_path = opt["model_file"] + "_dialogs"
        os.mkdir(self.dialog_dump_path)

        if opt["task"] != "convai2":
            raise ValueError("Only works on convai task")

        if shared:
            self._create_from_shared(shared)
        else:
            self._create_from_path(opt["model_file"])
            if APEX_AVAILABLE:
                self.init_apex()

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
                param_file.write(yaml.dump(PPO.get_default_parameters()))
            overwrite_params = {}
        self._construct_generator(overwrite_params, path)
        self.adversarial = BertAdversarial()
        self.writer = SummaryWriter(os.path.join(path, filename) + ".tensorboard")

    def _construct_generator(self, overwrite_params: Dict[str, Any], path: str):
        params = PPO.get_default_parameters()
        params.update(overwrite_params)
        params.update({"n_envs": self.batch_size})
        self.generator_policy = GptPolicy()
        params["policy"] = self.generator_policy
        self.generator = PPO(**params)

        gen_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if os.path.isdir(gen_dir):
            self.generator_policy.load(gen_dir)
        else:
            os.mkdir(gen_dir)
            self.generator_policy.save(gen_dir)

    def init_apex(self):
        self.generator.policy, self.generator.optimizer = amp.initialize(
            self.generator.policy, self.generator.optimizer
        )

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
        is_eval = any(["eval_labels" in observation for observation in observations])

        if is_eval:
            self.eval_step += 1
        else:
            self.train_step += 1

        #  Optimize generative model
        dialogs_neg, dialogs_pos, dialogs_to_generate = self.flatten(observations)

        (
            generated_dialogs,
            final_transitions,
            self.gen_episode_num,
        ) = self.generate_dialogs(dialogs_to_generate, self.gen_episode_num)

        try:
            generated_dialogs_converted = []
            for persona, history, generated in generated_dialogs:
                reply_str = self.generator_policy.tokenizer.convert_tokens_to_string(
                    generated
                )
                generated_dialogs_converted.append((persona, history + [reply_str]))

            generated_dialogs = generated_dialogs_converted
        except KeyError as ke:
            print(generated)
            raise ke

        X = [*dialogs_neg, *dialogs_pos, *generated_dialogs]
        y = torch.cat(
            (
                torch.zeros(size=(len(dialogs_neg),), dtype=torch.long),
                torch.ones(size=(len(dialogs_pos),), dtype=torch.long),
                torch.zeros(size=(len(generated_dialogs),), dtype=torch.long),
            )
        )

        logits, hidden_states, adv_loss = self.adversarial.fit_batch(X, y)

        positive_ids = torch.arange(0, len(dialogs_pos)) + len(dialogs_neg)
        generated_ids = positive_ids + len(generated_dialogs)
        positive_embs = hidden_states[positive_ids, 0, :]
        generated_embs = hidden_states[generated_ids, 0, :]

        adequacy_scores = torch.softmax(logits, dim=-1)[:, 1]
        adequacy_scores = adequacy_scores[generated_ids]
        next_sentence_similarity_scores = torch.nn.functional.cosine_similarity(
            positive_embs, generated_embs, dim=-1
        )

        reward_scores = (
            (adequacy_scores + self.similarity_coef * next_sentence_similarity_scores)
            .detach()
            .numpy()
        )

        for i, final_transition in enumerate(final_transitions):
            if final_transition is not None:
                final_transition[2] = reward_scores[i]
                self.generator.memorize(*final_transition, env_id=i)

        self.generator.update(self.gen_episode_num)

        if self.train_step % self.episode_num_log == 0 and self.train_step:
            metrics = self.generator.metrics(
                self.train_step if not is_eval else self.eval_step
            )
            metrics.update(
                {
                    "mean_positive_logits": torch.softmax(logits, dim=-1)[
                        positive_ids
                    ].mean(),
                    "mean_generated_logits": torch.softmax(logits, dim=-1)[
                        generated_ids
                    ].mean(),
                    "next_sentence_similarity_scores": next_sentence_similarity_scores.mean(),
                    "adv_loss": adv_loss,
                }
            )

            for key, v in metrics.items():
                self.writer.add_scalar(
                    key,
                    v,
                    global_step=self.train_step if not is_eval else self.eval_step,
                )

        if self.train_step % self.episode_num_dialog_dump == 0 and self.train_step != 0:
            with open(
                os.path.join(
                    self.dialog_dump_path, "dialogs{}.json".format(self.train_step)
                )
            ) as dialog_file:
                json.dump(generated_dialogs, dialog_file)

        return [{"id": self.id} for observation in observations]

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

    def generate_dialogs(
        self,
        dialogs: Iterable[Tuple[str, List[str]]],
        episode_num: int = 0,
        max_len: int = 512,
    ):
        global_step = 0

        done = np.zeros([len(dialogs)], dtype=bool)
        dialogs = [(*dialog, []) for dialog in dialogs]
        prev_dialog = [(*dialog, []) for dialog in dialogs]
        final_transitions = [None] * len(dialogs)
        for step in range(max_len):
            if done.all():
                break
            for i, dialog in enumerate(dialogs):
                if done[i]:
                    continue
                prev_dialog[i] = deepcopy(dialog)
                action = self.generator.act(dialog, episode_num=episode_num, env_id=i)
                word = self.generator_policy.tokenizer.convert_ids_to_tokens(action)[0]
                dialog[2].append(word)
                if word == self.generator_policy.tokenizer.eos_token:
                    episode_num += 1
                    done[i] = True
                    final_transitions[i] = [
                        prev_dialog[i],
                        action,
                        0,
                        done[i],
                        dialog,
                        global_step,
                    ]
                else:
                    self.generator.memorize(
                        prev_dialog[i],
                        action,
                        0,
                        done[i],
                        dialog,
                        global_step,
                        env_id=i,
                    )
                global_step += 1
        self.generator_policy.clear_cache()
        return dialogs, final_transitions, episode_num

    def __del__(self):
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
