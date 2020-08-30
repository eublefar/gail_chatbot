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
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from torch.distributions import Categorical
from collections import deque
import json
from .gpt_lm import GPTSimple, MIXED_PREC
from tensorboardX import SummaryWriter
import yaml
import json

torch.set_num_threads(8)


class GPTFineTune(Agent):
    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "GPTFineTune"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SUBPATHS = {
            "generator": "generator",
        }
        self.eval_step = 0
        self.train_step = 0
        self.gen_episode_num = 0
        self.opt = opt
        self.persona = None
        self.history = []
        self.last_label = None
        self.last_input = None
        self.generator = None
        self.batch_size = opt["batchsize"]
        self.sub_batch_size = 8
        self.metrics = {}
        self.is_eval = False

        # Hyperparameters
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 100
        self.episode_checkpoint_num = 100
        self.add_distractors = False
        self.dialog_dump_path = opt["model_file"] + "_dialogs"
        self.checkpoint_path = os.path.split(opt["model_file"])[0]
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

    def _create_from_path(self, path: str):
        path, filename = os.path.split(path)
        param_file = os.path.join(path, "parameters.yml")
        if os.path.isfile(param_file):
            with open(os.path.join(path, "parameters.yml")) as param_file:
                overwrite_params = yaml.load(param_file.read())
        else:
            with open(os.path.join(path, "parameters.yml"), "w") as param_file:
                def_parameters = {}
                overwrite_params = self._get_default_params()
                param_file.write(yaml.dump(overwrite_params))
        self.__dict__.update(overwrite_params.get("gail", {}))
        self._construct_model(path)
        self.writer = SummaryWriter(os.path.join(path, filename) + ".tensorboard")

    def _get_default_params(self):
        return {
            "gail": {
                "sub_batch_size": 8,
                "episode_num_log": 1,
                "episode_num_dialog_dump": 100,
                "episode_checkpoint_num": 200,
            }
        }

    def _construct_model(self, path):
        self.generator = GPTSimple()

        adv_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if os.path.isfile(adv_dir):
            self.generator.load(adv_dir)
        else:
            self.generator.save(adv_dir)
        if torch.cuda.device_count() > 1:
            self.generator.cuda(1)
        else:
            self.generator.to(self.device)

    def share(self) -> Dict[str, Any]:
        return dict(**super().share(), **{"generator": self.generator,})

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

        logits, labels, loss = self.generator.fit_batch(dialogs_pos)
        self.metrics["loss"] = loss
        labels = labels.view([-1, 1])
        logits = logits.view([-1, logits.shape[-1]])[(labels != -100)[:, 0], :]
        labels = labels[labels != -100]

        onehot = torch.zeros_like(logits)
        onehot.scatter_(dim=1, index=labels.unsqueeze(-1), value=1)

        logits = logits[:, (onehot == 1).any(dim=0)]
        onehot = onehot[:, (onehot == 1).any(dim=0)]

        self.metrics["ap"] = average_precision_score(
            onehot.numpy(), logits.numpy(), average="macro",
        )
        self.metrics["mAUC"] = average_precision_score(
            onehot.numpy(), logits.numpy(), average="macro",
        )
        self.metrics["accuracy"] = accuracy_score(
            onehot.numpy(), (logits > 0.5).int().numpy()
        )
        self.metrics["f1_score"] = f1_score(
            onehot.numpy(), (logits > 0.5).int().numpy(), average="macro"
        )

        if self.train_step % self.episode_num_log == 0 and self.train_step:
            self.write_metrics()

        if (
            self.train_step % self.episode_num_dialog_dump == 0
        ) and self.train_step != 0:
            self.checkpoint()

        return [{"id": self.id} for observation in observations]

    def decode_reply(self, generated_dialogs):
        generated_dialogs_converted = []
        for persona, history, generated in generated_dialogs:
            reply_str = self.generator_policy.tokenizer.decode(generated).replace(
                self.generator_policy.tokenizer.eos_token, ""
            )
            if reply_str == "":
                reply_str = "__SILENCE__"
            generated_dialogs_converted.append((persona, history + [reply_str]))
        return generated_dialogs_converted

    def write_metrics(self):
        metrics = {}
        metrics.update(self.metrics)
        self.metrics = {}
        for key, v in metrics.items():
            self.writer.add_scalar(
                key,
                v,
                global_step=self.train_step if not self.is_eval else self.eval_step,
            )

    def checkpoint(self):
        gen_p = os.path.join(
            self.checkpoint_path, self.MODEL_SUBPATHS["generator"]
        ) + "_{}".format(self.train_step)
        if not os.path.isdir(gen_p):
            os.mkdir(gen_p)
        self.generator_policy.save(gen_p)

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
