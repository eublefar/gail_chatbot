"""Main module."""
from typing import Iterable, Dict, Tuple, List, Any

import torch
import os
import yaml

from random import sample
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from parlai.core.agents import Agent
from parlai.core.message import Message

from gail_chatbot.light.bart_lm import BARTSimple
from gail_chatbot.light.light_chatbot_base import LightChatbotBase


class LightBartFineTune(LightChatbotBase):
    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        super().__init__(opt, shared)

        self.id = "LightBartFineTune"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SUBPATHS = {
            "generator": "generator",
        }
        self.eval_step = 0
        self.train_step = 0
        self.gen_episode_num = 0
        self.opt = opt
        self.last_label = None
        self.last_input = None
        self.generator = None
        self.batch_size = opt["batchsize"]
        self.sub_batch_size = 8
        self.metrics = {}
        self.is_eval = False

        self.checkpoint_overwrite = False

        # Hyperparameters
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 100
        self.episode_checkpoint_num = 100
        self.add_distractors = False
        self.dialog_dump_path = opt["model_file"] + "_dialogs"
        self.checkpoint_path = os.path.split(opt["model_file"])[0]
        if not os.path.isdir(self.dialog_dump_path):
            os.mkdir(self.dialog_dump_path)

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
        self.generator = BARTSimple(special_tokens=self.ctx_tokens).train()

        adv_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if os.path.isdir(adv_dir):
            self.generator.load(adv_dir)
        else:
            self.generator.save(adv_dir)
        if torch.cuda.device_count() > 1:
            self.generator.cuda(1)
        else:
            self.generator.to(self.device)

    def share(self) -> Dict[str, Any]:
        return dict(**super().share(), **{"generator": self.generator,})

    def act(self):
        raise NotImplementedError()

    def batch_act(self, observations: List[Message]):
        (dialogs_neg, dialogs_pos, dialogs_to_generate), emotes = super().batch_act(
            observations
        )
        emotes = torch.LongTensor(emotes).to(self.device)

        run = True
        bs = self.sub_batch_size
        while run:
            try:
                run = False
                logits, labels, loss = self.generator.fit_batch(
                    dialogs_pos, emotes, sub_batch=bs
                )
            except Exception as e:
                if "CUDA" in str(e):
                    print("CUDA error, reducing batch_size")
                    bs //= 2
                    run = True
                    self.generator.optimizer.zero_grad()
                else:
                    raise e

        if self.train_step % self.episode_num_log == 0 and self.train_step:
            self.metrics["loss"] = loss  # pyright: reportUnboundVariable=false
            labels = labels.view([-1, 1])
            logits = logits.view([-1, logits.shape[-1]])[(labels != -100)[:, 0], :]
            labels = labels[labels != -100]
            entropy = Categorical(logits=logits).entropy().mean().item()
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
            self.metrics["entropy"] = entropy
            self.metrics["accuracy"] = accuracy_score(
                onehot.numpy(),
                (torch.eq(logits, logits.max(dim=1).values.unsqueeze(1))).int().numpy(),
            )
            self.metrics["f1_score"] = f1_score(
                onehot.numpy(),
                (torch.eq(logits, logits.max(dim=1).values.unsqueeze(1))).int().numpy(),
                average="macro",
            )
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
        gen_p = os.path.join(self.checkpoint_path, self.MODEL_SUBPATHS["generator"]) + (
            "_{}".format(self.train_step) if self.checkpoint_overwrite else "_ck"
        )
        if not os.path.isdir(gen_p):
            os.mkdir(gen_p)
        self.generator.save(gen_p)

    def __del__(self):
        self.checkpoint()
        self.writer.close()
        super().__del__()

    def save(self, path: str = None):
        super().save(path=path)
        path, _ = os.path.split(path)
        gen_dir = os.path.join(path, self.MODEL_SUBPATHS["generator"])
        if not os.path.isdir(gen_dir):
            os.mkdir(gen_dir)
        self.generator.save(gen_dir)
