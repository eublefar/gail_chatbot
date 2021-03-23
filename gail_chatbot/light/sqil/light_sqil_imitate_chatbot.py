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
from parlai.core.message import Message

from gail_chatbot.light.sqil.bart_policy import BartPolicy
from gail_chatbot.light.sqil.light_imitate_mixin import LightImitateMixin
from gail_chatbot.light.sqil.light_selfplay_base_mixin import LightSelfplayBaseMixin

try:
    from torch.cuda.amp import autocast, GradScaler

    MIXED_PREC = True
except ImportError as e:
    MIXED_PREC = False

torch.set_num_threads(8)
# TODO:
# - Test data preparation
# - implement SQIL training


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

        # Batch sizes
        self.batch_size = opt["batchsize"]  # batch size of gradient update
        self.gen_sub_batch_size = 8
        self.update_batch_size = 8

        # Hyperparameters
        self.maxlen = 50
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 100
        self.episode_checkpoint_num = 100
        self.update_generator = True
        # Counters
        self.gen_episode_num = 0

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

    def _get_default_params(self):
        return {
            "sqil": {
                "maxlen": 50,
                "maxlen": 50,
                "gen_sub_batch_size": 8,
                "update_batch_size": 8,
                "episode_num_log": 1,
                "episode_num_dialog_dump": 100,
                "episode_checkpoint_num": 200,
                "update_generator": True,
            },
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

    def share(self) -> Dict[str, Any]:
        return dict(
            **super().share(),
            **{"generator": self.generator, "generator_target": self.generator_target,}
        )

    def batch_imitate(self, dialogs):
        # print("batch_imitate", dialogs)
        pass

    def batch_sample(self, dialogs):
        # print("batch_sample", dialogs)
        # self.pp.pprint(self.histories)
        return ["test" for dialog in dialogs]

    def batch_update(self):
        pass

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
        #         self.checkpoint([])
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
        self.generator_target.save(gen_target_p)
