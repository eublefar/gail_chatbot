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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SUBPATHS = {
            "generator": "generator",
            "adversarial": "adversarial.bin",
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
        self.adversarial = None
        self.batch_size = opt["batchsize"]
        self.adv_sub_batch_size = 8
        self.gen_sub_batch_size = 8
        self.gpt_update_batch_size = 8
        self.metrics = {}
        self.is_eval = False

        # Hyperparameters
        self.similarity_coef = 0.2
        self.episode_num_log = 1
        self.episode_num_dialog_dump = 200
        self.episode_checkpoint_num = 200
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

    #             if APEX_AVAILABLE:
    #                 self.init_apex()

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
        self._construct_adversarial(path)
        self.writer = SummaryWriter(os.path.join(path, filename) + ".tensorboard")

    def _construct_generator(self, overwrite_params: Dict[str, Any], path: str):
        params = PPO.get_default_parameters()
        params.update(overwrite_params)
        params.update({"n_envs": self.gen_sub_batch_size})
        # To stabilize training accumulate gradients over bigger batch size than adversarial
        params.update({"batch_size": self.batch_size * 2})

        self.generator_policy = GptPolicy()
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

    def init_apex(self):
        print(self.device)
        (self.generator.policy, self.generator.optimizer,) = amp.initialize(
            self.generator.policy, self.generator.optimizer, opt_level="O0",
        )
        self.generator_policy = self.generator.policy

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
        for i in range(
            (self.batch_size // self.gen_sub_batch_size)
            + int(self.batch_size % self.gen_sub_batch_size > 0)
        ):
            upper = (i + 1) * self.gen_sub_batch_size
            lower = i * self.gen_sub_batch_size
            (
                generated_dialogs,
                final_transitions,
                self.gen_episode_num,
            ) = self.generate_dialogs(
                dialogs_to_generate[lower:upper], self.gen_episode_num
            )
            generated_dialogs = self.decode_reply(generated_dialogs)
            gen_dialogs_batch.extend(generated_dialogs)
            # logits, loss
            scores = self.compute_rewards(
                generated_dialogs, dialogs_pos[lower:upper], dialogs_neg[lower:upper]
            )

            #             self.metrics["adv_loss"] = loss
            self.metrics["gen_reward"] = scores.mean()

            for i, final_transition in enumerate(final_transitions):
                if final_transition is not None:
                    final_transition[2] = scores[i]
            self.generator.batch_memorize(final_transitions)

        #         self.adversarial.scaler.step(self.adversarial.optimizer)
        #         self.adversarial.scaler.update()
        #         self.adversarial.optimizer.zero_grad()

        self.generator.update(self.gen_episode_num)

        if self.train_step % self.episode_num_log == 0 and self.train_step:
            self.write_metrics()

        if self.train_step % self.episode_num_dialog_dump == 0 and self.train_step != 0:
            self.checkpoint(gen_dialogs_batch)

        return [{"id": self.id} for observation in observations]

    def decode_reply(self, generated_dialogs):
        generated_dialogs_converted = []
        for persona, history, generated in generated_dialogs:
            reply_str = self.generator_policy.tokenizer.convert_tokens_to_string(
                generated
            ).replace(self.generator_policy.tokenizer.eos_token, "")
            if reply_str == "":
                reply_str = "__SILENCE__"
            generated_dialogs_converted.append((persona, history + [reply_str]))
        return generated_dialogs_converted

    def compute_rewards(self, dialogs_gen, dialogs_pos, dialogs_neg):
        X = [
            #             *dialogs_neg,
            *dialogs_pos,
            *dialogs_gen,
        ]
        y = torch.cat(
            (
                #                 torch.zeros(size=(len(dialogs_neg),), dtype=torch.long),
                torch.ones(size=(len(dialogs_pos),), dtype=torch.long),
                torch.zeros(size=(len(dialogs_gen),), dtype=torch.long),
            )
        ).to(self.adversarial.get_device())
        loss, logits, hidden_states = self.adversarial(
            X, y, sub_batch=self.adv_sub_batch_size
        )

        probs = torch.softmax(logits, dim=-1)

        self.metrics["neg_logits"] = probs[: len(dialogs_neg), 1].mean()
        self.metrics["pos_logits"] = probs[: len(dialogs_pos), 1].mean()
        self.metrics["gen_logits"] = probs[len(dialogs_pos) :, 1].mean()

        adequacy_scores = probs[-len(dialogs_gen) :, 1]

        positive_embs = hidden_states[: len(dialogs_pos), 0, :]
        generated_embs = hidden_states[len(dialogs_pos) :, 0, :]
        next_sentence_similarity_scores = torch.nn.functional.cosine_similarity(
            positive_embs, generated_embs, dim=-1
        )
        self.metrics[
            "next_sentence_similarity_scores"
        ] = next_sentence_similarity_scores.mean()
        reward_scores = (
            (adequacy_scores + self.similarity_coef * next_sentence_similarity_scores)
            .cpu()
            .detach()
            .numpy()
        )
        #         reward_scores = []
        #         for dialog in dialogs_gen:
        #             _, reply = dialog
        #             reply = reply[-1]
        #             reward_scores.append( 1 if ("Hello" in reply or "hi" in reply or "how" in reply or "what" in reply) else -1)
        return np.asarray(reward_scores)

    #     (
    #             ,
    # #             logits,
    # #             loss
    #         )

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

    def generate_dialogs(
        self,
        dialogs: Iterable[Tuple[str, List[str]]],
        episode_num: int = 0,
        max_len: int = 16,
    ):
        global_step = 0
        done = np.zeros([len(dialogs)], dtype=bool)
        dialogs = [(*dialog, []) for dialog in dialogs]
        prev_dialog = [(*dialog, []) for dialog in dialogs]
        final_transitions = [None] * len(dialogs)
        transitions = [None] * len(dialogs)
        for step in range(max_len):
            if done.all():
                break
            actions = self.generator.batch_act(dialogs, done)
            words = self.generator_policy.tokenizer.convert_ids_to_tokens(actions)
            for i, dialog in enumerate(dialogs):
                if done[i]:
                    continue
                prev_dialog[i] = deepcopy(dialog)
                dialog[2].append(words[i])
                if words[i] == self.generator_policy.tokenizer.eos_token or (
                    step == (max_len - 1)
                ):
                    episode_num += 1
                    done[i] = True
                    final_transitions[i] = [
                        prev_dialog[i],
                        actions[i],
                        0,
                        done[i],
                        dialog,
                    ]
                else:
                    transitions[i] = [prev_dialog[i], actions[i], 0, done[i], dialog]
                    global_step += 1
            if not all(final_transitions):
                self.generator.batch_memorize(transitions)

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
