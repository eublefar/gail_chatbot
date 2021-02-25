from typing import Dict, Tuple, List, Any

import string
import json
import os

from parlai.core.agents import Agent
from parlai.core.message import Message
from random import randint, sample, uniform, choice

from gail_chatbot.phrases import UNCERTAINTY_PHRASES, NO_KNOWLEDGE_PHRASES

letters = string.ascii_letters

import pathlib

path = pathlib.Path(__file__).parent.absolute()


class LightChatbotBase(Agent):
    """Abstract class that handles parsing Convai task into dialogue batches.
    """

    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "LightChatbotBase"
        # if opt["task"] != "light_dialog":
        #     raise ValueError("Only works on light_dialog task")

        self.eval_step = 0
        self.train_step = 0

        self.questions_dataset = None
        with open(os.path.join(path, "questions.json")) as f:
            self.questions_dataset = json.load(f)

        self.persona = None
        self.history = []
        self.last_label = None
        self.last_history = None
        self.noise_happened = False
        self.unknown_happened = False

        # self.questions_dataset = None
        # with open(os.path.join(path, "questions.json")) as f:
        #     self.questions_dataset = json.load(f)

        self.noise_frac = 0.001
        self.noise_distractor_frac = 0.6
        self.unknown_frac = 0.001

        self.utt_queue = []
        self.resp_queue = []

        self.ctx_tokens = [
            "_setting_name",
            "_setting_desc",
            "_partner_name",
            "_self_name",
            "_self_persona",
            "_other_persona",
        ]

        self.ignore_line_tokens = [
            "_self_say",
            "_self_act",
            "_partner_act",
            "_partner_emote",
            "_object_desc",
        ]
        self.filter_tokens = ["_partner_say"]

        self.emote_token = "_self_emote"
        self.emotes = """
            ponder
            nod
            sigh
            grin
            frown
            shrug
            blush
            smile
            gasp
            cry
            groan
            laugh
            scream
            dance
            growl
            stare
            wink
            nudge
            pout
            applaud
            wave
            yawn
        """.strip().split(
            "\n"
        )
        self.emotes = [em.strip() for em in self.emotes]

    def observe(self, observation: Message):

        if "text" not in observation:
            self.reset()
            return observation

        neg_obs = list(observation["label_candidates"])

        neg_obs.remove(
            observation["labels"][0]
            if "labels" in observation
            else observation["eval_labels"][0]
        )
        neg_sample = sample(neg_obs, 2)

        res = dict(observation)

        if not self.persona:
            res["text"] = self._extract_persona(observation["text"])
            if self.filter_tokens[0] not in observation["text"]:
                res["text"] += "\n" + self.filter_tokens[0] + " __SILENCE__"

        res["text"], res["emote"] = self._extract_emote(res["text"])

        res["text"] = self._filter_tokens(res["text"])

        if self.last_label is not None:
            self.history.append(self.last_label)

        if uniform(0, 1) < self.noise_frac and not self.noise_happened:
            self.noise_happened = True
            self._add_out_of_context_exchange(res, neg_sample)
        elif uniform(0, 1) < self.unknown_frac and not self.unknown_happened:
            self.unknown_happened = True
            self._add_unknown_question(res)
        else:
            self._add_utterance(res)

        res = self._build_result_dict(res, neg_sample)

        self.episode_done = observation["episode_done"]
        if self.episode_done:
            self.reset()
        return res

    def _add_utterance(self, res):
        if self.utt_queue:
            self.utt_queue.append(res["text"])
            self.resp_queue.append(
                res["labels"][0] if "labels" in res else res["eval_labels"][0]
            )

            self.history.append(self.utt_queue.pop(0))

            self.last_label = self.resp_queue.pop(0)
        else:
            self.history.append(res["text"])

            self.last_label = (
                res["labels"][0] if "labels" in res else res["eval_labels"][0]
            )

    def _add_unknown_question(self, res):
        self.history.append(choice(self.questions_dataset))

        self.last_label = choice(NO_KNOWLEDGE_PHRASES)

        self.utt_queue.append(res["text"])
        self.resp_queue.append(
            res["labels"][0] if "labels" in res else res["eval_labels"][0]
        )

    def _add_out_of_context_exchange(self, res, neg_sample):
        if uniform(0, 1) < self.noise_distractor_frac:
            self.history.append(neg_sample[1])
        else:
            randstr = " ".join(
                [
                    "".join(choice(letters) for i in range(randint(2, 10)))
                    for _ in range(randint(2, 7))
                ]
            )
            self.history.append(randstr)

        self.last_label = choice(UNCERTAINTY_PHRASES)

        self.utt_queue.append(res["text"])
        self.resp_queue.append(
            res["labels"][0] if "labels" in res else res["eval_labels"][0]
        )

    def _build_result_dict(self, res, neg_sample):
        randstr = " ".join(
            [
                "".join(choice(letters) for i in range(randint(2, 10)))
                for _ in range(randint(2, 7))
            ]
        )
        res["text"] = [
            (self.persona, self.history),  # Generate sample
            (self.persona, self.history + [self.last_label]),  # Positive sample
            (
                self.persona,
                self.history + [neg_sample[0] if uniform(0, 1) < 0.5 else randstr],
            ),  # Negative sample
        ]
        res[("labels" if "labels" in res else "eval_labels")] = [
            0,
            1,
            0,
        ]
        res["generate_mask"] = [
            1,
            0,
            0,
        ]
        return res

    def _extract_persona(self, text):
        lines = text.split("\n")
        persona = [line for line in lines if line.split(" ")[0] in self.ctx_tokens]
        if not persona:
            raise ValueError("Tried to parse persona but none found")
        self.persona = "\n".join(persona)
        return "\n".join(
            [line for line in lines if line.split(" ")[0] not in self.ctx_tokens]
        )

    def _extract_emote(self, text):
        lines = text.split("\n")
        emote_lines = [line for line in lines if line.split(" ")[0] == self.emote_token]
        if not emote_lines:
            emote = len(self.emotes)
        else:
            emote = self.emotes.index(emote_lines[0].split(" ")[-1])
        return (
            "\n".join(
                [line for line in lines if line.split(" ")[0] != self.emote_token]
            ),
            emote,
        )

    def _filter_tokens(self, text):
        lines = text.split("\n")
        return "\n".join(
            [
                line.replace(self.filter_tokens[0], "")
                for line in lines
                if line.split(" ")[0] not in self.ignore_line_tokens
            ]
        )

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
        return self.flatten(observations), [ob["emote"] for ob in observations]

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

    def reset(self):
        super().reset()
        self.history = []
        self.last_label = None
        self.persona = None
        self.resp_queue = []
        self.utt_queue = []
        self.noise_happened = False
        self.unknown_happened = False
