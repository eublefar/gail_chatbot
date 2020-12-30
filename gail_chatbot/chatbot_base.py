from typing import Dict, Tuple, List, Any

import string

from parlai.core.agents import Agent
from parlai.core.message import Message
from random import randint, sample, uniform, choice

from gail_chatbot.phrases import UNCERTAINTY_PHRASES


class ConvaiChatbotBase(Agent):
    """Abstract class that handles parsing Convai task into dialogue batches.
    """

    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "ConvaiChatbotBase"
        if opt["task"] != "convai2":
            raise ValueError("Only works on convai task")

        self.eval_step = 0
        self.train_step = 0

        self.persona = None
        self.history = []
        self.last_label = None
        self.last_history = None

        self.noise_frac = 0.3
        self.distractor_frac = 0.6

        self.utt_queue = []
        self.resp_queue = []

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

        if self.last_label is not None:
            self.history.append(self.last_label)

        if uniform(0, 1) < self.noise_frac:

            if uniform(0, 1) < self.distractor_frac:
                self.history.append(neg_sample[1])
            else:
                letters = string.ascii_letters
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
                observation["labels"][0]
                if "labels" in observation
                else observation["eval_labels"][0]
            )
        else:
            if self.utt_queue:
                self.utt_queue.append(res["text"])
                self.resp_queue.append(
                    observation["labels"][0]
                    if "labels" in observation
                    else observation["eval_labels"][0]
                )

                self.history.append(self.utt_queue.pop(0))

                self.last_label = self.resp_queue.pop(0)
            else:
                self.history.append(res["text"])

                self.last_label = (
                    observation["labels"][0]
                    if "labels" in observation
                    else observation["eval_labels"][0]
                )
        self.episode_done = observation["episode_done"]
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
        return self.flatten(observations)

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
