from typing import Dict, Any, Tuple

import string

from parlai.core.agents import Agent
from parlai.core.message import Message
from random import sample


import pathlib

path = pathlib.Path(__file__).parent.absolute()


class LightSelfplayBaseMixin(Agent):
    """Abstract class that handles parsing Light dialogue into 2 expert trajectories for imitation learning (2 sides of the dialogue).
    """

    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "LightChatbotBase"

        self.eval_step = 0
        self.train_step = 0

        self.persona_neutral = None
        self.persona_1_name = None
        self.persona_2_name = None
        self.persona_1_desc = None
        self.persona_2_desc = None
        self.history = []
        self.last_label = None
        self.last_history = None

        self.utt_queue = []
        self.resp_queue = []

        self.self_speaker_token = "<speaker_self>"

        self.other_speaker_token = "<speaker_other>"

        self.ctx_tokens = [
            "_setting_name",
            "_setting_desc",
            "_partner_name",
            "_self_name",
            "_self_persona",
            "_other_persona",
        ]

        self.neutral_ctx_tokens = [
            "_setting_name",
            "_setting_desc",
        ]

        self.self_ctx_tokens = [
            "_self_name",
            "_self_persona",
        ]

        self.other_ctx_tokens = [
            "_partner_name",
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

        res = dict(observation)

        if not self.persona_neutral:
            res["text"] = self._extract_persona(observation["text"])

        res["text"], res["emote"] = self._extract_emote(res["text"])

        res["text"] = self._filter_tokens(res["text"])

        if self.last_label is not None:
            self.history.append(self.self_speaker_token + self.last_label)

        self._add_utterance(res)

        self.episode_done = observation["episode_done"]
        if self.episode_done:
            self.reset()

        res["text"] = [
            (
                self._construct_persona(
                    self.persona_neutral,
                    self.persona_1_name,
                    self.persona_1_desc,
                    self.persona_2_name,
                    self.persona_2_desc,
                ),
                self.history + [self.self_speaker_token + self.last_label],
            ),
        ]
        res["text"].append(
            (
                self._construct_persona(
                    self.persona_neutral,
                    self.persona_2_name,
                    self.persona_2_desc,
                    self.persona_1_name,
                    self.persona_1_desc,
                ),
                self._convert_history_to_other(self.history),
            )
        )
        return res

    def _add_utterance(self, res):
        if self.utt_queue:
            if res["text"] != "":
                self.utt_queue.append(res["text"])
            self.resp_queue.append(
                res["labels"][0] if "labels" in res else res["eval_labels"][0]
            )

            self.history.append(self.other_speaker_token + self.utt_queue.pop(0))

            self.last_label = self.resp_queue.pop(0)
        else:
            if res["text"] != "":
                self.history.append(self.other_speaker_token + res["text"])

            self.last_label = (
                res["labels"][0] if "labels" in res else res["eval_labels"][0]
            )

    def _extract_persona(self, text):
        lines = text.split("\n")

        self.persona_neutral = "\n".join(
            [line for line in lines if line.split(" ")[0] in self.neutral_ctx_tokens]
        )

        self.persona_1_name = [
            line.replace(self.self_ctx_tokens[0], "")
            for line in lines
            if line.split(" ")[0] in self.self_ctx_tokens[0]
        ][0]
        self.persona_1_desc = [
            line.replace(self.self_ctx_tokens[1], "")
            for line in lines
            if line.split(" ")[0] in self.self_ctx_tokens[1]
        ][0]

        self.persona_2_name = [
            line.replace(self.other_ctx_tokens[0], "")
            for line in lines
            if line.split(" ")[0] in self.other_ctx_tokens[0]
        ][0]
        self.persona_2_desc = [
            line.replace(self.other_ctx_tokens[1], "")
            for line in lines
            if line.split(" ")[0] in self.other_ctx_tokens[1]
        ][0]

        if not self.persona_neutral:
            raise ValueError("Tried to parse persona but none found")

        return "\n".join(
            [line for line in lines if line.split(" ")[0] not in self.ctx_tokens]
        )

    def _construct_persona(
        self,
        persona_neutral,
        persona_self_name,
        persona_self_desc,
        persona_other_name,
        persona_other_desc,
    ):
        return (
            persona_neutral
            + self.generator_policy.tokenizer.sep_token
            + "\n"
            + self.self_speaker_token
            + self.self_ctx_tokens[0]
            + persona_self_name
            + "\n"
            + self.self_ctx_tokens[1]
            + persona_self_desc
            + self.generator_policy.tokenizer.sep_token
            + "\n"
            + self.other_speaker_token
            + self.other_ctx_tokens[0]
            + persona_other_name
            + "\n"
            + self.other_ctx_tokens[1]
            + persona_other_desc
        )

    def _convert_history_to_other(self, history):
        history = [
            turn.replace(self.self_speaker_token, self.other_speaker_token)
            if self.self_speaker_token in turn
            else turn.replace(self.other_speaker_token, self.self_speaker_token)
            for turn in history
        ]
        return history

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

    def batch_act(self, observations):
        # Add generated histories to data ones
        imitate_first = []
        sample_first = []
        imitate_second = []
        sample_second = []
        for observation in observations:
            len(observation["text"])
        utterances = self.batch_sample(imitate_first, sample_first)
        self._update_histories(utterances)
        utterances = self.batch_sample(imitate_second, sample_second)
        self._update_histories(utterances)
        self.batch_update()
        # Update generated histories with generated utterances

    def batch_sample(self, imitate, sample) -> Dict[int, str]:
        """Implement sampling utterances and memorizing """
        pass

    def batch_update(self):
        """Implement update here"""
        pass

    def _update_histories(self, utterances):
        pass

    def reset(self):
        super().reset()
        self.history = []
        self.last_label = None
        self.persona_neutral = None
        self.persona_1 = None
        self.persona_2 = None
        self.resp_queue = []
        self.utt_queue = []
        self.noise_happened = False
        self.unknown_happened = False
