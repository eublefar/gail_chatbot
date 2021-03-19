from typing import Dict, Any, List

import string

from parlai.core.agents import Agent
from parlai.core.message import Message
from random import sample


import pathlib

path = pathlib.Path(__file__).parent.absolute()


class LightImitateMixin(Agent):
    """Abstract class that handles passing expert trajectories alongside self-play sampling
    """

    def __init__(self, opt: Dict[str, Any], shared: Dict[str, Any] = None):
        self.id = "LightChatbotSelfPlay"
        self.self_speaker_token = "<speaker_self>"

        self.other_speaker_token = "<speaker_other>"

        self.persona_dict = {}
        self.history_dict = {}

    def act(self):
        raise NotImplementedError()

    def batch_act(self, observations):
        # Add generated histories to data ones
        imitate = []
        sample = []
        ids = []
        for observation in observations:
            if (
                observation["id"] not in self.persona_dict
                or self.persona_dict[observation["id"]] != observation["text"][0][0]
            ):
                self.persona_dict[observation["id"]] = observation["text"][0][0]
                self.history_dict[observation["id"]] = []

            sample.append(
                (
                    self.persona_dict.get(observation["id"], observation["text"][0][0]),
                    self.history_dict.get(observation["id"], []),
                )
            )
            imitate.extend(observation["text"])
            ids.append(observation["id"])

        utterances = self.batch_imitate(imitate)

        utterances = self.batch_sample(sample)
        self._update_histories(ids, utterances)

        sample = []
        for observation in observations:
            sample.append(
                (
                    observation["text"][1][0],
                    self._convert_history_to_other(
                        self.history_dict.get(observation["id"], [])
                    ),
                )
            )

        utterances = self.batch_sample(sample)
        self._update_histories(ids, utterances, other=True)

        self.batch_update()

    def batch_imitate(self, dialogs):
        """Implement sampling utterances and memorization here"""
        pass

    def batch_sample(self, dialogs) -> List[str]:
        """Implement update here"""
        pass

    def batch_update(self):
        """Update weights here"""
        pass

    def _update_histories(self, ids, utterances, other=False):
        for i, id in enumerate(ids):
            history = self.history_dict.get(id, [])
            history.append(
                self.self_speaker_token
                if not other
                else self.other_speaker_token + utterances[i]
            )
            self.history_dict[id] = history

    def _convert_history_to_other(self, history):
        history = [
            turn.replace(self.self_speaker_token, self.other_speaker_token)
            if self.self_speaker_token in turn
            else turn.replace(self.other_speaker_token, self.self_speaker_token)
            for turn in history
        ]
        return history
