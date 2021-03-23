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
        self.train_step = 0
        self.self_speaker_token = "<speaker_self>"

        self.other_speaker_token = "<speaker_other>"

        self.personas = [None] * opt["batchsize"]
        self.histories = [None] * opt["batchsize"]

    def act(self):
        raise NotImplementedError()

    def batch_act(self, observations):
        self.train_step += 1
        # Add generated histories to data ones
        imitate = []
        sample = []
        for i, observation in enumerate(observations):
            if (
                self.personas[i] is None
                or self.personas[i] != observation["text"][0][0]
            ):
                self.personas[i] = observation["text"][0][0]
                self.histories[i] = []

            sample.append((self.personas[i], self.histories[i],))
            imitate.extend(
                [dialog for dialog in observation["text"] if len(dialog[1]) > 0]
            )

        self.batch_imitate(imitate)

        utterances = self.batch_sample(sample)
        self._update_histories(utterances)
        print(self.histories)

        sample = []
        for i, observation in enumerate(observations):
            sample.append(
                (
                    observation["text"][1][0],
                    self._convert_history_to_other(self.histories[i]),
                )
            )

        utterances = self.batch_sample(sample)
        self._update_histories(utterances, other=True)
        print("upd2", self.histories)
        self.batch_update()
        if (
            self.train_step % self.episode_num_dialog_dump == 0
        ) and self.train_step != 0:
            self.checkpoint(sample)
        return [{"id": self.id} for _ in observations]

    def batch_imitate(self, dialogs):
        """Implement sampling utterances and memorization here"""
        pass

    def batch_sample(self, dialogs) -> List[str]:
        """Implement update here"""
        pass

    def batch_update(self):
        """Update weights here"""
        pass

    def _update_histories(self, utterances, other=False):
        for i in range(len(utterances)):
            history = self.histories[i]
            history.append(
                (self.self_speaker_token if not other else self.other_speaker_token)
                + utterances[i]
            )
            self.histories[i] = history

    def _convert_history_to_other(self, history):
        history = [
            turn.replace(self.self_speaker_token, self.other_speaker_token)
            if self.self_speaker_token in turn
            else turn.replace(self.other_speaker_token, self.self_speaker_token)
            for turn in history
        ]
        return history
