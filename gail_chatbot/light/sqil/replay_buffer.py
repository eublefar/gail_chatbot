import numpy as np
from typing import Dict, Tuple, Iterable
from copy import deepcopy
import gc


class ReplayBuffer:

    """A simple numpy replay buffer."""

    def __init__(
        self, size: int, batch_size: int = 32, gamma: float = 0.99, n_envs: int = 1
    ):
        self.obs_buf = np.zeros([size], dtype=np.object)
        self.next_obs_buf = np.zeros([size], dtype=np.object)
        self.acts_buf = np.zeros([size], dtype=np.object)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.gamma = gamma

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return transition

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=np.stack(list(self.obs_buf[idxs])),
            next_obs=np.stack(list(self.next_obs_buf[idxs])),
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )

    def sample_iterator(self) -> Iterable[Dict[str, np.ndarray]]:
        iters = (self.size // self.batch_size) + int(bool(self.size % self.batch_size))
        ids = np.arange(self.size)
        ids = np.random.permutation(ids)
        for i in range(iters):
            upper = (i + 1) * self.batch_size
            lower = i * self.batch_size
            idxs = ids[lower:upper]
            yield dict(
                obs=np.stack(list(self.obs_buf[idxs])),
                next_obs=np.stack(list(self.next_obs_buf[idxs])),
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                done=self.done_buf[idxs],
                # for N-step Learning
                indices=idxs,
            )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size

    def empty(self):
        self.obs_buf = np.zeros([self.max_size], dtype=np.object)
        self.next_obs_buf = np.zeros([self.max_size], dtype=np.object)
        self.acts_buf = np.zeros([self.max_size], dtype=np.object)
        self.rews_buf = np.zeros([self.max_size], dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        gc.collect()
        self.ptr, self.size, = 0, 0
