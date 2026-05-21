"""Replacement for imitation's ReplayBuffer when ``venv.observation_space`` is a Dict."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common import vec_env as vec_env_mod

from imitation.data import types


class DictObservationReplayBuffer:
    """FIFO buffer (max ``capacity`` rows) storing DictObs slices per timestep."""

    def __init__(self, capacity: int, venv: vec_env_mod.VecEnv):
        self.capacity = int(capacity)
        self._rows: list[tuple] = []

    @staticmethod
    def _row_obs(obs_batch, i: int):
        if isinstance(obs_batch, types.DictObs):
            return obs_batch[i]
        return obs_batch[i]

    def store(self, transitions: types.Transitions, truncate_ok: bool = True) -> None:
        td = types.dataclass_quick_asdict(transitions)
        n = len(td["acts"])
        if n == 0:
            raise ValueError("Trying to store empty data.")
        if n > self.capacity and not truncate_ok:
            raise ValueError("Not enough capacity to store data.")
        if n > self.capacity:
            start = n - self.capacity
            td = {k: v[start:] for k, v in td.items()}
            n = self.capacity

        for i in range(n):
            self._rows.append(
                (
                    self._row_obs(td["obs"], i),
                    self._row_obs(td["next_obs"], i),
                    td["acts"][i],
                    td["dones"][i],
                    td["infos"][i],
                )
            )
        overflow = len(self._rows) - self.capacity
        if overflow > 0:
            del self._rows[:overflow]

    def sample(self, n_samples: int) -> types.Transitions:
        if not self._rows:
            raise ValueError("Buffer is empty")
        ind = np.random.randint(len(self._rows), size=n_samples)
        obs_list = [self._rows[i][0] for i in ind]
        next_list = [self._rows[i][1] for i in ind]
        acts_b = np.stack([self._rows[i][2] for i in ind], axis=0)
        dones_b = np.array([self._rows[i][3] for i in ind], dtype=bool)
        infos_b = np.array([self._rows[i][4] for i in ind], dtype=object)
        # Stored rows are single-sample DictObs (no batch dim), so stack to add the
        # batch axis back. ``DictObs.concatenate`` would join along existing axes and
        # break for keys with differing per-sample ranks (e.g. depth vs. state).
        return types.Transitions(
            obs=types.DictObs.stack(obs_list),
            acts=acts_b,
            next_obs=types.DictObs.stack(next_list),
            dones=dones_b,
            infos=infos_b,
        )

    def size(self) -> int:
        return len(self._rows)


def patch_imitation_replay_buffer_for_dict_obs() -> None:
    """Monkey-patch ``imitation.data.buffer.ReplayBuffer`` before constructing AIRL."""
    import gymnasium.spaces as spaces
    import imitation.data.buffer as im_buffer

    Original = im_buffer.ReplayBuffer

    def ReplayBuffer(capacity, venv=None, **kwargs):  # noqa: N802 — factory API
        if venv is not None and isinstance(venv.observation_space, spaces.Dict):
            return DictObservationReplayBuffer(capacity, venv)
        return Original(capacity, venv, **kwargs)

    im_buffer.ReplayBuffer = ReplayBuffer  # type: ignore[misc, assignment]
