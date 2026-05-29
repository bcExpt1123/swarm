"""Additional Gymnasium wrappers."""

from __future__ import annotations

import gymnasium as gym


class RecordEpisodeStatsWrapper(gym.Wrapper):
    """Ensures episode-level stats are present in info for callbacks."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_len = 0

    def reset(self, *, seed=None, options=None):
        self._ep_reward = 0.0
        self._ep_len = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += float(reward)
        self._ep_len += 1
        if terminated or truncated:
            info = dict(info)
            info["episode_reward"] = self._ep_reward
            info["episode_length"] = self._ep_len
        return obs, reward, terminated, truncated, info
