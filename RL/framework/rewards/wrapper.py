"""Gymnasium wrapper applying modular reward shaping on top of MovingDroneAviary."""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from RL.framework.config.schemas import RewardWeights
from RL.framework.rewards.components import compute_shaped_reward, context_from_step


class ModularRewardWrapper(gym.Wrapper):
    """Adds YAML-weighted shaping terms; optionally scales native ``flight_reward`` increment."""

    def __init__(self, env: gym.Env, weights: RewardWeights):
        super().__init__(env)
        self.weights = weights
        self._prev_action: Optional[np.ndarray] = None
        self._prev_distance: float = float("inf")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_action = None
        self._prev_distance = float(info.get("distance_to_goal", np.linalg.norm(self._goal_from_obs(obs))))
        return obs, info

    @staticmethod
    def _goal_from_obs(obs: dict) -> np.ndarray:
        state = obs["state"]
        pos = state[0:3]
        search = state[-3:]
        return pos + search

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        state = np.asarray(obs["state"], dtype=np.float32)
        ctx = context_from_step(
            info=info,
            state=state,
            action=action,
            prev_action=self._prev_action,
            prev_distance=self._prev_distance,
        )
        shaped, breakdown = compute_shaped_reward(ctx, self.weights)
        reward = shaped
        if self.weights.use_env_reward:
            reward += self.weights.env_reward_scale * float(env_reward)

        info = dict(info)
        info["reward_breakdown"] = breakdown
        info["shaped_reward"] = shaped
        info["env_reward"] = float(env_reward)

        self._prev_action = np.asarray(action, dtype=np.float32).copy()
        self._prev_distance = ctx.distance_to_goal
        return obs, float(reward), terminated, truncated, info
