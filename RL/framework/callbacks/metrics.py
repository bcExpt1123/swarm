"""Episode and training metric callbacks."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """Log rolling mean episode return and length to TensorBoard / W&B."""

    def __init__(self, verbose: int = 0, window: int = 100):
        super().__init__(verbose)
        self.window = window
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_count = 0
        self._started = False

    def _on_step(self) -> bool:
        if not self._started:
            self._started = True
            print("PPO rollout started (first steps are slow while envs warm up)…", flush=True)
        infos = self.locals.get("infos", [])
        for info in infos:
            if info and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                self.episode_count += 1
                if self.logger is not None:
                    w = self.window
                    self.logger.record(
                        "episode/mean_reward",
                        float(np.mean(self.episode_rewards[-w:])),
                    )
                    self.logger.record(
                        "episode/mean_length",
                        float(np.mean(self.episode_lengths[-w:])),
                    )
                    self.logger.record("episode/count", float(self.episode_count))
        return True
