"""Curriculum progression callback."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from RL.framework.callbacks.eval_utils import run_training_eval
from RL.framework.curriculum.manager import CurriculumManager


class CurriculumCallback(BaseCallback):
    """Periodically evaluate and advance curriculum stage."""

    def __init__(
        self,
        curriculum: CurriculumManager,
        config,
        eval_freq: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.config = config
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self.eval_freq:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True
        n_ep = min(self.curriculum.stage.eval_episodes, 15)
        if self.verbose:
            print(
                f"[curriculum eval] {self.num_timesteps} steps — "
                f"running {n_ep} episodes…",
                flush=True,
            )
        summary = run_training_eval(
            self.model,
            self.config,
            n_episodes=n_ep,
            label="curriculum eval",
        )
        if summary is None:
            return True
        advanced = self.curriculum.record_eval(summary.success_rate)
        if self.logger is not None:
            for k, v in self.curriculum.metrics().items():
                self.logger.record(k, v)
            self.logger.record("eval/success_rate", summary.success_rate)
            self.logger.record("eval/collision_rate", summary.collision_rate)
            self.logger.record("eval/mean_reward", summary.mean_reward)
        if advanced and self.verbose:
            print(f"Curriculum advanced to stage: {self.curriculum.stage.name}", flush=True)
        return True
