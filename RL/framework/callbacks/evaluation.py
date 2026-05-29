"""Periodic evaluation during PPO training."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from RL.framework.callbacks.eval_utils import run_training_eval


class EvalCallback(BaseCallback):
    def __init__(self, config, eval_freq: int, n_episodes: int | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps < self.eval_freq:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True
        n_ep = self.n_episodes or self.config.evaluation.n_episodes
        if self.verbose:
            print(
                f"[eval] {self.num_timesteps} steps — running {min(n_ep, 15)} episodes…",
                flush=True,
            )
        summary = run_training_eval(
            self.model,
            self.config,
            n_episodes=n_ep,
            label="eval",
        )
        if summary is None:
            return True
        if self.logger is not None:
            self.logger.record("eval/success_rate", summary.success_rate)
            self.logger.record("eval/collision_rate", summary.collision_rate)
            self.logger.record("eval/mean_reward", summary.mean_reward)
            self.logger.record("eval/path_efficiency", summary.path_efficiency)
            self.logger.record("eval/action_smoothness", summary.action_smoothness)
            self.logger.record("eval/mean_energy", summary.mean_energy)
            self.logger.record("eval/validator_score", summary.mean_validator_score)
        if self.verbose:
            print(
                f"[eval] success={summary.success_rate:.2%} "
                f"collision={summary.collision_rate:.2%} "
                f"reward={summary.mean_reward:.3f}",
                flush=True,
            )
        return True
