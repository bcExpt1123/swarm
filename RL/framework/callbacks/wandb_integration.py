"""Weights & Biases integration for SB3."""

from __future__ import annotations

from typing import Any, Optional

from stable_baselines3.common.callbacks import BaseCallback

from RL.framework.config.schemas import LoggingConfig


def init_wandb(logging: LoggingConfig, config_dict: dict[str, Any]) -> Optional[Any]:
    if not logging.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required; pip install wandb") from exc

    return wandb.init(
        project=logging.wandb_project,
        entity=logging.wandb_entity,
        name=logging.wandb_run_name,
        config=config_dict,
    )


class WandbCallback(BaseCallback):
    """Mirror SB3 logger scalars to W&B."""

    def __init__(self, run: Any, log_freq: int = 100):
        super().__init__()
        self.run = run
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0 or self.logger is None:
            return True
        import wandb

        for key, value in self.logger.name_to_value.items():
            wandb.log({key: value}, step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.run is not None:
            self.run.finish()
