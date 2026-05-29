"""Modular RL training framework for Swarm MovingDroneAviary."""

from RL.framework.config.loader import load_config
from RL.framework.config.schemas import TrainingConfig

__all__ = ["TrainingConfig", "load_config"]
