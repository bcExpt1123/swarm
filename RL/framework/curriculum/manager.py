"""Stage-based curriculum with automatic progression."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from RL.framework.config.schemas import CurriculumConfig, CurriculumStage


@dataclass
class CurriculumState:
    stage_index: int = 0
    episodes_at_stage: int = 0
    success_history: list[float] = field(default_factory=list)


class CurriculumManager:
    """Selects challenge types and env randomization from the active stage."""

    def __init__(self, config: CurriculumConfig, base_seed: int = 0):
        self.config = config
        self.base_seed = base_seed
        self.state = CurriculumState()
        if not config.stages:
            config.stages = [
                CurriculumStage(name="default", challenge_types=[1, 2], min_success_rate=0.0)
            ]

    @property
    def stage(self) -> CurriculumStage:
        return self.config.stages[min(self.state.stage_index, len(self.config.stages) - 1)]

    def task_seed(self, rank: int, global_step: int) -> int:
        stage = self.stage
        noise = random.randint(0, 10_000) if self.config.randomize_env_params else 0
        return self.base_seed + rank * 7919 + stage.seed_offset + global_step + noise

    def sample_challenge_type(self, rng: random.Random) -> int:
        types = self.stage.challenge_types
        return rng.choice(types)

    def record_eval(self, success_rate: float) -> bool:
        """Record evaluation outcome; return True if stage advanced."""
        self.state.success_history.append(success_rate)
        self.state.episodes_at_stage += 1
        if not self.config.auto_advance or not self.config.enabled:
            return False
        if self.state.stage_index >= len(self.config.stages) - 1:
            return False
        stage = self.stage
        if len(self.state.success_history) < stage.min_episodes:
            return False
        recent = self.state.success_history[-stage.min_episodes :]
        if float(sum(recent) / len(recent)) >= stage.min_success_rate:
            self.state.stage_index += 1
            self.state.episodes_at_stage = 0
            self.state.success_history.clear()
            return True
        return False

    def metrics(self) -> dict[str, float]:
        return {
            "curriculum/stage_index": float(self.state.stage_index),
            "curriculum/stage_name": float(hash(self.stage.name) % 10_000),
            "curriculum/episodes_at_stage": float(self.state.episodes_at_stage),
        }
