"""YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

from RL.framework.config.schemas import (
    AIRLConfig,
    BCConfig,
    CheckpointConfig,
    CurriculumConfig,
    CurriculumStage,
    EvaluationConfig,
    LoggingConfig,
    OptunaConfig,
    PolicyConfig,
    PPOConfig,
    RewardWeights,
    TrainingConfig,
    VecEnvConfig,
)

T = TypeVar("T")


def _merge_dataclass(cls: Type[T], data: dict[str, Any] | None) -> T:
    if not data:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in field_names}
    if cls is CurriculumConfig and "stages" in filtered:
        filtered["stages"] = [
            CurriculumStage(**s) if isinstance(s, dict) else s for s in filtered["stages"]
        ]
    return cls(**filtered)


def load_config(path: str | Path) -> TrainingConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return TrainingConfig(
        seed=int(raw.get("seed", 2)),
        total_timesteps=int(raw.get("total_timesteps", 1_000_000)),
        device=str(raw.get("device", "auto")),
        resume_path=raw.get("resume_path"),
        reward=_merge_dataclass(RewardWeights, raw.get("reward")),
        curriculum=_merge_dataclass(CurriculumConfig, raw.get("curriculum")),
        ppo=_merge_dataclass(PPOConfig, raw.get("ppo")),
        policy=_merge_dataclass(PolicyConfig, raw.get("policy")),
        vec_env=_merge_dataclass(VecEnvConfig, raw.get("vec_env")),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging")),
        checkpoint=_merge_dataclass(CheckpointConfig, raw.get("checkpoint")),
        evaluation=_merge_dataclass(EvaluationConfig, raw.get("evaluation")),
        airl=_merge_dataclass(AIRLConfig, raw.get("airl")),
        bc=_merge_dataclass(BCConfig, raw.get("bc")),
        optuna=_merge_dataclass(OptunaConfig, raw.get("optuna")),
        extra=raw.get("extra", {}),
    )
