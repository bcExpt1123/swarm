"""Evaluation metrics aligned with validator scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    reward: float = 0.0
    length: int = 0
    success: bool = False
    collision: bool = False
    final_score: float = 0.0
    path_length: float = 0.0
    start_to_goal: float = 1.0
    action_deltas: list[float] = field(default_factory=list)
    energy: float = 0.0


@dataclass
class EvalSummary:
    success_rate: float
    collision_rate: float
    mean_reward: float
    mean_length: float
    path_efficiency: float
    action_smoothness: float
    mean_energy: float
    mean_validator_score: float

    def composite_objective(self, weights: dict[str, float]) -> float:
        obj = 0.0
        mapping = {
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "smoothness": self.action_smoothness,
            "mean_reward": self.mean_reward / max(abs(self.mean_reward), 1.0),
            "path_efficiency": self.path_efficiency,
        }
        for k, w in weights.items():
            obj += w * mapping.get(k, 0.0)
        return float(obj)


def aggregate_episodes(episodes: list[EpisodeMetrics]) -> EvalSummary:
    n = max(len(episodes), 1)
    successes = sum(1 for e in episodes if e.success)
    collisions = sum(1 for e in episodes if e.collision)
    efficiencies = []
    for e in episodes:
        ideal = max(e.start_to_goal, 1e-3)
        efficiencies.append(float(np.clip(ideal / max(e.path_length, ideal), 0.0, 1.0)))
    smooth = []
    for e in episodes:
        if e.action_deltas:
            smooth.append(1.0 - float(np.mean(e.action_deltas)))
        else:
            smooth.append(1.0)
    return EvalSummary(
        success_rate=successes / n,
        collision_rate=collisions / n,
        mean_reward=float(np.mean([e.reward for e in episodes])),
        mean_length=float(np.mean([e.length for e in episodes])),
        path_efficiency=float(np.mean(efficiencies)),
        action_smoothness=float(np.mean(smooth)),
        mean_energy=float(np.mean([e.energy for e in episodes])),
        mean_validator_score=float(np.mean([e.final_score for e in episodes])),
    )


def update_episode_from_info(
    ep: EpisodeMetrics,
    *,
    reward: float,
    info: dict[str, Any],
    action: np.ndarray,
    prev_action: np.ndarray | None,
    pos: np.ndarray,
) -> None:
    ep.reward += reward
    ep.length += 1
    ep.success = ep.success or bool(info.get("success", False))
    ep.collision = ep.collision or bool(info.get("collision", False))
    ep.final_score = float(info.get("score", ep.final_score))
    if prev_action is not None:
        ep.action_deltas.append(float(np.linalg.norm(action - prev_action)))
    ep.energy += float(np.linalg.norm(action))
    if ep.path_length == 0.0:
        ep.start_to_goal = float(info.get("distance_to_goal", 1.0))
    if ep.length > 1:
        ep.path_length += float(np.linalg.norm(pos - ep._last_pos))  # type: ignore[attr-defined]
    ep._last_pos = pos.copy()  # type: ignore[attr-defined]
