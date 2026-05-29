"""Modular reward components for drone navigation (configurable weights via YAML)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from RL.framework.config.schemas import RewardWeights


@dataclass
class RewardContext:
    """Per-step signals extracted from env info and observation."""

    distance_to_goal: float
    prev_distance_to_goal: float
    collision: bool
    success: bool
    min_clearance: float
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    action: np.ndarray
    prev_action: Optional[np.ndarray]
    alive: bool = True


def goal_progress_reward(ctx: RewardContext) -> float:
    delta = ctx.prev_distance_to_goal - ctx.distance_to_goal
    d_norm = float(np.clip(ctx.distance_to_goal / max(ctx.prev_distance_to_goal, 1e-3), 0.0, 1.0))
    return float(np.tanh(delta * (1.0 + d_norm)))


def collision_penalty(ctx: RewardContext) -> float:
    return -1.0 if ctx.collision else 0.0


def obstacle_proximity_penalty(ctx: RewardContext, danger: float = 1.0, safe: float = 3.0) -> float:
    c = ctx.min_clearance
    if c >= safe:
        return 0.0
    if c <= danger:
        return -1.0
    return -float((safe - c) / (safe - danger))


def hover_stability_reward(ctx: RewardContext, hover_dist: float = 2.0) -> float:
    if ctx.distance_to_goal > hover_dist:
        return 0.0
    speed = float(np.linalg.norm(ctx.linear_velocity))
    return float(np.clip(1.0 - speed / 1.5, 0.0, 1.0))


def smooth_action_penalty(ctx: RewardContext) -> float:
    if ctx.prev_action is None:
        return 0.0
    return -float(np.linalg.norm(ctx.action - ctx.prev_action))


def angular_velocity_penalty(ctx: RewardContext, max_rate: float = 3.141) -> float:
    rate = float(np.linalg.norm(ctx.angular_velocity))
    return -float(np.clip(rate / max_rate, 0.0, 1.0))


def alive_reward(ctx: RewardContext) -> float:
    return 1.0 if ctx.alive and not ctx.collision else 0.0


def energy_penalty(ctx: RewardContext) -> float:
    return -float(np.linalg.norm(ctx.action))


COMPONENT_FN = {
    "goal_progress": goal_progress_reward,
    "collision": collision_penalty,
    "obstacle_proximity": obstacle_proximity_penalty,
    "hover_stability": hover_stability_reward,
    "smooth_action": smooth_action_penalty,
    "angular_velocity": angular_velocity_penalty,
    "alive": alive_reward,
    "energy": energy_penalty,
}


def compute_shaped_reward(ctx: RewardContext, weights: RewardWeights) -> tuple[float, dict[str, float]]:
    """Weighted sum of modular components; returns total and per-component breakdown."""
    breakdown: dict[str, float] = {}
    total = 0.0
    for name, fn in COMPONENT_FN.items():
        w = getattr(weights, name, 0.0)
        if w == 0.0:
            continue
        val = float(fn(ctx))
        breakdown[name] = val
        total += w * val
    return total, breakdown


def context_from_step(
    *,
    info: dict[str, Any],
    state: np.ndarray,
    action: np.ndarray,
    prev_action: Optional[np.ndarray],
    prev_distance: float,
) -> RewardContext:
    dist = float(info.get("distance_to_goal", prev_distance))
    return RewardContext(
        distance_to_goal=dist,
        prev_distance_to_goal=prev_distance,
        collision=bool(info.get("collision", False)),
        success=bool(info.get("success", False)),
        min_clearance=float(info.get("min_clearance", 10.0)),
        linear_velocity=np.asarray(state[6:9], dtype=np.float32),
        angular_velocity=np.asarray(state[9:12], dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        prev_action=prev_action,
    )
