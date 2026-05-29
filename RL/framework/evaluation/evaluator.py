"""Rollout-based policy evaluation (single env, no VecNormalize training mode)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from gym_pybullet_drones.utils.enums import ActionType
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from RL.framework.config.schemas import TrainingConfig
from RL.framework.evaluation.metrics import (
    EpisodeMetrics,
    EvalSummary,
    aggregate_episodes,
    update_episode_from_info,
)
from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


def policy_has_nan(model: BaseAlgorithm) -> bool:
    """True if any policy parameter is non-finite (training diverged)."""
    for param in model.policy.parameters():
        if not torch.isfinite(param).all():
            return True
    return False


def _sanitize_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        k: np.nan_to_num(
            np.asarray(v, dtype=np.float32),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        for k, v in obs.items()
    }


def _normalize_obs_for_model(
    model: BaseAlgorithm,
    obs: dict[str, np.ndarray],
    *,
    norm_obs: Optional[bool] = None,
) -> Any:
    """Apply VecNormalize stats only when training used ``norm_obs``."""
    obs = _sanitize_obs(obs)
    if norm_obs is False:
        return obs
    venv = model.get_env()
    use_norm = norm_obs if norm_obs is not None else (
        isinstance(venv, VecNormalize) and bool(venv.norm_obs)
    )
    if not use_norm:
        return obs
    batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}
    return venv.normalize_obs(batched)


def _format_action_for_env(env, action: np.ndarray) -> np.ndarray:
    """BaseRLAviary expects shape (n_drones, action_dim), not a flat (5,) vector."""
    act = np.clip(
        np.asarray(action, dtype=np.float32).flatten(),
        env.action_space.low.flatten(),
        env.action_space.high.flatten(),
    )
    if not np.isfinite(act).all():
        act = np.zeros_like(act)
    if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
        norm = max(float(np.linalg.norm(act[:3])), 1e-6)
        act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
        act = np.clip(act, env.action_space.low.flatten(), env.action_space.high.flatten())
    return act[None, :]


def _predict_action(
    model: BaseAlgorithm,
    obs: dict[str, np.ndarray],
    *,
    deterministic: bool,
    norm_obs: Optional[bool] = None,
) -> np.ndarray:
    if policy_has_nan(model):
        raise RuntimeError(
            "Policy weights contain NaN/Inf — training diverged. "
            "Resume from an earlier checkpoint (e.g. ppo_swarm_*_steps.zip) "
            "and lower learning_rate / vf_coef (0.15 for custom CNN)."
        )
    obs_in = _normalize_obs_for_model(model, obs, norm_obs=norm_obs)
    raw, _ = model.predict(obs_in, deterministic=deterministic)
    action = np.asarray(raw, dtype=np.float32).flatten()
    if not np.isfinite(action).all():
        raise RuntimeError("Policy returned non-finite actions for this observation.")
    return action


def evaluate_model(
    model: BaseAlgorithm,
    config: TrainingConfig,
    *,
    n_episodes: int | None = None,
    deterministic: bool = True,
    norm_obs: Optional[bool] = None,
) -> EvalSummary:
    n_episodes = n_episodes or config.evaluation.n_episodes
    if norm_obs is None:
        norm_obs = config.vec_env.norm_obs
    episodes: list[EpisodeMetrics] = []

    for ep_i in range(n_episodes):
        task = random_task(sim_dt=SIM_DT, seed=config.seed + 10_000 + ep_i)
        env = make_env(task, gui=False)
        try:
            obs, _ = env.reset()
            obs = _sanitize_obs(obs)
            ep = EpisodeMetrics()
            ep._last_pos = np.asarray(obs["state"][0:3], dtype=np.float32)  # type: ignore[attr-defined]
            prev_action = None
            done = False
            while not done:
                raw = _predict_action(
                    model, obs, deterministic=deterministic, norm_obs=norm_obs
                )
                action_1d = raw.flatten()
                obs, reward, terminated, truncated, info = env.step(
                    _format_action_for_env(env, action_1d)
                )
                obs = _sanitize_obs(obs)
                pos = np.asarray(obs["state"][0:3], dtype=np.float32)
                update_episode_from_info(
                    ep,
                    reward=float(reward),
                    info=info,
                    action=action_1d,
                    prev_action=prev_action,
                    pos=pos,
                )
                prev_action = action_1d.copy()
                done = terminated or truncated
            episodes.append(ep)
        finally:
            env.close()

    return aggregate_episodes(episodes)


def evaluate_vec_model(
    model: BaseAlgorithm,
    venv: VecEnv,
    config: TrainingConfig,
    *,
    n_episodes: int | None = None,
) -> EvalSummary:
    """Evaluate using the training vec env (faster, approximate)."""
    n_episodes = n_episodes or config.evaluation.n_episodes
    episodes: list[EpisodeMetrics] = []
    obs = venv.reset()
    ep_states = [EpisodeMetrics() for _ in range(venv.num_envs)]
    completed = 0
    prev_actions: list[np.ndarray | None] = [None] * venv.num_envs

    while completed < n_episodes:
        actions, _ = model.predict(obs, deterministic=config.evaluation.deterministic)
        obs, rewards, dones, infos = venv.step(actions)
        for i in range(venv.num_envs):
            ep = ep_states[i]
            info = infos[i] if isinstance(infos, (list, tuple)) else infos
            if not isinstance(info, dict):
                continue
            action = np.asarray(actions[i], dtype=np.float32).flatten()
            state_obs = obs["state"][i] if isinstance(obs, dict) else obs[i]
            pos = np.asarray(state_obs[0:3], dtype=np.float32)
            if not hasattr(ep, "_last_pos"):
                ep._last_pos = pos.copy()  # type: ignore[attr-defined]
            update_episode_from_info(
                ep,
                reward=float(rewards[i]),
                info=info,
                action=action,
                prev_action=prev_actions[i],
                pos=pos,
            )
            prev_actions[i] = action.copy()
            if dones[i]:
                episodes.append(ep)
                ep_states[i] = EpisodeMetrics()
                ep_states[i]._last_pos = pos.copy()  # type: ignore[attr-defined]
                prev_actions[i] = None
                completed += 1
                if completed >= n_episodes:
                    break
    return aggregate_episodes(episodes[:n_episodes])
