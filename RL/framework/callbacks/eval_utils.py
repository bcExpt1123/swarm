"""Shared helpers for eval callbacks during PPO training."""

from __future__ import annotations

from stable_baselines3.common.base_class import BaseAlgorithm

from RL.framework.config.schemas import TrainingConfig
from RL.framework.evaluation.evaluator import evaluate_model, policy_has_nan
from RL.framework.evaluation.metrics import EvalSummary


def run_training_eval(
    model: BaseAlgorithm,
    config: TrainingConfig,
    *,
    n_episodes: int,
    label: str = "eval",
) -> EvalSummary | None:
    """Run eval during training; never crash the training loop."""
    if policy_has_nan(model):
        print(
            f"[{label}] SKIP at {getattr(model, 'num_timesteps', '?')} steps: "
            "policy weights are NaN/Inf (diverged). Resume from last "
            "ppo_swarm_*_steps.zip with lower lr and vf_coef≈0.15.",
            flush=True,
        )
        return None
    cap = min(n_episodes, config.evaluation.n_episodes, 15)
    try:
        return evaluate_model(
            model,
            config,
            n_episodes=cap,
            norm_obs=config.vec_env.norm_obs,
        )
    except Exception as exc:
        print(f"[{label}] failed (training continues): {exc}", flush=True)
        return None
