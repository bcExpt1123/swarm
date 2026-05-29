"""Configurable PPO policy architectures (MLP, CNN/custom, recurrent)."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv

from RL.framework.config.schemas import PolicyConfig, PPOConfig, TrainingConfig


def build_policy_kwargs(policy: PolicyConfig) -> dict[str, Any]:
    if policy.policy_type in ("custom", "cnn"):
        import sys
        from pathlib import Path

        rl_dir = Path(__file__).resolve().parents[2]
        if str(rl_dir) not in sys.path:
            sys.path.insert(0, str(rl_dir))
        from customnetwork import feature_extractor as fnetwork

        return dict(
            features_extractor_class=fnetwork,
            features_extractor_kwargs=dict(features_dim=policy.features_dim),
            net_arch=dict(pi=policy.pi_layers, vf=policy.vf_layers),
            share_features_extractor=policy.share_features_extractor,
            log_std_init=policy.log_std_init,
        )

    if policy.policy_type == "recurrent":
        return dict(
            net_arch=dict(pi=policy.pi_layers, vf=policy.vf_layers),
            lstm_hidden_size=policy.lstm_hidden_size,
            n_lstm_layers=policy.n_lstm_layers,
            shared_lstm=False,
            enable_critic_lstm=False,
        )

    # mlp / multiinput default arch
    return dict(
        net_arch=dict(pi=policy.pi_layers, vf=policy.vf_layers),
        log_std_init=policy.log_std_init,
    )


def resolve_policy_class(policy_type: str):
    if policy_type == "recurrent":
        try:
            from sb3_contrib import RecurrentPPO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Recurrent PPO requires sb3-contrib: pip install sb3-contrib"
            ) from exc
        return RecurrentPPO, "MlpLstmPolicy"
    if policy_type == "multiinput":
        return PPO, "MultiInputPolicy"
    if policy_type in ("custom", "cnn"):
        return PPO, ActorCriticPolicy
    return PPO, "MlpPolicy"


def create_ppo(
    config: TrainingConfig,
    venv: VecEnv,
    *,
    tensorboard_log: str | None = None,
    device: str = "auto",
) -> Any:
    """Instantiate PPO (or RecurrentPPO) from TrainingConfig."""
    ppo_cfg = config.ppo
    algo_cls, policy_cls = resolve_policy_class(config.policy.policy_type)

    kwargs: dict[str, Any] = dict(
        policy=policy_cls,
        env=venv,
        learning_rate=ppo_cfg.learning_rate,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        vf_coef=ppo_cfg.vf_coef,
        ent_coef=ppo_cfg.ent_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        n_epochs=ppo_cfg.n_epochs,
        normalize_advantage=ppo_cfg.normalize_advantage,
        seed=config.seed,
        device=device,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    if config.policy.policy_type in ("custom", "cnn", "mlp", "recurrent"):
        kwargs["policy_kwargs"] = build_policy_kwargs(config.policy)

    if ppo_cfg.clip_range_vf is not None and config.policy.policy_type == "custom":
        kwargs["clip_range_vf"] = ppo_cfg.clip_range_vf
        kwargs["vf_coef"] = min(ppo_cfg.vf_coef, 0.25)

    model = algo_cls(**kwargs)

    if ppo_cfg.lr_end is not None:
        lr0, lr1 = float(ppo_cfg.learning_rate), float(ppo_cfg.lr_end)

        def _schedule(progress_remaining: float) -> float:
            return lr1 + (lr0 - lr1) * float(progress_remaining)

        model.learning_rate = _schedule

    return model
