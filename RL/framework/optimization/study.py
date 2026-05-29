"""Optuna hyperparameter optimization for Swarm PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from RL.framework.config.loader import load_config
from RL.framework.config.schemas import TrainingConfig
from RL.framework.evaluation.evaluator import evaluate_model
from RL.framework.policies.builder import create_ppo
from RL.framework.env.factory import make_vec_env
from RL.framework.utils.seeding import set_global_seed


def suggest_ppo_params(trial: optuna.Trial, base: TrainingConfig) -> TrainingConfig:
    cfg = base
    cfg.ppo.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    cfg.ppo.gamma = trial.suggest_float("gamma", 0.95, 0.999)
    cfg.ppo.gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    cfg.ppo.clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
    cfg.ppo.ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.02, log=True)
    cfg.ppo.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    cfg.ppo.n_epochs = trial.suggest_int("n_epochs", 3, 15)
    cfg.ppo.n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if arch == "small":
        cfg.policy.pi_layers = [64, 32]
        cfg.policy.vf_layers = [64, 32]
    elif arch == "medium":
        cfg.policy.pi_layers = [128, 64, 32]
        cfg.policy.vf_layers = [128, 64, 32]
    else:
        cfg.policy.pi_layers = [256, 128, 64, 32]
        cfg.policy.vf_layers = [256, 128, 64, 32]
    cfg.policy.features_dim = trial.suggest_categorical("features_dim", [64, 96, 128])
    return cfg


def run_optuna_study(config_path: str | Path, n_trials: int | None = None) -> optuna.Study:
    base = load_config(config_path)
    n_trials = n_trials or base.optuna.n_trials

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_ppo_params(trial, base)
        cfg.total_timesteps = base.optuna.n_timesteps_per_trial
        cfg.seed = base.seed + trial.number
        set_global_seed(cfg.seed)
        venv = make_vec_env(cfg, training=True)
        model = create_ppo(cfg, venv, device=cfg.device)
        model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
        summary = evaluate_model(model, cfg, n_episodes=base.optuna.n_eval_episodes)
        venv.close()
        trial.set_user_attr("success_rate", summary.success_rate)
        trial.set_user_attr("collision_rate", summary.collision_rate)
        return summary.composite_objective(base.evaluation.metrics_weights)

    sampler = TPESampler(seed=base.optuna.sampler_seed)
    study = optuna.create_study(
        study_name=base.optuna.study_name,
        storage=base.optuna.storage,
        direction="maximize",
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
