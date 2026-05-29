"""Unified PPO training loop (pretrain / fine-tune)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_device

from RL.framework.callbacks import (
    CurriculumCallback,
    EpisodeStatsCallback,
    EvalCallback,
    WandbCallback,
    init_wandb,
)
from RL.framework.checkpoints.manager import CheckpointManager
from RL.framework.config.schemas import TrainingConfig
from RL.framework.curriculum.manager import CurriculumManager
from RL.framework.env.factory import make_vec_env
from RL.framework.policies.builder import create_ppo
from RL.framework.utils.seeding import set_global_seed


def _build_callbacks(
    config: TrainingConfig,
    model: Any,
    curriculum: CurriculumManager | None,
    ckpt_mgr: CheckpointManager,
    wandb_run: Any,
) -> list:
    callbacks = [
        EpisodeStatsCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(1, config.checkpoint.freq // config.vec_env.n_envs),
            save_path=str(ckpt_mgr.dir),
            name_prefix="ppo_swarm",
        ),
    ]
    eval_freq = max(1, config.evaluation.eval_freq // config.vec_env.n_envs)
    # Avoid running two full PyBullet eval sweeps on the same step.
    if curriculum is not None and config.curriculum.enabled:
        callbacks.append(CurriculumCallback(curriculum, config, eval_freq=eval_freq))
    elif config.evaluation.eval_freq > 0:
        callbacks.append(EvalCallback(config, eval_freq=eval_freq))
    if wandb_run is not None:
        callbacks.append(WandbCallback(wandb_run))
    return callbacks


def train_ppo(config: TrainingConfig, *, finetune: bool = False) -> PPO:
    """Run PPO training; ``finetune`` loads ``resume_path`` and uses gentler defaults if set."""
    set_global_seed(config.seed)
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(get_device(), flush=True)

    curriculum = CurriculumManager(config.curriculum, base_seed=config.seed) if config.curriculum.enabled else None
    norm_path = None
    if config.resume_path:
        rp = Path(config.resume_path)
        vn = rp.parent / "vecnormalize.pkl"
        if vn.is_file():
            norm_path = vn

    print(
        f"Building {config.vec_env.n_envs} env(s) "
        f"(subproc={config.vec_env.subproc}, first reset may take 1–2 min)…",
        flush=True,
    )
    venv = make_vec_env(config, curriculum=curriculum, norm_path=norm_path, training=True)
    print("Vec env ready.", flush=True)

    tb = None
    if config.logging.tensorboard_write_events:
        tb = config.logging.tensorboard_dir.strip() or None
        if tb:
            print(
                f"SB3 event logs → {tb}  (view later: tensorboard --logdir {tb})",
                flush=True,
            )
    else:
        print("TensorBoard event logging disabled for this run.", flush=True)
    wandb_run = init_wandb(config.logging, {"finetune": finetune, "seed": config.seed})

    if config.resume_path and Path(config.resume_path).is_file():
        print(f"Resuming from {config.resume_path}", flush=True)
        model = PPO.load(str(config.resume_path), env=venv, device=device)
        model.tensorboard_log = tb
        if finetune:
            model.ent_coef = config.ppo.ent_coef
            model.vf_coef = config.ppo.vf_coef
            model.max_grad_norm = config.ppo.max_grad_norm
            model.n_epochs = config.ppo.n_epochs
            model.clip_range = lambda _: config.ppo.clip_range
            if config.ppo.clip_range_vf is not None:
                model.clip_range_vf = config.ppo.clip_range_vf
            lr0 = float(config.ppo.learning_rate)
            if config.ppo.lr_end is not None:
                lr1 = float(config.ppo.lr_end)

                def _lr_schedule(progress_remaining: float) -> float:
                    return lr1 + (lr0 - lr1) * float(progress_remaining)

                model.learning_rate = _lr_schedule
            else:
                model.learning_rate = lr0
            print(
                f"Fine-tune overrides: lr={lr0}, vf_coef={config.ppo.vf_coef}, "
                f"clip={config.ppo.clip_range}, n_epochs={config.ppo.n_epochs}",
                flush=True,
            )
    else:
        print("Creating PPO model…", flush=True)
        model = create_ppo(config, venv, tensorboard_log=tb, device=device)

    ckpt_mgr = CheckpointManager(config.checkpoint.dir, config.checkpoint.keep_last_n)
    callbacks = _build_callbacks(config, model, curriculum, ckpt_mgr, wandb_run)

    rollout_steps = config.ppo.n_steps * config.vec_env.n_envs
    print(
        f"Starting PPO.learn for {config.total_timesteps:,} timesteps "
        f"(~{rollout_steps:,} env steps per rollout update)…",
        flush=True,
    )
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("Interrupted; saving checkpoint…", flush=True)
    except Exception as exc:
        print(f"Training error: {exc}", flush=True)
        print("Saving emergency checkpoint…", flush=True)
        ckpt_mgr.save_model(model, "ppo_emergency")
        raise

    ckpt_mgr.save_vecnormalize(venv)
    ckpt_mgr.export_submission(model, config.checkpoint.submission_output)
    ckpt_mgr.save_model(model, config.checkpoint.submission_output)
    venv.close()
    return model
