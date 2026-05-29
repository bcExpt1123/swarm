"""Vectorized environment construction for training and evaluation."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, VecNormalize

from RL.framework.config.schemas import RewardWeights, TrainingConfig, VecEnvConfig
from RL.framework.curriculum.manager import CurriculumManager
from RL.framework.rewards.wrapper import ModularRewardWrapper
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task, task_for_seed_and_type


def _make_single_env(
    *,
    rank: int,
    gui: bool,
    curriculum: Optional[CurriculumManager],
    reward_weights: Optional[RewardWeights],
    global_step: int = 0,
    base_seed: int = 0,
) -> Callable[[], gym.Env]:
    def _init():
        if curriculum is not None and curriculum.config.enabled:
            seed = curriculum.task_seed(rank, global_step)
            rng = random.Random(seed)
            ctype = curriculum.sample_challenge_type(rng)
            moving = rng.random() < curriculum.stage.moving_platform_prob
            task = task_for_seed_and_type(
                SIM_DT, seed=seed, challenge_type=ctype, moving_platform=moving
            )
        else:
            task = random_task(sim_dt=SIM_DT, seed=base_seed + rank * 7919)
        env = make_env(task, gui=gui)
        if reward_weights is not None:
            env = ModularRewardWrapper(env, reward_weights)
        return env

    return _init


def make_vec_env(
    config: TrainingConfig,
    *,
    curriculum: Optional[CurriculumManager] = None,
    norm_path: Optional[Path] = None,
    training: bool = True,
) -> VecEnv:
    """Build VecNormalize-wrapped parallel envs."""
    vec_cfg = config.vec_env
    reward_weights = config.reward if training else None

    if vec_cfg.gui and vec_cfg.n_envs != 1:
        raise ValueError("--gui requires n_envs=1")

    thunks = [
        _make_single_env(
            rank=i,
            gui=vec_cfg.gui,
            curriculum=curriculum,
            reward_weights=reward_weights,
            base_seed=config.seed,
        )
        for i in range(vec_cfg.n_envs)
    ]

    use_subproc = vec_cfg.subproc
    # PyBullet + SubprocVecEnv often hangs on Windows (spawn + GUI context).
    if sys.platform == "win32" and use_subproc:
        print(
            "Note: using DummyVecEnv on Windows (set vec_env.subproc: true to override).",
            flush=True,
        )
        use_subproc = False

    if vec_cfg.n_envs <= 1 or not use_subproc:
        venv: VecEnv = DummyVecEnv(thunks)
    else:
        print(f"Spawning {vec_cfg.n_envs} SubprocVecEnv workers…", flush=True)
        venv = SubprocVecEnv(thunks, start_method="spawn")

    venv = VecNormalize(
        venv,
        norm_obs=vec_cfg.norm_obs,
        norm_reward=vec_cfg.norm_reward,
        clip_obs=vec_cfg.clip_obs,
        training=training,
    )

    ckpt_dir = Path(config.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    venv = VecMonitor(venv, filename=str(ckpt_dir / "monitor.csv"))

    if norm_path is not None and norm_path.is_file():
        venv = VecNormalize.load(str(norm_path), venv)

    return venv
