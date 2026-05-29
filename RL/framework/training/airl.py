"""AIRL training pipeline using project SwarmAIRL + custom reward net."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from imitation.data import rollout

from RL.framework.data.expert_data import load_trajectories, save_trajectories
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim import Adam

from RL.framework.config.schemas import TrainingConfig
from RL.framework.policies.builder import build_policy_kwargs
from RL.framework.utils.paths import ensure_import_paths
from RL.framework.utils.seeding import set_global_seed
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


def _patch_dict_replay_buffer() -> None:
    ensure_import_paths()
    from dict_observation_replay_buffer import patch_imitation_replay_buffer_for_dict_obs

    patch_imitation_replay_buffer_for_dict_obs()


def make_airl_vec_env(n_envs: int, seed: int, gui: bool = False) -> DummyVecEnv:
    def thunk(i: int):
        def _init():
            task = random_task(sim_dt=SIM_DT, seed=seed + i * 7919)
            env = make_env(task, gui=gui)
            return RolloutInfoWrapper(env)

        return _init

    return DummyVecEnv([thunk(i) for i in range(n_envs)])


def load_demonstrations(
    config: TrainingConfig,
    venv: DummyVecEnv,
    device: str,
) -> list:
    airl = config.airl
    expert_pickle = Path(airl.expert_path)
    if expert_pickle.is_file():
        print(f"Loading demonstrations from {expert_pickle}")
        return load_trajectories(expert_pickle)

    pretrain = Path(airl.pretrain_path)
    if not pretrain.is_file():
        raise FileNotFoundError(
            f"No expert data at {expert_pickle} and no BC checkpoint at {pretrain}. "
            "Run collect_expert_data.py or train_pretrain.py first."
        )
    print(f"Sampling expert rollouts from {pretrain}")
    expert = PPO.load(str(pretrain), device=device)
    rng = np.random.default_rng(config.seed)
    demos = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_episodes=airl.min_expert_episodes),
        rng=rng,
    )
    save_trajectories(expert_pickle, demos)
    return demos


def run_airl_training(config: TrainingConfig) -> None:
    _patch_dict_replay_buffer()
    ensure_import_paths()
    from customnetwork import CustomShapedRewardNet as rewardnetwork
    from swarm_airl import SwarmAIRL

    set_global_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(get_device())

    venv = make_airl_vec_env(1, config.seed, gui=config.vec_env.gui)
    demonstrations = load_demonstrations(config, venv, device)
    airl_cfg = config.airl

    policy_kwargs = build_policy_kwargs(config.policy)
    pretrain = Path(airl_cfg.pretrain_path)
    if pretrain.is_file():
        learner = PPO.load(str(pretrain), env=venv, device=device, load_optimizer=False)
        learner.policy.optimizer = Adam(learner.policy.parameters(), lr=airl_cfg.gen_lr)
        if airl_cfg.log_std_boost != 0.0:
            learner.policy.log_std.data += airl_cfg.log_std_boost
    else:
        learner = PPO(
            ActorCriticPolicy,
            venv,
            policy_kwargs=policy_kwargs,
            learning_rate=airl_cfg.gen_lr,
            gae_lambda=0.95,
            ent_coef=0.02,
            n_steps=2048,
            batch_size=512,
            gamma=0.99,
            device=device,
            verbose=1,
        )

    from customnetwork import feature_extractor as fnetwork

    reward_net = rewardnetwork(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize=RunningNorm,
        feature_extractor=fnetwork,
        use_state=True,
        use_action=True,
        use_next_state=True,
        use_done=True,
        discount_factor=0.99,
        feature_output_dim=config.policy.features_dim,
        if_share_feature_extractor=False,
    )

    Path(airl_cfg.reward_dir).mkdir(parents=True, exist_ok=True)
    Path(airl_cfg.policy_dir).mkdir(parents=True, exist_ok=True)

    trainer = SwarmAIRL(
        demonstrations=demonstrations,
        demo_batch_size=airl_cfg.demo_batch_size,
        gen_replay_buffer_capacity=airl_cfg.gen_replay_buffer_capacity,
        gen_train_timesteps=airl_cfg.gen_train_timesteps,
        n_disc_updates_per_round=airl_cfg.n_disc_updates_per_round,
        disc_opt_kwargs={"lr": airl_cfg.disc_lr, "weight_decay": 1e-4},
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        init_tensorboard=True,
        log_dir=config.logging.tensorboard_dir,
    )
    trainer.allow_variable_horizon = True

    for n in range(1, airl_cfg.rounds + 1):
        print(f"AIRL round {n}/{airl_cfg.rounds}")
        trainer.train(airl_cfg.steps_per_round)
        torch.save(reward_net, os.path.join(airl_cfg.reward_dir, f"reward_net{n}.pth"))
        learner.save(os.path.join(airl_cfg.policy_dir, f"ppo_policy_{n}"))

    venv.close()
