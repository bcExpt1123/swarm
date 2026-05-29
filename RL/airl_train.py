"""
Parameters:
- Simple task:
```
--n_envs 1 \
--min_expert_episodes 10 \
--rounds 20 \
--steps_per_round 8000 \
--demo_batch_size 256 \
--gen_replay_buffer_capacity 1024 \
--gen_train_timesteps 300 \
--n_disc_updates_per_round 2 \
--disc_lr 1e-4 \
--gen_lr 1e-4
```
- Medium task:
```
--n_envs 1 \
--min_expert_episodes 24 \
--rounds 35 \
--steps_per_round 12000 \
--demo_batch_size 512 \
--gen_replay_buffer_capacity 2048 \
--gen_train_timesteps 500 \
--n_disc_updates_per_round 3 \
--disc_lr 5e-5 \
--gen_lr 5e-5
```
- Hard task:
```
--n_envs 1 \
--min_expert_episodes 50 \
--rounds 50 \
--steps_per_round 15000 \
--demo_batch_size 512 \
--gen_replay_buffer_capacity 4096 \
--gen_train_timesteps 800 \
--n_disc_updates_per_round 4 \
--disc_lr 3e-5 \
--gen_lr 3e-5 \
--log_std_boost 0.0
```
- Real Hard task:
```
--n_envs 1 \
--min_expert_episodes 100 \
--rounds 60 \
--steps_per_round 20000 \
--demo_batch_size 1024 \
--gen_replay_buffer_capacity 8192 \
--gen_train_timesteps 1000 \
--n_disc_updates_per_round 5 \
--disc_lr 1e-5 \
--gen_lr 1e-5 \
--log_std_boost 0.0
```
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from imitation.data import rollout

from RL.framework.data.expert_data import load_trajectories, save_trajectories
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.networks import RunningNorm
from torch.optim import Adam

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RL_DIR = Path(__file__).resolve().parent
for _p in (_RL_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dict_observation_replay_buffer import patch_imitation_replay_buffer_for_dict_obs

patch_imitation_replay_buffer_for_dict_obs()

from swarm_airl import SwarmAIRL

from customnetwork import CustomShapedRewardNet as rewardnetwork
from customnetwork import feature_extractor as fnetwork

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


def _make_vec_env(
    *,
    n_envs: int,
    seed: int,
    gui: bool,
) -> DummyVecEnv:
    """One Subproc-free vec env; each worker gets a fresh ``random_task`` map."""

    def make_thunk(env_index: int):
        def _init():
            base = seed + env_index * 7919
            task = random_task(sim_dt=SIM_DT, seed=base)
            env = make_env(task, gui=gui)
            return RolloutInfoWrapper(env)

        return _init

    return DummyVecEnv([make_thunk(i) for i in range(n_envs)])


def load_or_sample_demonstrations(
    *,
    expert_pickle: Path,
    pretrain_path: Path,
    venv: DummyVecEnv,
    min_episodes: int,
    rng: np.random.Generator,
    save_demo_path: Path | None,
    device: str,
):
    if expert_pickle.is_file():
        print(f"Loading cached demonstrations from {expert_pickle}")
        return load_trajectories(expert_pickle)

    if not pretrain_path.is_file():
        raise FileNotFoundError(
            f"No demonstrations file at {expert_pickle} and no BC checkpoint at {pretrain_path}. "
            "Run `python RL/pretrain_ppo.py` first, or pass `--expert_path` to saved rollouts."
        )

    print(f"Sampling expert trajectories with BC policy {pretrain_path} …")
    expert = PPO.load(str(pretrain_path), device=device)
    demos = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_episodes=min_episodes),
        rng=rng,
    )
    if save_demo_path is not None:
        save_demo_path.parent.mkdir(parents=True, exist_ok=True)
        save_trajectories(save_demo_path, demos)
        print(f"Wrote demonstrations to {save_demo_path}")
    return demos


def main():
    parser = argparse.ArgumentParser(description="AIRL on Swarm MovingDroneAviary")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n_envs", type=int, default=1, help="Vec env size (1 recommended for PyBullet).")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI (single env only).")
    parser.add_argument(
        "--expert_path",
        type=str,
        default="expert/expert_data_all.pkl",
        help="Cached imitation trajectories; if missing, sampled from --pretrain_path.",
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        default="model/bc_pretrain/pretrain_model.zip",
        help="BC PPO zip used to sample expert rollouts when expert_path is absent.",
    )
    parser.add_argument(
        "--save_expert_path",
        type=str,
        default="",
        help="If set, save freshly sampled rollouts to this .pkl for reuse.",
    )
    parser.add_argument("--min_expert_episodes", type=int, default=24)
    parser.add_argument(
        "--pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true and pretrain_path exists, load PPO for the generator (vs fresh init).",
    )
    parser.add_argument(
        "--gen_lr",
        type=float,
        default=5e-5,
        help="Generator Adam lr after loading BC weights.",
    )
    parser.add_argument(
        "--log_std_boost",
        type=float,
        default=0.0,
        help="Add to policy log_std after load (old AirSim script used +6; 0 is safer for Swarm).",
    )
    parser.add_argument("--rounds", type=int, default=49, help="Outer AIRL training rounds (default ~ original 1..49).")
    parser.add_argument("--steps_per_round", type=int, default=15000)
    parser.add_argument("--demo_batch_size", type=int, default=512)
    parser.add_argument("--gen_replay_buffer_capacity", type=int, default=2048)
    parser.add_argument("--gen_train_timesteps", type=int, default=500)
    parser.add_argument("--n_disc_updates_per_round", type=int, default=4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--tensorboard_log", type=str, default="./ppo_logs/")
    parser.add_argument("--reward_dir", type=str, default="model/airl/reward")
    parser.add_argument("--policy_dir", type=str, default="model/airl/policy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(get_device(), flush=True)

    if args.gui and args.n_envs != 1:
        raise SystemExit("--gui requires --n_envs 1")

    venv = _make_vec_env(n_envs=args.n_envs, seed=args.seed, gui=args.gui)

    expert_pickle = Path(args.expert_path)
    pretrain_path = Path(args.pretrain_path)
    save_demo = Path(args.save_expert_path) if args.save_expert_path else None

    demonstrations = load_or_sample_demonstrations(
        expert_pickle=expert_pickle,
        pretrain_path=pretrain_path,
        venv=venv,
        min_episodes=args.min_expert_episodes,
        rng=rng,
        save_demo_path=save_demo,
        device=device_str,
    )

    policy_kwargs = dict(
        features_extractor_class=fnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], vf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-2.7,
    )

    if args.pretrain and pretrain_path.is_file():
        print(f"Loading generator from BC checkpoint {pretrain_path}")
        learner = PPO.load(
            str(pretrain_path),
            env=venv,
            device=device_str,
            load_optimizer=False,
            print_system_info=False,
        )
        learner.policy.optimizer = Adam(learner.policy.parameters(), lr=args.gen_lr)
        if args.log_std_boost != 0.0:
            learner.policy.log_std.data += args.log_std_boost
            print(f"log_std after boost: {learner.policy.log_std.data}")
    else:
        learner = PPO(
            env=venv,
            policy=ActorCriticPolicy,
            policy_kwargs=policy_kwargs,
            gae_lambda=0.95,
            ent_coef=0.02,
            learning_rate=args.gen_lr,
            n_steps=2048,
            batch_size=512,
            gamma=0.9,
            clip_range=0.2,
            clip_range_vf=0.1,
            normalize_advantage=True,
            vf_coef=0.15,
            max_grad_norm=2.0,
            n_epochs=10,
            seed=args.seed,
            device=device_str,
            verbose=1,
        )

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
        feature_output_dim=96,
        if_share_feature_extractor=False,
    )

    Path(args.reward_dir).mkdir(parents=True, exist_ok=True)
    Path(args.policy_dir).mkdir(parents=True, exist_ok=True)

    airl_trainer = SwarmAIRL(
        demonstrations=demonstrations,
        demo_batch_size=args.demo_batch_size,
        gen_replay_buffer_capacity=args.gen_replay_buffer_capacity,
        gen_train_timesteps=args.gen_train_timesteps,
        n_disc_updates_per_round=args.n_disc_updates_per_round,
        disc_opt_kwargs={"lr": args.disc_lr, "weight_decay": 1e-4},
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        init_tensorboard=True,
        log_dir=args.tensorboard_log,
    )
    airl_trainer.allow_variable_horizon = True

    for n in range(1, args.rounds + 1):
        print(f"AIRL round {n}/{args.rounds}", flush=True)
        airl_trainer.train(args.steps_per_round)
        torch.save(reward_net, os.path.join(args.reward_dir, f"reward_net{n}.pth"))
        learner.save(os.path.join(args.policy_dir, f"ppo_policy_{n}"))

    venv.close()


if __name__ == "__main__":
    main()
