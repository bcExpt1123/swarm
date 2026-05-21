#!/usr/bin/env python3
"""
PPO training on Swarm ``MovingDroneAviary`` (same stack as ``train_RL.py``).

Default policy is SB3 ``MultiInputPolicy`` + ``VecNormalize``. Use ``--policy-type custom``
for ``ActorCriticPolicy`` + ``customnetwork`` (same as BC / AIRL pretrain).

Parameters:
- Simple task:
```
python ppo_train.py \
  --timesteps 300000 \
  --n-envs 4 \
  --policy-type custom \
  --learning-rate 3e-4 \
  --ent-coef 0.001 \
  --gamma 0.95 \
  --n_steps 1024 \
  --batch_size 512 \
  --n-epochs 5 \
  --clip-range 0.2 \
  --vf_coef 0.5 \
  --max-grad-norm 0.5 \
  --resume model/bc_pretrain/pretrain_model.zip \
  --no-norm-obs --no-norm-reward
```
- Medium task:
```
python ppo_train.py \
  --timesteps 1500000 \
  --n-envs 6 \
  --policy-type custom \
  --learning-rate 3e-4 \
  --lr-end 5e-5 \
  --ent-coef 0.002 \
  --gamma 0.97 \
  --n_steps 2048 \
  --batch_size 1024 \
  --n-epochs 8 \
  --clip-range 0.15 \
  --vf_coef 0.6 \
  --max-grad-norm 0.5 \
  --checkpoint-freq 250000 \
  --resume model/bc_pretrain/pretrain_model.zip
```
- Hard task:
```
python ppo_train.py \
  --timesteps 5000000 \
  --n-envs 8 \
  --policy-type custom \
  --learning-rate 2e-4 \
  --lr-end 3e-5 \
  --ent-coef 0.003 \
  --gamma 0.99 \
  --n_steps 2048 \
  --batch_size 1024 \
  --n-epochs 10 \
  --clip-range 0.1 \
  --vf_coef 0.8 \
  --max-grad-norm 0.3 \
  --checkpoint-freq 500000 \
  --resume model/airl/policy/ppo_policy_49.zip
```
- Real hard task:
```
python ppo_train.py \
  --timesteps 15000000 \
  --n-envs 8 \
  --policy-type custom \
  --learning-rate 1e-4 \
  --lr-end 1e-5 \
  --ent-coef 0.005 \
  --gamma 0.995 \
  --n_steps 4096 \
  --batch_size 2048 \
  --n-epochs 5 \
  --clip-range 0.08 \
  --vf_coef 1.0 \
  --max-grad-norm 0.2 \
  --checkpoint-freq 1000000 \
  --resume model/airl/policy/ppo_policy_49.zip
```

Examples::

    python RL/ppo_train.py --timesteps 1000000 --n-envs 8
    python RL/ppo_train.py --policy-type custom --resume ./model/bc_pretrain/pretrain_model.zip
    # Fine-tune after AIRL with gentler updates (often helps late-run oscillation):
    python RL/ppo_train.py --policy-type custom --resume model/airl/policy/ppo_policy_49.zip \\
      --no-norm-obs --no-norm-reward --learning-rate 3e-4 --lr-end 1e-5 \\
      --ent-coef 0.003 --n-epochs 5 --clip-range 0.1 --max-grad-norm 0.3
    python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain/pretrain_model.zip --timesteps 5000000 --no-norm-obs --no-norm-reward
    python RL/ppo_train.py --policy-type custom --resume checkpoints/ppo_swarm/ppo_swarm_move_500000_steps.zip --no-norm-obs --no-norm-reward --timesteps 5000000 --learning-rate 3e-4 --lr-end 1e-5 --ent-coef 0.003 --n-epochs 5 --clip-range 0.1 --max-grad-norm 0.3
    
    python RL/ppo_train.py --policy-type custom --resume model/airl/policy_48/ppo_policy_49.zip --no-norm-obs --no-norm-reward --timesteps 2500000 --n_steps 512 --n-envs 8 --learning-rate 3e-5 --lr-end 1e-5 --seed 2 --checkpoint-freq 500000 --tensorboard ./ppo_logs/
    python RL/ppo_train.py --policy-type custom --resume model/airl/policy_48/ppo_policy_49.zip --no-norm-obs --no-norm-reward --timesteps 2500000 --n_steps 512 --n-envs 8 --learning-rate 3e-5 --lr-end 1e-5 --seed 2 --checkpoint-freq 500000 --tensorboard ./ppo_logs/ --checkpoint-dir ./checkpoints/ppo_swarm_1 --output-file ppo_policy_1
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import VecNormalize

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RL_DIR = Path(__file__).resolve().parent
for _p in (_RL_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


class EpisodeRewardCallback(BaseCallback):
    """Rolling mean episode return / length from Monitor info."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if info and "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
                self.episode_count += 1
                mean_r = float(np.mean(self.episode_rewards[-100:]))
                mean_l = float(np.mean(self.episode_lengths[-100:]))
                if self.verbose > 0:
                    print(
                        f"Episode {self.episode_count}: "
                        f"mean_reward(100)={mean_r:.3f}, mean_len(100)={mean_l:.1f}"
                    )
                if self.logger is not None:
                    self.logger.record("episode/mean_reward", mean_r)
                    self.logger.record("episode/mean_length", mean_l)
                    self.logger.record("episode/count", float(self.episode_count))
                    self.logger.dump(self.episode_count)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO on Swarm MovingDroneAviary")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total env steps.")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments (1 = DummyVecEnv).")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI (forces n-envs=1).")
    parser.add_argument(
        "--policy-type",
        choices=("multiinput", "custom"),
        default="multiinput",
        help="multiinput: SB3 default CNN+MLP for Dict obs; custom: RL/customnetwork.py extractor.",
    )
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0005)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="If set, linearly decay learning_rate to this value over training "
        "(SB3: lr = lr_end + (learning_rate - lr_end) * progress_remaining). "
        "Use with a lower --learning-rate when resuming to reduce late instability.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.8)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--tensorboard", type=str, default="./ppo_logs/", help="Log dir or empty to disable.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ppo_swarm/")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000, help="Save every N steps (global).")
    parser.add_argument("--resume", type=str, default="", help="Optional path to .zip to continue training.")
    parser.add_argument(
        "--norm-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VecNormalize observation clipping/normalization.",
    )
    parser.add_argument(
        "--norm-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--output-file", type=str, default="ppo_policy")
    args = parser.parse_args()

    if args.gui and args.n_envs != 1:
        raise SystemExit("--gui requires --n-envs 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(get_device())

    base_seed = args.seed

    def make_training_env(rank: int):
        def _init():
            task = random_task(sim_dt=SIM_DT, seed=base_seed + rank)
            return make_env(task, gui=args.gui)

        return _init

    if args.n_envs <= 1:
        venv = DummyVecEnv([make_training_env(0)])
    else:
        venv = SubprocVecEnv([make_training_env(i) for i in range(args.n_envs)])

    venv = VecNormalize(
        venv,
        norm_obs=args.norm_obs,
        norm_reward=args.norm_reward,
        clip_obs=args.clip_obs,
    )
    venv = VecMonitor(venv, filename=str(Path(args.checkpoint_dir) / "monitor.csv"))

    tb_log = args.tensorboard.strip() or None

    callbacks: list[BaseCallback] = [
        EpisodeRewardCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=args.checkpoint_dir,
            name_prefix="ppo_swarm",
        ),
    ]

    # if args.resume:
    #     resume_path = Path(args.resume)
    #     if not resume_path.is_file():
    #         raise FileNotFoundError(resume_path)
    #     print(f"Loading {resume_path}")
    #     model = PPO.load(str(resume_path), env=venv, device=device_str)
    #     model.tensorboard_log = tb_log
    if args.policy_type == "multiinput":
        model = PPO(
            "MultiInputPolicy",
            venv,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.n_epochs,
            seed=args.seed,
            device=device_str,
            verbose=1,
            tensorboard_log=tb_log,
        )
    else:
        from customnetwork import feature_extractor as fnetwork

        policy_kwargs = dict(
            features_extractor_class=fnetwork,
            features_extractor_kwargs=dict(features_dim=96),
            net_arch=dict(pi=[128, 64, 32, 16], vf=[128, 64, 32, 16]),
            share_features_extractor=False,
            log_std_init=-1.8,
        )
        model = PPO(
            ActorCriticPolicy,
            venv,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=0.1,
            normalize_advantage=True,
            vf_coef=0.15,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.n_epochs,
            seed=args.seed,
            device=device_str,
            verbose=1,
            tensorboard_log=tb_log,
        )

    model.set_random_seed(args.seed)
    print(f"LR_END => {str(args.lr_end)}")
    if args.lr_end is not None:
        lr0 = float(args.learning_rate)
        lr1 = float(args.lr_end)

        def _lr_schedule(progress_remaining: float) -> float:
            print(f"Progress remaining => {str(progress_remaining)}")
            return lr1 + (lr0 - lr1) * float(progress_remaining)

        model.learning_rate = _lr_schedule
        print(f"Using linear LR schedule: {lr0} -> {lr1} over run.")

    # ================================================
    resume_path = Path(args.resume)
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    print(f"Loading {resume_path}")
    model.tensorboard_log = tb_log
    model.set_parameters(resume_path)
    model.n_epochs = args.n_epochs
    model.clip_range = lambda _: args.clip_range
    model.vf_coef = args.vf_coef
    model.ent_coef = args.ent_coef
    # ================================================

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\nInterrupted; saving current model…")

    out_dir = _REPO_ROOT / "swarm" / "submission_template"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.output_file
    model.save(str(stem))
    print(f"Saved policy to {stem}.zip")

    venv.close()


if __name__ == "__main__":
    main()
