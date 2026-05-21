#!/usr/bin/env python3
"""
Train a PPO model for drone navigation.

Workflow:
1. Train model → saved to swarm/submission_template/ppo_policy.zip
2. Test with: python tests/test_rpc.py swarm/submission_template/ --zip
3. Submission.zip created in Submission/
4. Run miner (reads from Submission/submission.zip)
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT

NUMBER_OF_ENV = 8
def main():
    parser = argparse.ArgumentParser(description="Train PPO model for Swarm subnet")
    parser.add_argument("--timesteps", type=int, default=1e7, help="Training timesteps")
    args = parser.parse_args()
    # seeds = [2,3,6,8,9,10,11.13]

    def make_training_env(seed):
        def _init():
            task = random_task(sim_dt=SIM_DT, seed=seed)
            return make_env(task, gui=False)
        return _init

    env = SubprocVecEnv([make_training_env(2+s) for s in range(NUMBER_OF_ENV)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        device="cuda",
        tensorboard_log="./ppo_logs/",
        learning_rate=lambda f: 1e-3 * f,
        n_steps=2048,
        batch_size=1024,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.8,
        ent_coef=0.0005,
        max_grad_norm=0.5,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // NUMBER_OF_ENV,  # adjust for vector env
        save_path="./checkpoints/",
        name_prefix="ppo_drone"
    )

    try:
        model.learn(args.timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")

    output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ppo_policy_new1.zip"
    model.save(str(model_path))
    
    print(f"\nModel saved to: {str(model_path)}")
    print("\nNext steps:")
    print("   1. Test: python tests/test_rpc.py swarm/submission_template/ --zip")
    print("   2. Run miner (reads from Submission/submission.zip)")

    env.close()


if __name__ == "__main__":
    main()
