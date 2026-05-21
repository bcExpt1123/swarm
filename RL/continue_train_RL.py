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
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT

import torch
torch.set_num_threads(4)


def main(i):
    parser = argparse.ArgumentParser(description="Train PPO model for Swarm subnet")
    parser.add_argument("--timesteps", type=int, default=5e5, help="Training timesteps")
    args = parser.parse_args()

    def make_training_env(seed):
        def _init():
            task = random_task(sim_dt=SIM_DT, seed=seed)
            return make_env(task, gui=False)
        return _init

    # env = DummyVecEnv([make_training_env(5*i+1+s) for s in range(5)]) # 15
    # env = SubprocVecEnv([make_training_env(10*i+1+s) for s in range(10)])
    env = DummyVecEnv([make_training_env(5)]) # 15
    # "results/save-04.29.2026_11.35.20/final_model.zip",
    model = PPO.load(
        "swarm/submission_template/ppo_policy_init.zip",
        env=env,
        device="cuda",
        n_steps=2048,
        batch_size=512,
        verbose=1
    )
    model.learn(args.timesteps)

    folder = datetime.now().strftime("save-%m.%d.%Y_%H.%M.%S")
    output_dir = Path(__file__).parent.parent / "results" / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final_model.zip"
    model.save(str(model_path))
    
    print(f"\nModel saved to: {model_path}")
    print("\nNext steps:")
    print("   1. Test: python tests/test_rpc.py swarm/submission_template/ --zip")
    print("   2. Run miner (reads from Submission/submission.zip)")

    env.close()

# continue training
if __name__ == "__main__":

    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for i in range(10):
        try:
            print(f"\nStarting training iteration {i+1}...")
            main(i+30)
        except Exception as e:
            print(f"Error occurred while training model for iteration {i+1}: {e}")
    print(f"Finished training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # main(3)
