import argparse
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT


def main(i):
    timesteps = int(1e6)
    output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"ppo_policy_v1_{i}.zip"
    if model_path.exists():
        print(f"Warning: {model_path} already exists and will be overwritten.")
        return
    def make_training_env(seed):
        def _init():
            task = random_task(sim_dt=SIM_DT, seed=seed)
            return make_env(task, gui=False)
        return _init

    env = SubprocVecEnv([make_training_env(i*10+1+s) for s in range(10)])
    
    if i == 0:
        model = PPO(
            "MultiInputPolicy",
            env,
            device="cuda",
            tensorboard_log="./ppo_logs/",
            learning_rate=3e-4,
            max_grad_norm=0.5,
            n_steps=2048,
            batch_size=512,
            verbose=1
        )
    else:
        model = PPO.load(
            f"swarm/submission_template/ppo_policy_v1_{i-1}.zip",
            env=env,
            device="cuda",
            verbose=1
        )

    try:
        model.learn(timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")

    
    model.save(str(model_path))
    
    print(f"\nModel saved to: {str(model_path)}")
    env.close()


if __name__ == "__main__":
    for i in range(50):
        print(f"\nStarting training iteration {i+1} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main(i)
