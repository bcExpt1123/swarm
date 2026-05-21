"""
BC pretrain for MovingDroneAviary: collect random-action rollouts via
``make_env`` + ``random_task``, then fit ``ActorCriticPolicy`` + ``customnetwork``
feature extractor with negative log-likelihood (see ``swarm/utils/env_factory.py``).

Example::
- Simple task(hover / basic stability):
```
python RL/pretrain_ppo.py \
  --seed 2 \
  --collect_episodes 40 \
  --max_steps_per_episode 2048 \
  --train_step 80 \
  --bc_lr 2e-4 \
  --batch_size 512 \
  --learning_rate 3e-4 \
  --n_steps 2048 \
  --ent_coef 0.01 \
  --max_grad_norm 0.3 \
  --output ./model/bc_pretrain_simple/simple_drone.zip
```
- Medium task(waypoint / mild dynamics):
```
python RL/pretrain_ppo.py \
  --seed 2 \
  --collect_episodes 100 \
  --max_steps_per_episode 3072 \
  --train_step 150 \
  --bc_lr 1e-4 \
  --batch_size 1024 \
  --learning_rate 3e-4 \
  --n_steps 4096 \
  --ent_coef 0.005 \
  --max_grad_norm 0.3 \
  --output ./model/bc_pretrain/medium_drone.zip
```

- Hard task(full 3D control / disturbances):
```
python RL/pretrain_ppo.py \
  --seed 2 \
  --collect_episodes 150 \
  --max_steps_per_episode 4096 \
  --train_step 200 \
  --bc_lr 1e-4 \
  --batch_size 1024 \
  --n_steps 4096 \
  --learning_rate 3e-4 \
  --n_epochs 10 \
  --ent_coef 0.005 \
  --max_grad_norm 0.3 \
  --clip_range 0.2 \
  --clip_range_vf 0.1 \
  --output ./model/bc_pretrain/hard_drone_pretrain.zip
```
- Real hard task(recommended):
```
python RL/pretrain_ppo.py \
  --seed 42 \
  --collect_episodes 250 \
  --max_steps_per_episode 4096 \
  --train_step 300 \
  --bc_lr 1e-4 \
  --batch_size 1024 \
  --learning_rate 3e-4 \
  --n_steps 4096 \
  --ent_coef 0.01 \
  --max_grad_norm 0.3 \
  --output ./model/bc_pretrain/hard_drone_pretrain_v2.zip
```
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device, set_random_seed
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from customnetwork import feature_extractor as fnetwork

from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


class BehavioralCloningDataset(Dataset):
    def __init__(self, observations: list, actions: np.ndarray):
        self.observations = observations
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        return self.observations[idx], self.actions[idx]


def collate_bc(batch):
    obs_list = [b[0] for b in batch]
    acts = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
    depth = np.stack([o["depth"] for o in obs_list], axis=0)
    state = np.stack([o["state"] for o in obs_list], axis=0)
    obs_batch = {
        "depth": torch.from_numpy(depth),
        "state": torch.from_numpy(state),
    }
    return obs_batch, torch.from_numpy(acts)


def collect_random_demonstrations(
    *,
    collect_episodes: int,
    seed: int,
    gui: bool,
    max_steps_per_episode: int,
) -> tuple[list, np.ndarray]:
    """Roll out ``env.action_space.sample()`` trajectories for BC targets."""
    rng = np.random.default_rng(seed)
    obs_all: list = []
    acts_all: list = []

    for _ in range(collect_episodes):
        task_seed = int(rng.integers(0, 2**31))
        task = random_task(sim_dt=SIM_DT, seed=task_seed)
        env = make_env(task, gui=gui)
        try:
            obs, _ = env.reset(seed=task.map_seed)
            steps = 0
            while steps < max_steps_per_episode:
                action = env.action_space.sample().astype(np.float32)
                obs_all.append(obs)
                acts_all.append(action)
                obs, _reward, terminated, truncated, _info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
        finally:
            env.close()

    if not acts_all:
        raise RuntimeError(
            "No transitions collected; increase collect_episodes or max_steps_per_episode."
        )
    return obs_all, np.stack(acts_all, axis=0)


def main():
    parser = argparse.ArgumentParser(description="BC pretrain PPO policy on MovingDroneAviary")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--clip_range_vf", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=0.15)
    parser.add_argument(
        "--train_step",
        type=int,
        default=50,
        help="Number of passes over the BC dataset (epochs).",
    )
    parser.add_argument("--testing", action="store_true", help="Run short eval rollouts after BC.")
    parser.add_argument("--testing_episode", type=int, default=10)
    parser.add_argument("--collect_episodes", type=int, default=48)
    parser.add_argument("--max_steps_per_episode", type=int, default=4096)
    parser.add_argument("--bc_lr", type=float, default=5e-4)
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI for collection / eval env.")
    parser.add_argument(
        "--output",
        type=str,
        default="./model/bc_pretrain/pretrain_model.zip",
        help="Path for saved PPO checkpoint (.zip).",
    )
    args = parser.parse_args()

    random_seed = args.seed
    random.seed(random_seed)
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    print("Collecting random demonstrations...")
    observations, actions_np = collect_random_demonstrations(
        collect_episodes=args.collect_episodes,
        seed=random_seed,
        gui=args.gui,
        max_steps_per_episode=args.max_steps_per_episode,
    )
    print(f"Collected {len(actions_np)} transitions.")

    task = random_task(sim_dt=SIM_DT, seed=random_seed)
    env = make_env(task, gui=args.gui)

    device = get_device()
    print(device)

    policy_kwargs = dict(
        features_extractor_class=fnetwork,
        features_extractor_kwargs=dict(features_dim=96),
        net_arch=dict(pi=[128, 64, 32, 16], vf=[128, 64, 32, 16]),
        share_features_extractor=False,
        log_std_init=-1.0,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    learner = PPO(
        env=env,
        policy=ActorCriticPolicy,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.9,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=True,
        vf_coef=0.15,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        seed=random_seed,
        device=device_str,
        verbose=1,
    )
    learner.set_random_seed(random_seed)

    dataset = BehavioralCloningDataset(observations, actions_np)
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_bc,
        drop_last=False,
    )

    policy = learner.policy
    optimizer = optim.Adam(policy.parameters(), lr=args.bc_lr)
    losses: list[float] = []

    policy.train()
    for epoch in range(args.train_step):
        epoch_losses: list[float] = []
        for batch_obs, batch_acts in dataloader:
            batch_obs = {k: v.to(device) for k, v in batch_obs.items()}
            batch_acts = batch_acts.to(device)

            latent_pi, _ = policy.mlp_extractor(policy.features_extractor(batch_obs))
            dist = policy._get_action_dist_from_latent(latent_pi)
            loss = -dist.log_prob(batch_acts).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"Epoch {epoch + 1}/{args.train_step}, mean BC loss: {mean_loss:.6f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # SB3 writes ``<path>.zip``; pass basename without trailing .zip
    save_base = out_path.stem if out_path.suffix.lower() == ".zip" else out_path.name
    save_dir = out_path.parent / save_base
    learner.save(str(save_dir))
    print(f"Saved policy to {save_dir}.zip")

    loss_plot_path = out_path.parent / "bc_loss_curve.png"
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="BC mean NLL", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Behaviour cloning loss")
    plt.legend()
    plt.grid()
    plt.savefig(loss_plot_path, dpi=120)
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    if args.testing:
        policy.eval()
        for episode in range(args.testing_episode):
            obs, _ = env.reset(seed=int(random_seed + 1000 + episode))
            done = False
            total_reward = 0.0
            step_count = 0
            while not done and step_count < args.max_steps_per_episode:
                action, _states = learner.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += float(reward)
                step_count += 1
            print(f"Episode {episode + 1}: return = {total_reward:.3f}, steps = {step_count}")

    env.close()


if __name__ == "__main__":
    main()
