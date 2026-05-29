"""Behavior cloning pretraining (wraps existing pretrain_ppo logic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device, set_random_seed
from torch.utils.data import DataLoader, Dataset

from RL.framework.config.schemas import TrainingConfig
from RL.framework.policies.builder import build_policy_kwargs
from RL.framework.utils.paths import ensure_import_paths
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task


class _BCDataset(Dataset):
    def __init__(self, observations: list, actions: np.ndarray):
        self.observations = observations
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        return self.observations[idx], self.actions[idx]


def collate_bc(batch):
    """Stack Dict observations for SB3 policy (required; default collate breaks)."""
    obs_list = [b[0] for b in batch]
    acts = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
    depth = np.stack([o["depth"] for o in obs_list], axis=0)
    state = np.stack([o["state"] for o in obs_list], axis=0)
    return {
        "depth": torch.from_numpy(depth),
        "state": torch.from_numpy(state),
    }, torch.from_numpy(acts)


def collect_random_rollouts(config: TrainingConfig) -> tuple[list, np.ndarray]:
    observations: list = []
    actions: list = []
    bc = config.bc
    print(
        f"Collecting {bc.collect_episodes} random episodes "
        f"(up to {bc.max_steps_per_episode} steps each) — this can take a while…",
        flush=True,
    )
    for ep in range(bc.collect_episodes):
        task = random_task(sim_dt=SIM_DT, seed=config.seed + ep)
        env = make_env(task, gui=False)
        try:
            obs, _ = env.reset(seed=task.map_seed)
            for step in range(bc.max_steps_per_episode):
                action = env.action_space.sample().astype(np.float32)
                observations.append(obs)
                actions.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        if (ep + 1) % max(1, bc.collect_episodes // 10) == 0 or ep == 0:
            print(
                f"  BC collection: episode {ep + 1}/{bc.collect_episodes}, "
                f"{len(actions)} transitions so far",
                flush=True,
            )
    if not actions:
        raise RuntimeError("No BC transitions collected; check env reset/step.")
    print(f"Collected {len(actions)} transitions.", flush=True)
    return observations, np.asarray(actions, dtype=np.float32)


def run_bc_pretrain(config: TrainingConfig, output_path: Path) -> Path:
    """Collect rollouts and fit policy with NLL (behavior cloning)."""
    ensure_import_paths()
    set_random_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(get_device(), flush=True)

    observations, actions = collect_random_rollouts(config)
    dataset = _BCDataset(observations, actions)
    batch_size = min(config.bc.batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_bc,
        drop_last=len(dataset) > batch_size,
    )

    task = random_task(sim_dt=SIM_DT, seed=config.seed)
    env = make_env(task, gui=False)
    policy_kwargs = build_policy_kwargs(config.policy)
    model = PPO(
        ActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.ppo.learning_rate,
        seed=config.seed,
        device=device,
        verbose=0,
    )
    env.close()

    optimizer = optim.Adam(model.policy.parameters(), lr=config.bc.bc_lr)
    model.policy.train()
    print(f"BC training for {config.bc.train_steps} epochs…", flush=True)
    for step in range(config.bc.train_steps):
        epoch_losses: list[float] = []
        for batch_obs, batch_acts in loader:
            batch_obs = {k: v.to(device) for k, v in batch_obs.items()}
            batch_acts = batch_acts.to(device)
            distribution = model.policy.get_distribution(batch_obs)
            log_prob = distribution.log_prob(batch_acts)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        if (step + 1) % 10 == 0 or step == 0:
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"BC step {step + 1}/{config.bc.train_steps} loss={mean_loss:.4f}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"BC checkpoint saved to {output_path.with_suffix('.zip')}", flush=True)
    return output_path.with_suffix(".zip")
