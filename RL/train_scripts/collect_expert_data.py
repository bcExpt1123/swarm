#!/usr/bin/env python3
"""Collect expert demonstrations for AIRL / offline analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO

from RL.framework.config.loader import load_config
from RL.framework.data.expert_data import save_trajectories
from RL.framework.training.airl import make_airl_vec_env
from RL.framework.utils.paths import ensure_import_paths
from RL.framework.utils.seeding import set_global_seed

ensure_import_paths()

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "airl_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect expert rollouts for AIRL")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--policy", type=str, default="RL/train_scripts/checkpoints/pretrain/ppo_policy.zip", help="Expert PPO .zip (default: airl.pretrain_path).")
    parser.add_argument("--output", type=str, default="RL/train_scripts/expert/expert_data.pkl", help="Path to save collected trajectories (pickle).")
    parser.add_argument("--min-episodes", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    policy_path = Path(args.policy or config.airl.pretrain_path)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Expert policy not found: {policy_path}")

    out = Path(args.output or config.airl.expert_path)
    min_ep = args.min_episodes or config.airl.min_expert_episodes

    set_global_seed(config.seed)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    venv = make_airl_vec_env(1, config.seed, gui=config.vec_env.gui)
    expert = PPO.load(str(policy_path), env=venv, device=device)
    rng = np.random.default_rng(config.seed)

    print(f"Collecting >= {min_ep} episodes from {policy_path} …", flush=True)
    demos = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_episodes=min_ep),
        rng=rng,
    )
    save_trajectories(out, demos)
    print(f"Saved {len(demos)} trajectories to {out} (pickle; DictObs-safe)", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
