#!/usr/bin/env python3
"""Evaluate a trained PPO policy on random validator-style tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_baselines3 import PPO

from RL.framework.config.loader import load_config
from RL.framework.evaluation.evaluator import evaluate_model
from RL.framework.utils.paths import ensure_import_paths

ensure_import_paths()

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "finetune_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Swarm drone policy")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--model", type=str, required=True, help="Path to .zip policy.")
    parser.add_argument("--n-episodes", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    config = load_config(args.config)
    n_ep = args.n_episodes or config.evaluation.n_episodes
    model = PPO.load(args.model)
    summary = evaluate_model(model, config, n_episodes=n_ep, deterministic=config.evaluation.deterministic)

    report = {
        "success_rate": summary.success_rate,
        "collision_rate": summary.collision_rate,
        "mean_reward": summary.mean_reward,
        "mean_length": summary.mean_length,
        "path_efficiency": summary.path_efficiency,
        "action_smoothness": summary.action_smoothness,
        "mean_energy": summary.mean_energy,
        "mean_validator_score": summary.mean_validator_score,
        "composite": summary.composite_objective(config.evaluation.metrics_weights),
    }
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
