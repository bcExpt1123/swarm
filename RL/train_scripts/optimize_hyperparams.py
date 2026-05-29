#!/usr/bin/env python3
"""Optuna hyperparameter search for PPO on MovingDroneAviary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.framework.config.loader import load_config
from RL.framework.optimization.study import run_optuna_study
from RL.framework.utils.paths import ensure_import_paths

ensure_import_paths()

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "optuna_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for Swarm PPO")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--n-trials", type=int, default=0, help="Override YAML n_trials.")
    args = parser.parse_args()

    config = load_config(args.config)
    n_trials = args.n_trials or config.optuna.n_trials
    study = run_optuna_study(args.config, n_trials=n_trials)

    out = Path(config.checkpoint.dir) / "best_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params written to {out}")


if __name__ == "__main__":
    main()
