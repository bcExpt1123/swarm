#!/usr/bin/env python3
"""AIRL adversarial imitation + generator PPO updates."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.framework.config.loader import load_config
from RL.framework.training.airl import run_airl_training
from RL.framework.utils.paths import ensure_import_paths

ensure_import_paths()

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "airl_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="AIRL training on Swarm drone env")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    print("=== AIRL training ===")
    run_airl_training(config)
    print(f"Policies saved under {config.airl.policy_dir}/ppo_policy_*.zip")
    print("Next: train_finetune.py --resume <latest airl policy>")


if __name__ == "__main__":
    main()
