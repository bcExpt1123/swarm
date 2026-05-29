#!/usr/bin/env python3
"""Fine-tune PPO on harder curriculum stages (resume from pretrain/AIRL)."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from RL.framework.config.loader import load_config
from RL.framework.training.trainer import train_ppo
from RL.framework.utils.paths import ensure_import_paths

ensure_import_paths()

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "finetune_default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO fine-tuning for validator score")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--resume", type=str, default="", help="Override resume_path from YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config.resume_path = args.resume

    if not config.resume_path or not Path(config.resume_path).is_file():
        raise FileNotFoundError(
            f"Resume checkpoint required for fine-tune: {config.resume_path}. "
            "Run train_pretrain.py or train_airl.py first."
        )

    print("=== PPO fine-tune (gentler updates, validator-aligned reward) ===")
    train_ppo(config, finetune=True)
    print(f"Exported: swarm/submission_template/{config.checkpoint.submission_output}.zip")


if __name__ == "__main__":
    main()
