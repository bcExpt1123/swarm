#!/usr/bin/env python3
"""BC pretraining + PPO warm-start for Swarm drone navigation."""
# SWARM_TRAIN_SCRIPT=train_pretrain  (marker: if you do not see this line in output, wrong program ran)

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Banner before heavy imports — must NOT print "Serving TensorBoard" (that is the TB *server*, not this script).
print("=" * 60, flush=True)
print("SWARM train_pretrain.py", flush=True)
print("Python:", sys.executable, flush=True)
print("=" * 60, flush=True)

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
from pathlib import Path

print("Loading training modules (may take 30–60s on first import)…", flush=True)
from RL.framework.config.loader import load_config
from RL.framework.training.bc import run_bc_pretrain
from RL.framework.training.trainer import train_ppo
from RL.framework.utils.paths import ensure_import_paths

ensure_import_paths()
print("Modules loaded.", flush=True)

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "pretrain_default.yaml"


def _resolve_resume_path(config, bc_out: Path, skip_bc: bool) -> None:
    if config.resume_path and Path(config.resume_path).is_file():
        return
    candidates = [
        bc_out.with_suffix(".zip"),
        Path(config.checkpoint.dir) / "ppo_policy.zip",
        Path(config.checkpoint.dir) / "bc_pretrain.zip",
    ]
    for path in candidates:
        if path.is_file():
            config.resume_path = str(path)
            print(f"Using checkpoint: {config.resume_path}", flush=True)
            return
    if skip_bc:
        raise FileNotFoundError(
            "No policy checkpoint found for --skip-bc. Train BC first (omit --skip-bc), or pass:\n"
            "  --resume path/to/model.zip\n"
            "Expected one of:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="BC + PPO pretrain for MovingDroneAviary")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--skip-bc", action="store_true", help="Skip behavior cloning phase.")
    parser.add_argument("--skip-ppo", action="store_true", help="Only run BC.")
    parser.add_argument("--bc-output", type=str, default="", help="BC checkpoint path.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="PPO checkpoint .zip (required for --skip-bc if no bc_pretrain.zip yet).",
    )
    args = parser.parse_args()
    print("argv:", sys.argv, flush=True)

    config = load_config(args.config)
    if args.resume:
        config.resume_path = args.resume

    ckpt_dir = Path(config.checkpoint.dir)
    bc_out = Path(args.bc_output) if args.bc_output else ckpt_dir / "bc_pretrain"

    if not args.skip_bc:
        print("=== Phase 1: Behavior Cloning ===", flush=True)
        run_bc_pretrain(config, bc_out)
        if not config.resume_path:
            config.resume_path = str(bc_out.with_suffix(".zip"))

    if args.skip_ppo:
        return

    if args.skip_bc or not config.resume_path:
        _resolve_resume_path(config, bc_out, skip_bc=args.skip_bc)

    print("=== Phase 2: PPO pretrain ===", flush=True)
    train_ppo(config, finetune=False)
    print(
        f"Done. Policy: swarm/submission_template/{config.checkpoint.submission_output}.zip",
        flush=True,
    )


if __name__ == "__main__":
    main()
