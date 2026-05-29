#!/usr/bin/env python3
"""Launch TensorBoard for Swarm RL logs (run from repo root in a separate terminal)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_logdir(raw: str) -> Path:
    """Resolve logdir from repo root (cwd), not relative to this script's folder."""
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Log directory does not exist: {path}")
    return path


def _prefer_airl_summary(path: Path) -> Path:
    """imitation AIRL writes event files under ``<log_dir>/summary/``."""
    summary = path / "summary"
    if summary.is_dir():
        events = list(summary.glob("events.out.tfevents.*"))
        if events:
            print(f"Using AIRL summary events: {summary}", flush=True)
            return summary
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Start TensorBoard web UI for Swarm RL logs")
    parser.add_argument(
        "--logdir",
        type=str,
        default="RL/train_scripts/logs/pretrain/",
        help="Log root (SB3) or AIRL log_dir (events often in <logdir>/summary/)",
    )
    args = parser.parse_args()

    log_path = _prefer_airl_summary(_resolve_logdir(args.logdir))

    print(f"TensorBoard logdir={log_path}", flush=True)
    print("Open http://localhost:6006/ after startup.", flush=True)
    sys.exit(subprocess.call(["tensorboard", f"--logdir={log_path}"]))


if __name__ == "__main__":
    main()
