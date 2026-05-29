"""Checkpoint and VecNormalize persistence."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from stable_baselines3.common.vec_env import VecNormalize

from RL.framework.utils.paths import submission_template_dir


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 5):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save_model(self, model: Any, name: str) -> Path:
        path = self.dir / name
        model.save(str(path))
        self._prune_old(path.stem)
        return path.with_suffix(".zip")

    def save_vecnormalize(self, venv: VecNormalize, name: str = "vecnormalize") -> Path:
        path = self.dir / f"{name}.pkl"
        venv.save(str(path))
        return path

    def export_submission(self, model: Any, output_stem: str) -> Path:
        """Save policy where ``drone_agent.py`` expects ``ppo_policy.zip``."""
        out_dir = submission_template_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / output_stem
        model.save(str(stem))
        return stem.with_suffix(".zip")

    def _prune_old(self, prefix: str) -> None:
        zips = sorted(self.dir.glob(f"{prefix}*.zip"), key=lambda p: p.stat().st_mtime)
        while len(zips) > self.keep_last_n:
            oldest = zips.pop(0)
            oldest.unlink(missing_ok=True)

    def copy_resume(self, src: Path, dst_name: str) -> Path:
        dst = self.dir / dst_name
        shutil.copy2(src, dst)
        return dst
