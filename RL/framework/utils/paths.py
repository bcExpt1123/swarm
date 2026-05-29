"""Repository path helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_import_paths() -> None:
    root = repo_root()
    rl_dir = root / "RL"
    for p in (rl_dir, root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def submission_template_dir() -> Path:
    return repo_root() / "swarm" / "submission_template"
