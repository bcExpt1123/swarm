"""Save/load imitation trajectories with Dict observations (MovingDroneAviary).

``imitation.data.serialize`` uses HuggingFace datasets and raises
``ValueError: DictObs are not currently supported``. Use these helpers instead.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_trajectories(path: str | Path, trajectories: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_trajectories(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        return pickle.load(f)
