"""Deterministic seeding across Python, NumPy, PyTorch, and SB3."""

from __future__ import annotations

import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
