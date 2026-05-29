"""Ensure repo root is on sys.path before framework imports."""

from __future__ import annotations

from RL.framework.utils.paths import ensure_import_paths

ensure_import_paths()
