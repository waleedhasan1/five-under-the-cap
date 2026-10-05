"""Load project settings from ``config/settings.yaml``."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Return the parsed settings file.

    Args:
        path: YAML file to read. Defaults to ``config/settings.yaml``.

    Returns:
        The settings as nested dicts.
    """
    with open(path) as f:
        settings: dict[str, Any] = yaml.safe_load(f)
    return settings


def all_seasons(settings: dict[str, Any] | None = None) -> list[str]:
    """Training seasons followed by the test season, oldest first."""
    s = settings or load_settings()
    return [*s["seasons"]["train"], s["seasons"]["test"]]


def path(key: str, settings: dict[str, Any] | None = None) -> Path:
    """Resolve a ``paths`` entry from settings to an absolute path under the repo root."""
    s = settings or load_settings()
    return ROOT / s["paths"][key]


def config_hash(path: Path = SETTINGS_PATH) -> str:
    """Short SHA-256 of the settings file, stored in model metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
