#!/usr/bin/env python3
"""Shared configuration loader for local env-style settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = BASE_DIR / ".env"


def _parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        values[key] = value

    return values


@lru_cache(maxsize=1)
def load_settings() -> dict[str, str]:
    """Load settings from .env into process env and return resolved values."""
    settings = _parse_env_file(DEFAULT_ENV_FILE)
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    return settings


def get_setting(key: str, default: str | None = None) -> str | None:
    """Read a string setting from process env, then .env, then default."""
    load_settings()
    return os.environ.get(key) or default


def get_int_setting(key: str, default: int) -> int:
    """Read an integer setting with fallback to default on invalid values."""
    raw_value = get_setting(key)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_path_setting(key: str, default: str) -> Path:
    """Read a path setting relative to the repo root unless absolute."""
    raw_value = get_setting(key, default) or default
    path = Path(raw_value)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path
