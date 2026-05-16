from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils import resolve_path


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ["processed_data_dir", "feature_data_dir", "model_dir", "output_dir"]:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
        resolve_path(config[key]).mkdir(parents=True, exist_ok=True)
