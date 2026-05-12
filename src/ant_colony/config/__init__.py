"""Configuration loader for the ant colony simulation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load world configuration from a YAML file.

    Parameters
    ----------
    path : str or Path or None
        Path to the config file.  If *None*, the bundled default is used.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def get_food_defs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the compact food config into a list of food-type definitions."""
    food_cfg = cfg["food"]
    return [
        {
            "name": "bush",
            "count": food_cfg["bush_count"],
            "value": food_cfg["bush_value"],
            "respawn_ticks": food_cfg["bush_respawn_ticks"],
            "symbol": "B",
        },
        {
            "name": "mushroom",
            "count": food_cfg["mushroom_count"],
            "value": food_cfg["mushroom_value"],
            "respawn_ticks": food_cfg["mushroom_respawn_ticks"],
            "symbol": "M",
        },
        {
            "name": "crystal",
            "count": food_cfg["crystal_count"],
            "value": food_cfg["crystal_value"],
            "respawn_ticks": food_cfg["crystal_respawn_ticks"],
            "symbol": "C",
        },
    ]
