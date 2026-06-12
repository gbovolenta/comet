"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return its contents.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        The parsed YAML content as a dictionary.
    """

    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf8") as fh:
        return yaml.safe_load(fh)
