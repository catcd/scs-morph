from __future__ import annotations

from pathlib import Path
from typing import Any

from scs_morph.utils.io import load_yaml


def load_configs(config_dir: str) -> dict[str, Any]:
    config_path = Path(config_dir)
    paths = load_yaml(str(config_path / "paths.yaml"))
    datasets = load_yaml(str(config_path / "datasets.yaml"))
    feature_spaces = load_yaml(str(config_path / "feature_spaces.yaml"))
    return {
        "paths": paths,
        "datasets": datasets,
        "feature_spaces": feature_spaces,
    }
