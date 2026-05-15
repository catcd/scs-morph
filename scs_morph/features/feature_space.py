from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeatureSpace:
    name: str
    dataset_column: str
    feature_names_file: str | None
    family: str
    pipeline: str
    atlas: str


def load_feature_spaces(config_path: str) -> dict[str, FeatureSpace]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    spaces: dict[str, FeatureSpace] = {}
    for name, config in raw_config.items():
        spaces[name] = FeatureSpace(
            name=name,
            dataset_column=config["dataset_column"],
            feature_names_file=config.get("feature_names_file"),
            family=config["family"],
            pipeline=config["pipeline"],
            atlas=config["atlas"],
        )
    return spaces
