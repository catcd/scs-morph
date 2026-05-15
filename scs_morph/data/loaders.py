from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    return pd.read_csv(path_obj, low_memory=False)


def load_all_raw_datasets(paths_config: dict[str, Any], datasets_config: dict[str, str]) -> dict[str, pd.DataFrame]:
    raw_dir = Path(paths_config["raw_dir"])
    datasets: dict[str, pd.DataFrame] = {}

    for dataset_id, file_name in datasets_config.items():
        file_path = raw_dir / file_name
        datasets[dataset_id] = load_raw_csv(str(file_path))

    return datasets
