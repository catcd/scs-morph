import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_table(df: pd.DataFrame, path_without_ext: str) -> Path:
    output_path = Path(path_without_ext)
    ensure_dir(output_path.parent)
    parquet_path = output_path.with_suffix(".parquet")
    csv_path = output_path.with_suffix(".csv")

    try:
        df.to_parquet(parquet_path, index=True)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=True)
        return csv_path


def read_table(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".parquet":
        return pd.read_parquet(path_obj)
    if path_obj.suffix.lower() == ".csv":
        return pd.read_csv(path_obj, index_col=0)
    raise ValueError(f"Unsupported file type: {path_obj.suffix}")


def save_json(obj: Any, path: str) -> Path:
    ensure_dir(Path(path).parent.as_posix())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    return Path(path)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
