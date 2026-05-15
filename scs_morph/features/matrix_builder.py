from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd

from scs_morph.data.feature_names import load_feature_names
from scs_morph.data.vector_parser import MISSING_VECTOR_VALUE, parse_vector_cell
from scs_morph.utils.io import ensure_dir, save_json, save_table
from scs_morph.features.feature_space import FeatureSpace


def build_feature_matrix(
    df: pd.DataFrame,
    feature_space: FeatureSpace,
    raw_dir: str,
    index_col: str | None = None,
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_space.dataset_column not in df.columns:
        raise KeyError(f"Feature column '{feature_space.dataset_column}' not found.")

    feature_names: list[str] | None = None
    expected_dim = None
    if feature_space.feature_names_file:
        names_path = Path(raw_dir) / feature_space.feature_names_file
        if names_path.exists():
            feature_names = load_feature_names(str(names_path))
            expected_dim = len(feature_names)
        else:
            warnings.warn(
                f"Feature-name file not found for {feature_space.name}: {names_path}",
                RuntimeWarning,
                stacklevel=2,
            )

    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    indices: list[Any] = []

    for idx, row in df.iterrows():
        cell = row[feature_space.dataset_column]
        vector_status = "ok"
        try:
            values = parse_vector_cell(cell)
        except ValueError:
            if expected_dim is not None:
                values = np.full(expected_dim, MISSING_VECTOR_VALUE, dtype=float)
                vector_status = "parse_failed_all_missing"
            elif drop_missing:
                continue
            else:
                raise

        if values.size == 0:
            if expected_dim is not None:
                values = np.full(expected_dim, MISSING_VECTOR_VALUE, dtype=float)
                vector_status = "missing_all"
            elif drop_missing:
                continue

        if expected_dim is not None and values.size != expected_dim:
            found_dim = values.size
            values = np.full(expected_dim, MISSING_VECTOR_VALUE, dtype=float)
            vector_status = f"dimension_mismatch_all_missing:{found_dim}->{expected_dim}"

        if values.size == 0 and drop_missing:
            continue

        if feature_names is None:
            columns = [f"feature_{i}" for i in range(values.size)]
        else:
            columns = feature_names

        row_values = {name: float(value) for name, value in zip(columns, values)}
        rows.append(row_values)

        metadata_row = row.drop(labels=[feature_space.dataset_column]).to_dict()
        metadata_row[f"{feature_space.name}_vector_status"] = vector_status
        metadata.append(metadata_row)
        if index_col is not None and index_col in row:
            indices.append(row[index_col])
        else:
            indices.append(idx)

    X = pd.DataFrame(rows, index=pd.Index(indices, name=index_col or None))
    meta = pd.DataFrame(metadata, index=pd.Index(indices, name=index_col or None))

    return X, meta


def save_feature_matrix(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
    feature_space_name: str,
) -> None:
    base_path = Path(output_dir) / dataset_name / feature_space_name
    ensure_dir(str(base_path))
    save_table(X, str(base_path / "X"))
    save_table(meta, str(base_path / "meta"))
    metadata = {
        "dataset": dataset_name,
        "feature_space": feature_space_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "features": list(X.columns),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    save_json(metadata, str(base_path / "metadata.json"))
