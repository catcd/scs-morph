from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from scs_morph.data.feature_names import load_feature_names
from scs_morph.data.vector_parser import parse_vector_cell


def _is_missing_value(cell: Any) -> bool:
    if cell is None:
        return True
    if isinstance(cell, float) and np.isnan(cell):
        return True
    if isinstance(cell, str) and cell.strip() == "":
        return True
    return False


def validate_feature_column(
    df: pd.DataFrame,
    column: str,
    feature_names: list[str] | None = None,
    expected_dim: int | None = None,
    sample_n: int | None = None,
) -> dict[str, Any]:
    n_rows = len(df)
    n_missing = 0
    n_parse_success = 0
    n_parse_failed = 0
    inferred_dims: set[int] = set()
    errors: list[str] = []

    if expected_dim is None and feature_names is not None:
        expected_dim = len(feature_names)

    series = df[column]
    if sample_n is not None:
        series = series.head(sample_n)

    for idx, cell in series.items():
        if _is_missing_value(cell):
            n_missing += 1
            continue

        try:
            parsed = parse_vector_cell(cell)
        except ValueError as exc:
            n_parse_failed += 1
            errors.append(f"Row {idx}: {exc}")
            continue

        if parsed.size == 0:
            n_missing += 1
            continue

        n_parse_success += 1
        inferred_dims.add(int(parsed.size))

        if expected_dim is not None and parsed.size != expected_dim:
            errors.append(f"Row {idx}: dimension {parsed.size}, expected {expected_dim}")

    dim_matches = None
    if expected_dim is not None:
        dim_matches = all(dim == expected_dim for dim in inferred_dims)

    return {
        "column": column,
        "n_rows": n_rows,
        "n_missing": n_missing,
        "n_parse_success": n_parse_success,
        "n_parse_failed": n_parse_failed,
        "inferred_dims": sorted(inferred_dims),
        "expected_dim": expected_dim,
        "dim_matches": dim_matches,
        "all_errors": " | ".join(errors) if errors else None,
    }


def validate_all_feature_spaces(
    raw_datasets: dict[str, pd.DataFrame],
    feature_spaces_config: dict[str, dict[str, Any]],
    raw_dir: str,
    sample_n: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_path = Path(raw_dir)

    for dataset_name, df in raw_datasets.items():
        for space_name, space_config in feature_spaces_config.items():
            column = space_config.get("dataset_column")
            feature_names_file = space_config.get("feature_names_file")
            feature_names = None
            expected_dim = None

            if feature_names_file:
                names_path = raw_path / feature_names_file
                if names_path.exists():
                    feature_names = load_feature_names(str(names_path))
                    expected_dim = len(feature_names)
                else:
                    warnings.warn(
                        f"Feature-name file not found for {space_name}: {names_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            if not isinstance(column, str) or column not in df.columns:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "feature_space": space_name,
                        "column": column,
                        "missing_column": True,
                        "n_rows": len(df),
                        "n_missing": None,
                        "n_parse_success": None,
                        "n_parse_failed": None,
                        "inferred_dims": None,
                        "expected_dim": expected_dim,
                        "dim_matches": None,
                        "all_errors": None,
                    }
                )
                continue

            metrics = validate_feature_column(
                df,
                column,
                feature_names=feature_names,
                expected_dim=expected_dim,
                sample_n=sample_n,
            )
            metrics.update(
                {
                    "dataset": dataset_name,
                    "feature_space": space_name,
                    "column": column,
                    "missing_column": False,
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)
