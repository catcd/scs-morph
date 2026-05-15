from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_column_values(series: pd.Series, max_values: int = 10) -> dict[str, Any]:
    non_missing = int(series.notna().sum())
    missing = int(series.isna().sum())
    unique_values = int(series.dropna().astype(str).str.strip().nunique())
    dtype = str(series.dtype)
    top_counts = series.dropna().astype(str).str.strip().value_counts().head(max_values)
    top_values = "; ".join([f"{value}:{count}" for value, count in top_counts.items()])

    return {
        "dtype": dtype,
        "unique_values": unique_values,
        "non_missing": non_missing,
        "missing": missing,
        "top_values": top_values,
    }


def audit_dataset_encodings(dataset_name: str, df: pd.DataFrame, max_values: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in sorted(df.columns):
        summary = summarize_column_values(df[column], max_values=max_values)
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def audit_all_raw_datasets(datasets: dict[str, pd.DataFrame], max_values: int = 10) -> pd.DataFrame:
    audit_rows: list[pd.DataFrame] = []
    for dataset_name, df in sorted(datasets.items()):
        audit_rows.append(audit_dataset_encodings(dataset_name, df, max_values=max_values))
    if audit_rows:
        return pd.concat(audit_rows, ignore_index=True)
    return pd.DataFrame(
        columns=["dataset", "column", "dtype", "unique_values", "non_missing", "missing", "top_values"]
    )
