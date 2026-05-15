from __future__ import annotations

import pandas as pd


def apply_inclusion_filter(metadata: pd.DataFrame, inclusion_filter: str | None) -> pd.Series:
    """Return a boolean mask for a pandas query-style inclusion filter."""
    if not inclusion_filter:
        return pd.Series(True, index=metadata.index)

    matched_index = metadata.query(inclusion_filter).index
    return pd.Series(metadata.index.isin(matched_index), index=metadata.index)


def apply_cohort_filter(metadata: pd.DataFrame, cohort: str | None) -> pd.Series:
    """Return a boolean mask for cohort selection; 'pooled' keeps all rows."""
    if cohort is None or str(cohort).lower() == "pooled" or "cohort" not in metadata.columns:
        return pd.Series(True, index=metadata.index)
    return metadata["cohort"].astype(str) == str(cohort)
