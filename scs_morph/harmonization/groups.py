from __future__ import annotations

import pandas as pd


def select_first_scan_per_subject(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    age_col: str = "age",
) -> pd.DataFrame:
    grouped = (
        df.sort_values(by=[subject_col, age_col], na_position="last")
        .groupby(subject_col, sort=False)
        .head(1)
    )
    return grouped.reset_index(drop=True)
