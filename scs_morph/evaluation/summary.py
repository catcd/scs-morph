from __future__ import annotations

import pandas as pd


def label_counts(df: pd.DataFrame, column: str) -> dict:
    if df.empty or column not in df.columns:
        return {}
    return df[column].fillna("NA").value_counts().to_dict()

