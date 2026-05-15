from __future__ import annotations

import numpy as np
import pandas as pd

from scs_morph.harmonization.labels import derive_oasis_cdr_group
from scs_morph.harmonization.normalizers import (
    normalize_numeric,
    normalize_sex,
    normalize_subject_id,
)


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def harmonize_oasis1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = "OASIS1"
    df["cohort"] = "OASIS1"
    df["subject_id"] = _series_or_nan(df, "ID").apply(normalize_subject_id)
    df["scan_id"] = _series_or_nan(df, "ID").apply(normalize_subject_id)
    df["age"] = _series_or_nan(df, "Age").apply(normalize_numeric)
    df["sex"] = _series_or_nan(df, "M/F").apply(normalize_sex)
    df["education"] = _series_or_nan(df, "Educ").apply(normalize_numeric)
    df["cdr"] = _series_or_nan(df, "CDR").apply(normalize_numeric)
    df["mmse"] = _series_or_nan(df, "MMSE").apply(normalize_numeric)
    df["diagnosis_raw"] = _series_or_nan(df, "CDR")
    df["diagnosis_3class"] = df["diagnosis_raw"].apply(derive_oasis_cdr_group)
    df["diagnosis_binary_impairment_cdr"] = df["diagnosis_3class"].map(
        {"impaired_like": 1, "CN_like": 0}
    )
    icv_source = df["ICV"] if "ICV" in df.columns else _series_or_nan(df, "eTIV")
    df["icv"] = icv_source.apply(normalize_numeric)
    df["field_strength"] = np.nan
    return df
