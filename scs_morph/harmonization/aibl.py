from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from scs_morph.harmonization.labels import derive_aibl_dx, make_binary_label
from scs_morph.harmonization.normalizers import (
    normalize_field_strength,
    normalize_numeric,
    normalize_protocol_string,
    normalize_sex,
    normalize_subject_id,
)


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def harmonize_aibl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = "AIBL"
    df["cohort"] = "AIBL"

    subject_column = _first_matching_column(df, ["Subject ID", "Image ID", "ImageID", "subject_id"])
    image_column = _first_matching_column(df, ["Image ID", "ImageID", "Image_ID"])

    df["subject_id"] = df[subject_column].apply(normalize_subject_id) if subject_column else np.nan
    df["scan_id"] = df[image_column].apply(normalize_subject_id) if image_column else df["subject_id"]
    df["age"] = _series_or_nan(df, "Age").apply(normalize_numeric)
    df["sex"] = _series_or_nan(df, "Sex").apply(normalize_sex)
    df["education"] = np.nan
    df["scanner_protocol"] = _series_or_nan(df, "Imaging Protocol").apply(normalize_protocol_string)
    df["field_strength"] = _series_or_nan(df, "Imaging Protocol").apply(normalize_field_strength)

    df["diagnosis_raw"] = df.apply(lambda row: derive_aibl_dx(row), axis=1)
    df["diagnosis_3class"] = df["diagnosis_raw"]
    df["diagnosis_binary_ad_cn"] = make_binary_label(df["diagnosis_3class"], "AD", "CN")
    df["diagnosis_binary_mci_cn"] = make_binary_label(df["diagnosis_3class"], "MCI", "CN")
    df["diagnosis_binary_ad_mci"] = make_binary_label(df["diagnosis_3class"], "AD", "MCI")
    df["mmse"] = _series_or_nan(df, "MMSCORE").apply(normalize_numeric)
    df["icv"] = _series_or_nan(df, "ICV").apply(normalize_numeric)

    return df
