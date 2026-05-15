from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from scs_morph.harmonization.labels import make_binary_label, normalize_adni_dx
from scs_morph.harmonization.normalizers import (
    normalize_field_strength,
    normalize_ethnicity,
    normalize_image_uid,
    normalize_marital_status,
    normalize_numeric,
    normalize_race,
    normalize_sex,
    normalize_subject_id,
    normalize_visit_code,
)


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def extract_adni_path_fields(file_path: Any) -> dict[str, Any]:
    if pd.isna(file_path):
        return {"preprocessing_string": np.nan, "image_uid": np.nan}

    path = str(file_path).replace("\\", "/")
    components = [component for component in path.split("/") if component]
    if len(components) < 4:
        return {"preprocessing_string": np.nan, "image_uid": np.nan}

    preprocessing_string = components[1] if len(components) > 1 else np.nan
    image_uid_component = components[3] if len(components) > 3 else components[-1]
    image_uid = normalize_image_uid(image_uid_component)

    return {"preprocessing_string": preprocessing_string, "image_uid": image_uid}


def harmonize_adni(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "file_path" in df.columns:
        path_info = df["file_path"].apply(extract_adni_path_fields).apply(pd.Series)
        df = pd.concat([df, path_info], axis=1)
    else:
        df["preprocessing_string"] = np.nan
        df["image_uid"] = np.nan

    df["dataset"] = "ADNI"
    df["cohort"] = df.get("COLPROT")
    ptid_series = df["PTID"] if "PTID" in df.columns else pd.Series(np.nan, index=df.index)
    df["subject_id"] = ptid_series.apply(normalize_subject_id)
    if "IMAGEUID" in df.columns:
        df["image_uid"] = df["image_uid"].fillna(df["IMAGEUID"].apply(normalize_image_uid))
    df["scan_id"] = df["image_uid"].fillna("")
    empty_scan_id = df["scan_id"] == ""
    df.loc[empty_scan_id, "scan_id"] = df.loc[empty_scan_id].index.astype(str)

    if "VISCODE" in df.columns and "VISCODE2" in df.columns:
        df["visit_code"] = df["VISCODE"].combine_first(df["VISCODE2"]).apply(normalize_visit_code)
    elif "VISCODE" in df.columns:
        df["visit_code"] = df["VISCODE"].apply(normalize_visit_code)
    elif "VISCODE2" in df.columns:
        df["visit_code"] = df["VISCODE2"].apply(normalize_visit_code)
    else:
        df["visit_code"] = np.nan

    df["age"] = _series_or_nan(df, "AGE").apply(normalize_numeric)
    df["sex"] = _series_or_nan(df, "PTGENDER").apply(normalize_sex)
    df["education"] = _series_or_nan(df, "PTEDUCAT").apply(normalize_numeric)
    df["ethnicity"] = _series_or_nan(df, "PTETHCAT").apply(normalize_ethnicity)
    df["race"] = _series_or_nan(df, "PTRACCAT").apply(normalize_race)
    df["marital_status"] = _series_or_nan(df, "PTMARRY").apply(normalize_marital_status)
    df["field_strength"] = _series_or_nan(df, "FLDSTRENG").apply(normalize_field_strength)
    df["diagnosis_raw"] = _series_or_nan(df, "DX")
    df["diagnosis_3class"] = df["diagnosis_raw"].apply(normalize_adni_dx)
    df["diagnosis_binary_ad_cn"] = make_binary_label(df["diagnosis_3class"], "AD", "CN")
    df["diagnosis_binary_mci_cn"] = make_binary_label(df["diagnosis_3class"], "MCI", "CN")
    df["diagnosis_binary_ad_mci"] = make_binary_label(df["diagnosis_3class"], "AD", "MCI")
    df["mmse"] = _series_or_nan(df, "MMSE").apply(normalize_numeric)
    df["icv"] = _series_or_nan(df, "ICV").apply(normalize_numeric)

    return df
