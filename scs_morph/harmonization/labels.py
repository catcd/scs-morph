import re

import numpy as np
import pandas as pd


def normalize_adni_dx(value) -> str:
    if pd.isna(value):
        return "OTHER"
    text = str(value).strip().lower()
    if text in {"cn", "normal", "cognitively normal"}:
        return "CN"
    if text in {"mci", "emci", "lmci", "smc", "mild cognitive impairment"}:
        return "MCI"
    if text in {"ad", "dementia", "alzheimers", "alzheimer's"}:
        return "AD"
    return "OTHER"


def make_binary_label(series: pd.Series, positive: str, negative: str) -> pd.Series:
    return series.map(lambda value: 1 if str(value).upper() == positive else 0 if str(value).upper() == negative else np.nan)


def derive_aibl_dx(row) -> str:
    if row.get("DXNORM") == 1 or str(row.get("DXNORM")).strip() == "1":
        return "CN"
    if row.get("DXMCI") == 1 or str(row.get("DXMCI")).strip() == "1":
        return "MCI"
    if row.get("DXAD") == 1 or str(row.get("DXAD")).strip() == "1":
        return "AD"
    return "OTHER"


def derive_oasis_cdr_group(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "OTHER"

    if numeric == 0:
        return "CN_like"
    if numeric > 0:
        return "impaired_like"
    return "OTHER"
