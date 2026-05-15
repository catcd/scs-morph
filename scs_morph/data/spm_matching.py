import re

import pandas as pd


def normalize_image_uid(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    digits = re.findall(r"\d+", text)
    if not digits:
        return None
    return digits[0]


def match_adni_spm(
    adni_harmonized: pd.DataFrame,
    spm_raw: pd.DataFrame,
    join_type: str = "inner",
) -> pd.DataFrame:
    adni = adni_harmonized.copy()
    spm = spm_raw.copy()

    adni["image_uid_norm"] = adni["image_uid"].apply(normalize_image_uid)
    spm["IMAGEUID_norm"] = spm["IMAGEUID"].apply(normalize_image_uid)

    joined = adni.merge(
        spm,
        how=join_type,
        left_on="image_uid_norm",
        right_on="IMAGEUID_norm",
        suffixes=("", "_spm"),
    )
    return joined
