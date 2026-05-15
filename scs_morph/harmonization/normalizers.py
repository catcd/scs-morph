from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

MISSING_STRINGS = {"", "nan", "na", "none", "null", "unknown", "n/a"}


def _normalize_text(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan
    return text


def normalize_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan

    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", text)
        if match:
            return float(match.group(0))

    return np.nan


def normalize_subject_id(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and not np.isnan(value):
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_image_uid(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float) and not np.isnan(value):
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan

    match = re.search(r"I(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    digits = re.search(r"(\d+)", text)
    if digits:
        return digits.group(1)

    return text


def normalize_sex(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if text in {"m", "male", "1", "1.0"}:
        return "M"
    if text in {"f", "female", "2", "2.0"}:
        return "F"
    if text in {"x", "other", "unknown", "unspecified"}:
        return "X"
    if text in MISSING_STRINGS:
        return np.nan
    return text.upper()


def normalize_field_strength(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool):
        digit = float(value)
        if np.isnan(digit):
            return np.nan
        if digit.is_integer():
            return f"{int(digit)}T"
        return f"{digit:g}T"

    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan

    match = re.search(r"Field Strength\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:Tesla|T)\b", text, re.IGNORECASE)
    if match:
        strength = float(match.group(1))
        if strength.is_integer():
            return f"{int(strength)}T"
        return f"{strength:g}T"

    return np.nan


def normalize_ethnicity(value: Any) -> str | float:
    normalized = _normalize_text(value)
    if pd.isna(normalized):
        return np.nan

    text = normalized.lower()
    if "hisp" in text and "not" not in text:
        return "hispanic_latino"
    if "not" in text or "non" in text:
        return "not_hispanic_latino"
    if "unknown" in text:
        return "unknown"
    return normalized


def normalize_race(value: Any) -> str | float:
    normalized = _normalize_text(value)
    if pd.isna(normalized):
        return np.nan

    text = normalized.lower()
    if "white" in text:
        return "white"
    if "black" in text or "african" in text:
        return "black"
    if "asian" in text:
        return "asian"
    if "more than one" in text:
        return "more_than_one"
    if "am indian" in text or "alaskan" in text:
        return "american_indian_alaskan"
    if "hawaiian" in text or "pacific" in text:
        return "hawaiian_other_pacific_islander"
    if "unknown" in text:
        return "unknown"
    return normalized


def normalize_marital_status(value: Any) -> str | float:
    normalized = _normalize_text(value)
    if pd.isna(normalized):
        return np.nan

    text = normalized.lower()
    if "married" in text:
        return "married"
    if "widowed" in text:
        return "widowed"
    if "divorced" in text:
        return "divorced"
    if "never" in text or "single" in text:
        return "never_married"
    if "unknown" in text:
        return "unknown"
    return normalized


def normalize_visit_code(value: Any) -> str | float:
    normalized = _normalize_text(value)
    if pd.isna(normalized):
        return np.nan
    return normalized.lower()


def normalize_protocol_string(value: Any) -> str | float:
    normalized = _normalize_text(value)
    if pd.isna(normalized):
        return np.nan
    return re.sub(r"\s*;\s*", "; ", str(normalized))
