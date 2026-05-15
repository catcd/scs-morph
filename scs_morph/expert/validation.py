from __future__ import annotations

import re
import pandas as pd

BAD_TEXT_PATTERN = re.compile(r"(^|\s)nan([,.\s]|$)|none|unknown_network|unknown_subregion|[{}]", re.IGNORECASE)


def has_bad_claim_text(text) -> bool:
    if text is None or pd.isna(text):
        return True
    text = str(text).strip()
    return len(text) < 10 or bool(BAD_TEXT_PATTERN.search(text))


def normalize_text(text: str) -> str:
    text = re.sub(r"\b(adni|aibl|oasis1?|spm|cat12|freesurfer|fastsurfer)\b", "", str(text).lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

