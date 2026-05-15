from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scs_morph.evaluation.matching import get_support_set


def _paired_arrays(x, y) -> tuple[np.ndarray, np.ndarray]:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def safe_spearman(x, y) -> float | None:
    a, b = _paired_arrays(x, y)
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return None
    value = stats.spearmanr(a, b).correlation
    return float(value) if np.isfinite(value) else None


def safe_pearson(x, y) -> float | None:
    a, b = _paired_arrays(x, y)
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return None
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else None


def jaccard(set_a, set_b) -> float | None:
    a, b = set(set_a), set(set_b)
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def _select(values, indices):
    arr = np.asarray(values, dtype=float)
    if indices is None:
        return arr
    return arr[list(indices)] if len(indices) else np.asarray([], dtype=float)


def direction_retention(source_values, target_values, source_support_indices=None) -> float | None:
    source, target = _paired_arrays(source_values, target_values)
    if source_support_indices is not None:
        source = _select(source, source_support_indices)
        target = _select(target, source_support_indices)
    mask = (source != 0) & (target != 0)
    if not mask.any():
        return None
    return float((np.sign(source[mask]) == np.sign(target[mask])).mean())


def sign_flip_fraction(source_values, target_values, support_indices=None) -> float | None:
    retained = direction_retention(source_values, target_values, support_indices)
    return None if retained is None else float(1 - retained)


def _z(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    std = np.nanstd(arr)
    return np.zeros_like(arr) if std == 0 or not np.isfinite(std) else (arr - np.nanmean(arr)) / std


def magnitude_drift(source_values, target_values, support_indices=None) -> float | None:
    source, target = _paired_arrays(source_values, target_values)
    if support_indices is not None:
        source = _select(source, support_indices)
        target = _select(target, support_indices)
    if len(source) == 0:
        return None
    return float(np.nanmean(np.abs(_z(source) - _z(target))))


def rank_overlap_topk(source_values, target_values, k: int = 20) -> float | None:
    source, target = _paired_arrays(source_values, target_values)
    if len(source) == 0:
        return None
    k = min(k, len(source))
    a = set(np.argsort(np.abs(source))[-k:])
    b = set(np.argsort(np.abs(target))[-k:])
    return jaccard(a, b)


def weighted_jaccard_from_abs_values(source_values, target_values) -> float | None:
    source, target = _paired_arrays(source_values, target_values)
    if len(source) == 0:
        return None
    source_abs, target_abs = np.abs(source), np.abs(target)
    denom = np.maximum(source_abs, target_abs).sum()
    if denom == 0:
        return None
    return float(np.minimum(source_abs, target_abs).sum() / denom)


def compute_evidence_stability(matched_df: pd.DataFrame, source_col: str = "source_t", target_col: str = "target_t", top_k: int = 20) -> dict:
    if matched_df is None or matched_df.empty or source_col not in matched_df or target_col not in matched_df:
        return {"evidence_spearman": None, "evidence_pearson": None, "rank_overlap_topk": None, "weighted_jaccard_abs": None, "n_matched": 0}
    source = matched_df[source_col]
    target = matched_df[target_col]
    return {
        "evidence_spearman": safe_spearman(source, target),
        "evidence_pearson": safe_pearson(source, target),
        "rank_overlap_topk": rank_overlap_topk(source, target, top_k),
        "weighted_jaccard_abs": weighted_jaccard_from_abs_values(source, target),
        "n_matched": int(len(_paired_arrays(source, target)[0])),
    }


def compute_support_stability(source_stats: pd.DataFrame, target_stats: pd.DataFrame, matched_df: pd.DataFrame, top_k: int = 20) -> dict:
    source_support = get_support_set(source_stats, top_k)
    target_support = get_support_set(target_stats, top_k)
    if matched_df is not None and not matched_df.empty and "roi" in matched_df.columns:
        matched_rois = set(matched_df["roi"].astype(str))
        source_support &= matched_rois
        target_support &= matched_rois
    overlap = source_support & target_support
    return {
        "source_support_size": len(source_support),
        "target_support_size": len(target_support),
        "support_jaccard": jaccard(source_support, target_support),
        "support_overlap_count": len(overlap),
        "support_overlap_fraction_source": len(overlap) / len(source_support) if source_support else None,
        "support_overlap_fraction_target": len(overlap) / len(target_support) if target_support else None,
    }


def _support_rows(matched_df: pd.DataFrame, source_support: set[str] | None) -> pd.DataFrame:
    if not source_support or "roi" not in matched_df.columns:
        return matched_df
    return matched_df[matched_df["roi"].astype(str).isin(source_support)]


def compute_direction_stability(matched_df: pd.DataFrame, source_support=None) -> dict:
    if matched_df is None or matched_df.empty:
        return {"direction_retention_all": None, "direction_retention_source_support": None, "sign_flip_fraction_all": None, "sign_flip_fraction_source_support": None}
    support_df = _support_rows(matched_df, source_support)
    return {
        "direction_retention_all": direction_retention(matched_df["source_coef"], matched_df["target_coef"]),
        "direction_retention_source_support": direction_retention(support_df["source_coef"], support_df["target_coef"]),
        "sign_flip_fraction_all": sign_flip_fraction(matched_df["source_coef"], matched_df["target_coef"]),
        "sign_flip_fraction_source_support": sign_flip_fraction(support_df["source_coef"], support_df["target_coef"]),
    }


def compute_magnitude_stability(matched_df: pd.DataFrame, source_support=None) -> dict:
    if matched_df is None or matched_df.empty:
        return {"magnitude_drift_all": None, "magnitude_drift_source_support": None}
    support_df = _support_rows(matched_df, source_support)
    return {
        "magnitude_drift_all": magnitude_drift(matched_df["source_t"], matched_df["target_t"]),
        "magnitude_drift_source_support": magnitude_drift(support_df["source_t"], support_df["target_t"]),
    }


def compute_group_stability(matched_group_df: pd.DataFrame, top_k: int = 10) -> dict:
    if matched_group_df is None or matched_group_df.empty:
        return {"group_evidence_spearman": None, "group_evidence_pearson": None, "group_support_jaccard": None, "group_direction_retention": None, "n_matched_groups": 0}
    return {
        "group_evidence_spearman": safe_spearman(matched_group_df["source_t"], matched_group_df["target_t"]),
        "group_evidence_pearson": safe_pearson(matched_group_df["source_t"], matched_group_df["target_t"]),
        "group_support_jaccard": rank_overlap_topk(matched_group_df["source_t"], matched_group_df["target_t"], top_k),
        "group_direction_retention": direction_retention(matched_group_df["source_coef"], matched_group_df["target_coef"]),
        "n_matched_groups": int(len(matched_group_df)),
    }
