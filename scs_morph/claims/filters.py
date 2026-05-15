from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ClaimCriteria:
    min_n: int = 30
    top_k: int = 20
    max_atomic_claims: int = 30
    atomic_q_threshold: float | None = 0.10
    atomic_p_threshold: float | None = None
    atomic_min_abs_t: float = 2.0
    atomic_min_abs_coef: float | None = None
    composite_min_support_rois: int = 3
    composite_min_mean_abs_t: float = 2.0
    group_min_rois: int = 3
    group_min_support_rois: int = 2
    group_min_fraction_support: float = 0.20
    group_min_mean_abs_t: float = 1.5
    ranking_top_k: int = 20
    ranking_min_valid_rois: int = 5
    direction_min_fraction_same_sign: float = 0.70
    direction_min_support_rois: int = 5
    latent_min_explained_variance: float = 0.05
    latent_top_k: int = 20
    latent_min_metadata_abs_corr: float = 0.20
    latent_min_metadata_abs_effect: float = 0.30
    normative_z_threshold: float = 1.5
    normative_min_support_rois: int = 3
    normative_min_mean_abs_z: float = 1.0
    normative_max_atomic_claims: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClaimCriteria":
        base = cls()
        if not data:
            return base
        values = {**base.__dict__, **data}
        return cls(**{key: values[key] for key in base.__dict__})


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def roi_passes_atomic_criteria(row, criteria: ClaimCriteria) -> bool:
    q = row.get("q")
    p = row.get("p")
    t = row.get("t")
    coef = row.get("coef")
    has_sig = False
    if criteria.atomic_q_threshold is not None and _finite(q):
        has_sig = float(q) <= criteria.atomic_q_threshold
    elif criteria.atomic_p_threshold is not None and _finite(p):
        has_sig = float(p) <= criteria.atomic_p_threshold
    if not has_sig:
        return False
    if _finite(t) and abs(float(t)) < criteria.atomic_min_abs_t:
        return False
    if criteria.atomic_min_abs_coef is not None:
        if not _finite(coef) or abs(float(coef)) < criteria.atomic_min_abs_coef:
            return False
    return True


def filter_atomic_rois(results_df: pd.DataFrame, criteria: ClaimCriteria) -> pd.DataFrame:
    passed = results_df[results_df.apply(lambda row: roi_passes_atomic_criteria(row, criteria), axis=1)].copy()
    if passed.empty:
        return passed
    passed["_abs_t"] = pd.to_numeric(passed["t"], errors="coerce").abs()
    passed["_abs_coef"] = pd.to_numeric(passed["coef"], errors="coerce").abs()
    passed = passed.sort_values(["_abs_t", "_abs_coef"], ascending=False)
    return passed.drop(columns=["_abs_t", "_abs_coef"]).head(criteria.max_atomic_claims)


def composite_passes_criteria(
    results_df: pd.DataFrame,
    support_regions: list[str],
    criteria: ClaimCriteria,
) -> tuple[bool, dict]:
    support = results_df[results_df["roi"].isin(support_regions)]
    mean_abs_t = float(pd.to_numeric(support["t"], errors="coerce").abs().mean()) if len(support) else np.nan
    n = int(pd.to_numeric(results_df.get("n", pd.Series(dtype=float)), errors="coerce").max() or 0)
    details = {
        "n_support_rois": int(len(support_regions)),
        "mean_abs_t_support": mean_abs_t,
        "n": n,
        "required_support_rois": criteria.composite_min_support_rois,
        "required_mean_abs_t": criteria.composite_min_mean_abs_t,
        "required_n": criteria.min_n,
    }
    passed = (
        len(support_regions) >= criteria.composite_min_support_rois
        and _finite(mean_abs_t)
        and mean_abs_t >= criteria.composite_min_mean_abs_t
        and n >= criteria.min_n
    )
    return passed, details


def ranking_passes_criteria(results_df: pd.DataFrame, criteria: ClaimCriteria) -> tuple[bool, dict]:
    valid = int(pd.to_numeric(results_df["t"], errors="coerce").notna().sum()) if "t" in results_df else 0
    details = {"n_valid_rois": valid, "required_valid_rois": criteria.ranking_min_valid_rois}
    return valid >= criteria.ranking_min_valid_rois, details


def direction_summary_passes_criteria(
    support_effects: dict[str, float],
    criteria: ClaimCriteria,
) -> tuple[bool, dict]:
    values = np.asarray([v for v in support_effects.values() if _finite(v)], dtype=float)
    if values.size == 0:
        return False, {"n_support_rois": 0, "fraction_same_sign": 0.0}
    frac_pos = float((values > 0).mean())
    frac_neg = float((values < 0).mean())
    frac = max(frac_pos, frac_neg)
    details = {
        "n_support_rois": int(values.size),
        "fraction_same_sign": frac,
        "positive_fraction": frac_pos,
        "negative_fraction": frac_neg,
    }
    return values.size >= criteria.direction_min_support_rois and frac >= criteria.direction_min_fraction_same_sign, details


def group_passes_criteria(group_summary_row, criteria: ClaimCriteria) -> bool:
    return (
        int(group_summary_row.get("n_group_rois", 0)) >= criteria.group_min_rois
        and int(group_summary_row.get("n_support_rois", 0)) >= criteria.group_min_support_rois
        and float(group_summary_row.get("fraction_support", 0.0)) >= criteria.group_min_fraction_support
        and abs(float(group_summary_row.get("mean_abs_t", 0.0))) >= criteria.group_min_mean_abs_t
    )


def latent_axis_passes_criteria(axis_summary, criteria: ClaimCriteria) -> bool:
    return float(axis_summary.get("explained_variance_ratio", 0.0)) >= criteria.latent_min_explained_variance


def normative_claim_passes_criteria(normative_summary, criteria: ClaimCriteria) -> bool:
    return (
        int(normative_summary.get("n_support_rois", 0)) >= criteria.normative_min_support_rois
        and float(normative_summary.get("mean_abs_z", 0.0)) >= criteria.normative_min_mean_abs_z
    )

