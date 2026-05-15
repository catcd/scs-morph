from __future__ import annotations

import numpy as np
import pandas as pd

from scs_morph.evaluation.pairs import EvaluationPair


def aggregate_roi_statistics(stats_df: pd.DataFrame, group_col: str, evidence_cols=("coef", "t")) -> pd.DataFrame:
    if stats_df.empty or group_col not in stats_df.columns:
        return pd.DataFrame()
    rows = []
    usable = stats_df.copy()
    usable[group_col] = usable[group_col].where(usable[group_col].notna(), None)
    usable = usable[usable[group_col].notna()]
    for group_id, group in usable.groupby(group_col, dropna=True):
        coef = pd.to_numeric(group.get("coef"), errors="coerce")
        t = pd.to_numeric(group.get("t"), errors="coerce")
        q = pd.to_numeric(group.get("q"), errors="coerce") if "q" in group.columns else pd.Series(np.nan, index=group.index)
        rows.append({
            "group_id": str(group_id),
            "n_rois": int(len(group)),
            "mean_coef": float(coef.mean()) if coef.notna().any() else np.nan,
            "median_coef": float(coef.median()) if coef.notna().any() else np.nan,
            "mean_t": float(t.mean()) if t.notna().any() else np.nan,
            "median_t": float(t.median()) if t.notna().any() else np.nan,
            "mean_abs_t": float(t.abs().mean()) if t.notna().any() else np.nan,
            "positive_fraction": float((coef > 0).mean()) if coef.notna().any() else np.nan,
            "negative_fraction": float((coef < 0).mean()) if coef.notna().any() else np.nan,
            "min_q": float(q.min()) if q.notna().any() else np.nan,
            "n_q_0_10": int((q <= 0.10).sum()) if q.notna().any() else 0,
            "support_rois": "; ".join(group.get("roi", pd.Series(dtype=str)).astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def choose_common_group_column(source_feature_space: str, target_feature_space: str, requested_strategy: str) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    if requested_strategy == "exact_roi":
        return None, warnings
    if requested_strategy == "network_7_like":
        return "network_7_like", warnings
    if requested_strategy == "lobe":
        return "lobe", warnings
    if requested_strategy == "group_summary":
        return "roi_group", warnings
    if requested_strategy == "no_direct_roi_match":
        return None, warnings
    warnings.append(f"Unknown group strategy {requested_strategy}")
    return None, warnings


def _fallback_group_col(stats: pd.DataFrame, preferred: str | None) -> str | None:
    candidates = [preferred, "network", "roi_group", "anatomical_group", "lobe", "network_7_like"]
    for candidate in candidates:
        if candidate and candidate in stats.columns and stats[candidate].notna().any():
            return candidate
    return None


def aggregate_for_pair(source_stats: pd.DataFrame, target_stats: pd.DataFrame, pair: EvaluationPair) -> pd.DataFrame:
    preferred, _ = choose_common_group_column(pair.source_feature_space, pair.target_feature_space, pair.roi_match_strategy)
    source_col = _fallback_group_col(source_stats, preferred)
    target_col = _fallback_group_col(target_stats, preferred)
    if source_col is None or target_col is None:
        return pd.DataFrame()
    source = aggregate_roi_statistics(source_stats, source_col)
    target = aggregate_roi_statistics(target_stats, target_col)
    if source.empty or target.empty:
        return pd.DataFrame()
    merged = source.merge(target, on="group_id", suffixes=("_source", "_target"))
    return merged.rename(columns={
        "mean_coef_source": "source_coef",
        "mean_coef_target": "target_coef",
        "mean_t_source": "source_t",
        "mean_t_target": "target_t",
    })


def group_support_set(group_stats: pd.DataFrame, top_k: int = 10) -> set[str]:
    if group_stats.empty or "mean_t" not in group_stats.columns:
        return set()
    ranked = group_stats.assign(abs_t=pd.to_numeric(group_stats["mean_t"], errors="coerce").abs())
    return set(ranked.dropna(subset=["abs_t"]).sort_values("abs_t", ascending=False)["group_id"].head(top_k).astype(str))

