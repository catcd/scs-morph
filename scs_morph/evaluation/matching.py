from __future__ import annotations

import pandas as pd

from scs_morph.claims.family import ClaimFamily
from scs_morph.evaluation.aggregation import aggregate_for_pair
from scs_morph.evaluation.pairs import EvaluationPair


GROUP_COLS = ["hemisphere", "roi_group", "anatomical_group", "network", "network_7_like", "lobe"]


def match_roi_features(source_stats: pd.DataFrame, target_stats: pd.DataFrame, strategy: str = "exact_roi") -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if source_stats is None or target_stats is None or source_stats.empty or target_stats.empty:
        return pd.DataFrame(), ["Missing source or target ROI statistics."]
    if strategy == "no_direct_roi_match":
        return pd.DataFrame(), warnings
    if strategy != "exact_roi":
        pair = EvaluationPair(
            pair_id="temporary",
            experiment_id="temporary",
            experiment_type="atlas_sensitivity",
            source_dataset="",
            source_feature_space="",
            source_setting_id=None,
            source_claim_type="continuous_association",
            source_target=None,
            source_filter=None,
            target_dataset="",
            target_feature_space="",
            target_setting_id=None,
            target_claim_type="continuous_association",
            target_target=None,
            target_filter=None,
            roi_match_strategy=strategy,
        )
        return aggregate_for_pair(source_stats, target_stats, pair), warnings
    columns = ["roi", "coef", "t", "p", "q", *[c for c in GROUP_COLS if c in source_stats.columns]]
    source = source_stats[[c for c in columns if c in source_stats.columns]].copy()
    target = target_stats[[c for c in columns if c in target_stats.columns]].copy()
    matched = source.merge(target, on="roi", suffixes=("_source", "_target"))
    rename = {
        "coef_source": "source_coef",
        "coef_target": "target_coef",
        "t_source": "source_t",
        "t_target": "target_t",
        "p_source": "source_p",
        "p_target": "target_p",
        "q_source": "source_q",
        "q_target": "target_q",
    }
    return matched.rename(columns=rename), warnings


def get_support_set(stats_df: pd.DataFrame, top_k: int = 20, support_col: str = "t", q_threshold: float = 0.10) -> set[str]:
    if stats_df is None or stats_df.empty or "roi" not in stats_df.columns or support_col not in stats_df.columns:
        return set()
    stats = stats_df.copy()
    stats[support_col] = pd.to_numeric(stats[support_col], errors="coerce")
    stats = stats.dropna(subset=[support_col])
    if stats.empty:
        return set()
    significant = pd.DataFrame()
    if "q" in stats.columns:
        q = pd.to_numeric(stats["q"], errors="coerce")
        significant = stats[q <= q_threshold]
    ranked = significant if len(significant) >= top_k else stats
    ranked = ranked.assign(abs_value=ranked[support_col].abs()).sort_values("abs_value", ascending=False)
    return set(ranked["roi"].astype(str).head(top_k))


def match_claim_families(source_family: ClaimFamily, target_family: ClaimFamily, pair: EvaluationPair) -> dict:
    return {
        "source_roi_stats": None,
        "target_roi_stats": None,
        "matched_roi_stats": pd.DataFrame(),
        "source_support": set(),
        "target_support": set(),
        "warnings": [],
    }


def infer_pair_status(pair: EvaluationPair, source_family: ClaimFamily | None, target_family: ClaimFamily | None, matched: dict) -> str:
    if source_family is None or target_family is None:
        return "skipped"
    if matched.get("matched_roi_stats") is None:
        return "skipped"
    return "evaluated"

