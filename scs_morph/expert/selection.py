from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scs_morph.expert.validation import has_bad_claim_text, normalize_text


LABEL_RANK = {
    "stable_and_transferable": 4,
    "stable_not_transferable": 3,
    "transferable_but_drifted": 2,
    "fragile": 1,
    "insufficient": 0,
}
TRANSFER_RANK = {"strong": 3, "moderate": 2, "weak": 1, "not_applicable": 0, "insufficient": 0}


def load_review_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = config["paths"]
    return (
        pd.read_csv(paths["claim_cards_summary"]),
        pd.read_csv(paths["claim_family_summary"]),
        pd.read_csv(paths["evaluation_results"]),
    )


def _summarize_eval(group: pd.DataFrame) -> dict[str, Any]:
    def num_col(name: str) -> pd.Series:
        if name not in group.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(group[name], errors="coerce")

    labels = group["overall_label"].dropna().astype(str).tolist()
    stability = group["stability_label"].dropna().astype(str).tolist()
    transfer = group["transfer_label"].dropna().astype(str).tolist()
    evidence = num_col("evidence_metrics.evidence_spearman")
    support = num_col("support_metrics.support_jaccard")
    auc = num_col("transfer_metrics.auc_oriented")
    corr = num_col("transfer_metrics.spearman_corr")
    best_label = max(labels, key=lambda x: LABEL_RANK.get(x, -1)) if labels else "insufficient"
    worst_label = min(labels, key=lambda x: LABEL_RANK.get(x, 99)) if labels else "insufficient"
    best_transfer = max(transfer, key=lambda x: TRANSFER_RANK.get(x, -1)) if transfer else "not_applicable"
    max_evidence = evidence.max()
    min_support = support.min()
    high_fragile = bool(((evidence >= 0.70) & (support < 0.20)).any() or ((group["overall_label"] == "fragile") & (evidence >= 0.70)).any())
    return {
        "n_evaluation_pairs": int(len(group)),
        "best_overall_label": best_label,
        "worst_overall_label": worst_label,
        "any_stable_and_transferable": bool((group["overall_label"] == "stable_and_transferable").any()),
        "any_fragile": bool((group["overall_label"] == "fragile").any() or (group["stability_label"] == "fragile").any()),
        "max_evidence_spearman": float(max_evidence) if pd.notna(max_evidence) else np.nan,
        "median_evidence_spearman": float(evidence.median()) if evidence.notna().any() else np.nan,
        "min_support_jaccard": float(min_support) if pd.notna(min_support) else np.nan,
        "median_support_jaccard": float(support.median()) if support.notna().any() else np.nan,
        "best_transfer_label": best_transfer,
        "max_auc_oriented": float(auc.max()) if auc.notna().any() else np.nan,
        "max_abs_transfer_corr": float(corr.abs().max()) if corr.notna().any() else np.nan,
        "related_experiment_ids": "; ".join(sorted(group["experiment_id"].dropna().astype(str).unique())),
        "high_concordance_fragile_support": high_fragile,
        "stability_label": Counter(stability).most_common(1)[0][0] if stability else "insufficient",
        "transfer_label": best_transfer,
        "overall_label": best_label,
        "evidence_spearman": float(evidence.max()) if evidence.notna().any() else np.nan,
        "support_jaccard": float(support.min()) if support.notna().any() else np.nan,
        "direction_retention_source_support": num_col("direction_metrics.direction_retention_source_support").median(),
        "transfer_auc_oriented": float(auc.max()) if auc.notna().any() else np.nan,
        "transfer_spearman_corr": float(corr.iloc[corr.abs().argmax()]) if corr.notna().any() else np.nan,
    }


def enrich_claims_with_evaluation(claims_df: pd.DataFrame, evaluation_df: pd.DataFrame) -> pd.DataFrame:
    claims = claims_df.copy()
    summaries = []
    for claim_family_id, claim_group in claims.groupby("claim_family_id", dropna=False):
        related = evaluation_df[
            (evaluation_df.get("source_claim_family_id") == claim_family_id)
            | (evaluation_df.get("target_claim_family_id") == claim_family_id)
        ]
        fallback_cols = {"source_dataset", "target_dataset", "source_feature_space", "target_feature_space", "source_claim_type", "target_claim_type"}
        if related.empty and fallback_cols.issubset(evaluation_df.columns):
            related = evaluation_df[
                (evaluation_df.get("source_dataset").eq(claim_group["dataset"].iloc[0]) | evaluation_df.get("target_dataset").eq(claim_group["dataset"].iloc[0]))
                & (evaluation_df.get("source_feature_space").eq(claim_group["feature_space"].iloc[0]) | evaluation_df.get("target_feature_space").eq(claim_group["feature_space"].iloc[0]))
                & (evaluation_df.get("source_claim_type").eq(claim_group["claim_type"].iloc[0]) | evaluation_df.get("target_claim_type").eq(claim_group["claim_type"].iloc[0]))
            ]
        summary = _summarize_eval(related) if not related.empty else _summarize_eval(pd.DataFrame(columns=evaluation_df.columns))
        summary["claim_family_id"] = claim_family_id
        summaries.append(summary)
    enriched = claims.merge(pd.DataFrame(summaries), on="claim_family_id", how="left")
    strata = enriched.apply(assign_review_strata, axis=1, result_type="expand")
    enriched = pd.concat([enriched, strata], axis=1)
    enriched["dedup_key"] = enriched.apply(make_claim_dedup_key, axis=1)
    return enriched


def validate_claim_for_review(row) -> bool:
    if has_bad_claim_text(row.get("claim_text")):
        return False
    if str(row.get("status", "")).lower() == "error":
        return False
    return True


def make_claim_dedup_key(row) -> str:
    region = row.get("roi") or row.get("roi_group") or row.get("network") or row.get("lobe") or ""
    parts = [
        row.get("claim_level", ""),
        row.get("claim_type", ""),
        row.get("target", ""),
        row.get("feature_family", ""),
        region,
        row.get("direction", ""),
        normalize_text(str(row.get("claim_text", "")))[:160],
    ]
    return "|".join(normalize_text(str(p)) for p in parts)


def assign_review_strata(row) -> dict:
    claim_type = str(row.get("claim_type", "")).lower()
    target = str(row.get("target", "")).lower()
    claim_level = str(row.get("claim_level", "")).lower()
    experiments = str(row.get("related_experiment_ids", "")).lower()
    if "latent" in claim_level or "normative" in claim_level:
        target_type = "latent_or_normative"
    elif "diagnosis" in claim_type or "cdr" in claim_type or "diagnosis" in target:
        target_type = "diagnosis_or_cdr"
    elif target == "mmse":
        target_type = "mmse"
    elif target == "age":
        target_type = "age"
    elif any(x in claim_type for x in ["acquisition", "demographic"]) or any(x in experiments for x in ["pipeline", "atlas", "fastsurfer"]):
        target_type = "acquisition_or_demographic_or_pipeline_or_atlas"
    else:
        target_type = "other"
    if bool(row.get("high_concordance_fragile_support", False)):
        eval_cat = "high_concordance_fragile_support"
    elif bool(row.get("any_stable_and_transferable", False)) or row.get("best_overall_label") == "stable_and_transferable":
        eval_cat = "stable_and_transferable"
    elif row.get("best_overall_label") == "stable_not_transferable":
        eval_cat = "stable_not_transferable"
    elif bool(row.get("any_fragile", False)) or row.get("worst_overall_label") == "fragile":
        eval_cat = "fragile"
    elif row.get("best_overall_label") == "insufficient":
        eval_cat = "insufficient"
    elif row.get("stability_label") == "moderate":
        eval_cat = "moderate"
    else:
        eval_cat = "other"
    family = str(row.get("feature_family", "")).lower()
    fs = str(row.get("feature_space", "")).lower()
    if "volume" in family or "volume" in fs:
        feature_group = "volume"
    elif "thickness" in family or "thickness" in fs:
        feature_group = "thickness"
    else:
        feature_group = "other"
    return {
        "target_type": target_type,
        "evaluation_category": eval_cat,
        "feature_family_group": feature_group,
        "dataset_group": row.get("dataset"),
    }


def _candidate_pool(claims_df: pd.DataFrame, config: dict, require_passed: bool) -> pd.DataFrame:
    excluded = set(config["selection"].get("exclude_claim_levels", []))
    pool = claims_df[claims_df.apply(validate_claim_for_review, axis=1)].copy()
    pool = pool[~pool["claim_level"].isin(excluded)]
    if require_passed and "criteria_passed" in pool:
        pool = pool[pool["criteria_passed"].astype(str).str.lower().isin(["true", "1"])]
    pool["selection_score"] = (
        pool["evaluation_category"].map({"stable_and_transferable": 5, "high_concordance_fragile_support": 5, "fragile": 4, "stable_not_transferable": 4, "moderate": 3, "insufficient": 2}).fillna(1)
        + pool["claim_level"].map({"composite": 3, "atomic_roi": 2, "group_summary": 2, "ranking": 1, "latent_axis": 2, "latent_axis_association": 2, "normative_deviation": 2, "normative_atomic_roi": 2}).fillna(1)
    )
    return pool.sample(frac=1, random_state=config["selection"].get("random_seed", 42)).sort_values("selection_score", ascending=False)


def _take_unique(pool: pd.DataFrame, selected: list[pd.Series], n: int, reason: str, used: set[str]) -> None:
    for _, row in pool.iterrows():
        if len(selected) >= n:
            return
        key = row["dedup_key"]
        if key in used:
            continue
        item = row.copy()
        item["selection_reason"] = reason
        item["review_priority"] = len(selected) + 1
        selected.append(item)
        used.add(key)


def select_main_expert_claims(claims_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    n = int(config["selection"]["main_expert_n"])
    pool = _candidate_pool(claims_df, config, require_passed=config["selection"].get("require_criteria_passed_for_main", True))
    selected: list[pd.Series] = []
    used: set[str] = set()
    for level, target_n in config["main_expert_composition"]["claim_level_targets"].items():
        _take_unique(pool[pool["claim_level"] == level], selected, min(n, len(selected) + int(target_n)), f"claim_level:{level}", used)
    for column, minimums in [
        ("target_type", config["main_expert_composition"]["target_type_minimums"]),
        ("evaluation_category", config["main_expert_composition"]["evaluation_category_minimums"]),
        ("feature_family_group", config["main_expert_composition"]["feature_family_minimums"]),
        ("dataset", config["main_expert_composition"]["dataset_minimums"]),
    ]:
        current = pd.DataFrame(selected)
        for value, minimum in minimums.items():
            have = int((current[column] == value).sum()) if not current.empty and column in current else 0
            _take_unique(pool[pool[column] == value], selected, min(n, len(selected) + max(0, int(minimum) - have)), f"{column}:{value}", used)
    _take_unique(pool, selected, n, "diverse_fill", used)
    out = pd.DataFrame(selected).head(n).copy()
    out["review_set"] = "main_expert"
    return out


def select_llm_extended_claims(claims_df: pd.DataFrame, main_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    n = int(config["selection"]["llm_extended_n"])
    pool = _candidate_pool(claims_df, config, require_passed=False)
    selected = [row.copy() for _, row in main_df.iterrows()]
    used = set(main_df["dedup_key"])
    priority = pool[
        (pool["evaluation_category"].isin(["fragile", "high_concordance_fragile_support"]))
        | (pool["target_type"] == "latent_or_normative")
        | (pool["related_experiment_ids"].astype(str).str.contains("ATLAS|FASTSURFER|SPM", case=False, regex=True, na=False))
    ]
    _take_unique(priority, selected, n, "llm_priority", used)
    _take_unique(pool, selected, n, "llm_diverse_fill", used)
    out = pd.DataFrame(selected).head(n).copy()
    out["review_set"] = "llm_extended"
    return out


def select_reserve_claims(claims_df: pd.DataFrame, main_df: pd.DataFrame, llm_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    n = int(config["selection"]["reserve_n"])
    pool = _candidate_pool(claims_df, config, require_passed=False)
    used = set(main_df["dedup_key"]) | set(llm_df["dedup_key"])
    selected: list[pd.Series] = []
    _take_unique(pool[~pool["dedup_key"].isin(used)], selected, n, "reserve_nonoverlap", set())
    if len(selected) < n:
        _take_unique(pool[~pool["dedup_key"].isin(set(main_df["dedup_key"]))], selected, n, "reserve_llm_overlap", set(x["dedup_key"] for x in selected))
    out = pd.DataFrame(selected).head(n).copy()
    out["review_set"] = "reserve"
    out["overlaps_llm_extended"] = out["dedup_key"].isin(set(llm_df["dedup_key"]))
    return out


def _counts(df: pd.DataFrame, column: str) -> dict:
    return df[column].fillna("NA").value_counts().to_dict() if not df.empty and column in df else {}


def create_selection_report(main_df: pd.DataFrame, llm_df: pd.DataFrame, reserve_df: pd.DataFrame, all_candidates_df: pd.DataFrame) -> dict:
    return {
        "n_candidates": int(len(all_candidates_df)),
        "n_selected_main": int(len(main_df)),
        "n_selected_llm": int(len(llm_df)),
        "n_selected_reserve": int(len(reserve_df)),
        "main_by_claim_level": _counts(main_df, "claim_level"),
        "main_by_claim_type": _counts(main_df, "claim_type"),
        "main_by_dataset": _counts(main_df, "dataset"),
        "main_by_feature_space": _counts(main_df, "feature_space"),
        "main_by_target_type": _counts(main_df, "target_type"),
        "main_by_evaluation_category": _counts(main_df, "evaluation_category"),
        "llm_by_claim_level": _counts(llm_df, "claim_level"),
        "reserve_by_claim_level": _counts(reserve_df, "claim_level"),
    }


def save_selection_report(report: dict, output_csv: str, output_md: str) -> None:
    rows = []
    for key, value in report.items():
        rows.append({"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, dict) else value})
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    lines = ["# Expert Review Selection Report\n"]
    for row in rows:
        lines.append(f"- **{row['metric']}**: {row['value']}\n")
    Path(output_md).write_text("\n".join(lines), encoding="utf-8")
