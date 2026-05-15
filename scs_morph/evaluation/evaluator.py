from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs_morph.claims.family import ClaimFamily
from scs_morph.evaluation.aggregation import aggregate_for_pair
from scs_morph.evaluation.loaders import (
    find_claim_family,
    load_feature_matrix,
    load_roi_statistics_for_setting,
)
from scs_morph.evaluation.matching import get_support_set, match_roi_features
from scs_morph.evaluation.metrics import (
    compute_direction_stability,
    compute_evidence_stability,
    compute_group_stability,
    compute_magnitude_stability,
    compute_support_stability,
)
from scs_morph.evaluation.pairs import EvaluationPair, EvaluationResult
from scs_morph.evaluation.thresholds import assign_overall_label, assign_stability_label, assign_transfer_label
from scs_morph.evaluation.transfer import evaluate_claim_score_transfer


def _base_result(pair: EvaluationPair, source_family: ClaimFamily | None = None, target_family: ClaimFamily | None = None) -> EvaluationResult:
    return EvaluationResult(
        pair_id=pair.pair_id,
        experiment_id=pair.experiment_id,
        experiment_type=pair.experiment_type,
        source_claim_family_id=source_family.claim_family_id if source_family else None,
        target_claim_family_id=target_family.claim_family_id if target_family else None,
        source_setting_id=pair.source_setting_id,
        target_setting_id=pair.target_setting_id,
        source_dataset=pair.source_dataset,
        target_dataset=pair.target_dataset,
        source_feature_space=pair.source_feature_space,
        target_feature_space=pair.target_feature_space,
        source_claim_type=pair.source_claim_type,
        target_claim_type=pair.target_claim_type,
        source_target=pair.source_target,
        target_target=pair.target_target,
        roi_match_strategy=pair.roi_match_strategy,
    )


def _skip(pair: EvaluationPair, reason: str, source_family: ClaimFamily | None = None, target_family: ClaimFamily | None = None) -> EvaluationResult:
    result = _base_result(pair, source_family, target_family)
    result.status = "skipped"
    result.warnings.append(reason)
    result.overall_label = "insufficient"
    return result


def evaluate_pair(pair: EvaluationPair, claim_families: list[ClaimFamily], paths_config: dict, thresholds_config: dict) -> EvaluationResult:
    source_family = find_claim_family(claim_families, pair.source_setting_id, pair.source_dataset, pair.source_feature_space, pair.source_claim_type, pair.source_target)
    target_family = find_claim_family(claim_families, pair.target_setting_id, pair.target_dataset, pair.target_feature_space, pair.target_claim_type, pair.target_target)
    if source_family is None:
        return _skip(pair, "source claim family missing")
    if target_family is None:
        return _skip(pair, "target claim family missing", source_family, None)

    per_setting = Path(paths_config.get("processed_dir", "data/processed")) / "claim_cards" / "per_setting"
    source_setting_id = pair.source_setting_id or source_family.setting_id
    target_setting_id = pair.target_setting_id or target_family.setting_id
    source_stats = load_roi_statistics_for_setting(source_setting_id, str(per_setting))
    target_stats = load_roi_statistics_for_setting(target_setting_id, str(per_setting))
    if source_stats is None:
        return _skip(pair, f"source ROI statistics missing for {source_setting_id}", source_family, target_family)
    if target_stats is None:
        return _skip(pair, f"target ROI statistics missing for {target_setting_id}", source_family, target_family)

    warnings: list[str] = []
    if pair.roi_match_strategy == "no_direct_roi_match":
        matched = pd.DataFrame()
        group_matched = pd.DataFrame()
    elif pair.roi_match_strategy == "exact_roi":
        matched, match_warnings = match_roi_features(source_stats, target_stats, "exact_roi")
        warnings.extend(match_warnings)
        group_matched = pd.DataFrame()
    else:
        group_matched = aggregate_for_pair(source_stats, target_stats, pair)
        matched = group_matched.copy()
    if pair.roi_match_strategy != "no_direct_roi_match" and matched.empty:
        return _skip(pair, f"no matched features using {pair.roi_match_strategy}", source_family, target_family)

    result = _base_result(pair, source_family, target_family)
    result.n_source_features = int(len(source_stats))
    result.n_target_features = int(len(target_stats))
    result.n_matched_features = int(len(matched))

    source_support = get_support_set(source_stats, pair.top_k)
    target_support = get_support_set(target_stats, pair.top_k)
    result.n_source_support = len(source_support)
    result.n_target_support = len(target_support)

    if pair.roi_match_strategy == "no_direct_roi_match":
        result.evidence_metrics = {"n_matched": 0}
        result.support_metrics = {"support_jaccard": None}
        result.direction_metrics = {}
        result.magnitude_metrics = {}
        result.group_metrics = {}
    elif pair.roi_match_strategy == "exact_roi":
        result.evidence_metrics = compute_evidence_stability(matched, "source_t", "target_t", pair.top_k)
        result.support_metrics = compute_support_stability(source_stats, target_stats, matched, pair.top_k)
        result.direction_metrics = compute_direction_stability(matched, source_support)
        result.magnitude_metrics = compute_magnitude_stability(matched, source_support)
        result.group_metrics = {}
    else:
        result.group_metrics = compute_group_stability(group_matched, min(pair.top_k, 10))
        result.evidence_metrics = {
            "evidence_spearman": result.group_metrics.get("group_evidence_spearman"),
            "evidence_pearson": result.group_metrics.get("group_evidence_pearson"),
            "rank_overlap_topk": result.group_metrics.get("group_support_jaccard"),
            "weighted_jaccard_abs": None,
            "n_matched": result.group_metrics.get("n_matched_groups", 0),
        }
        result.support_metrics = {"support_jaccard": result.group_metrics.get("group_support_jaccard")}
        result.direction_metrics = {"direction_retention_source_support": result.group_metrics.get("group_direction_retention")}
        result.magnitude_metrics = compute_magnitude_stability(group_matched, None)

    transfer_metrics = {}
    if pair.transfer_target and pair.transfer_target_type not in {None, "none"}:
        if pair.roi_match_strategy == "exact_roi":
            try:
                target_X, target_meta = load_feature_matrix(pair.target_dataset, pair.target_feature_space, str(Path(paths_config.get("processed_dir", "data/processed")) / "feature_matrices"))
                matched_features = matched["roi"].astype(str).tolist() if "roi" in matched.columns else []
                transfer_metrics = evaluate_claim_score_transfer(pair, source_stats, target_X, target_meta, matched_features)
            except Exception as exc:
                transfer_metrics = {"error": str(exc)}
        else:
            transfer_metrics = {"error": "claim-score transfer requires exact matched target feature columns"}
    result.transfer_metrics = transfer_metrics

    stability_basis = {**result.evidence_metrics, **result.support_metrics, **result.direction_metrics}
    if pair.roi_match_strategy not in {"exact_roi", "no_direct_roi_match"}:
        stability_basis = result.group_metrics
    result.stability_label = assign_stability_label(stability_basis, thresholds_config)
    result.transfer_label = assign_transfer_label(result.transfer_metrics, pair.transfer_target_type, thresholds_config)
    result.overall_label = assign_overall_label(result.stability_label, result.transfer_label)
    result.warnings.extend(warnings)
    result.status = "evaluated"
    return result


def evaluate_pairs(pairs: list[EvaluationPair], claim_families: list[ClaimFamily], paths_config: dict, thresholds_config: dict) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for pair in pairs:
        try:
            results.append(evaluate_pair(pair, claim_families, paths_config, thresholds_config))
        except Exception as exc:
            result = _base_result(pair)
            result.status = "error"
            result.warnings.append(str(exc))
            results.append(result)
    return results

