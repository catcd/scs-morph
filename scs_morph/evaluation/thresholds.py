from __future__ import annotations


def assign_stability_label(metrics: dict, thresholds: dict) -> str:
    n = metrics.get("n_matched") or metrics.get("n_matched_groups") or 0
    if n < 5:
        return "insufficient"
    stable = thresholds.get("stability_thresholds", {}).get("stable", {})
    moderate = thresholds.get("stability_thresholds", {}).get("moderate", {})
    spearman = metrics.get("evidence_spearman", metrics.get("group_evidence_spearman"))
    support = metrics.get("support_jaccard", metrics.get("group_support_jaccard"))
    direction = metrics.get("direction_retention_source_support", metrics.get("group_direction_retention"))
    if _meets(spearman, stable.get("evidence_spearman_min")) and _meets(support, stable.get("support_jaccard_min")) and _meets(direction, stable.get("direction_retention_min")):
        return "stable"
    if _meets(spearman, moderate.get("evidence_spearman_min")) and _meets(support, moderate.get("support_jaccard_min")) and _meets(direction, moderate.get("direction_retention_min")):
        return "moderate"
    return "fragile"


def _meets(value, threshold) -> bool:
    if threshold is None:
        return True
    return value is not None and value >= threshold


def assign_transfer_label(metrics: dict, target_type: str | None, thresholds: dict) -> str:
    if target_type in {None, "none"}:
        return "not_applicable"
    if not metrics or metrics.get("n", 0) < 5:
        return "insufficient"
    config = thresholds.get("transfer_thresholds", {}).get(str(target_type), {})
    if target_type == "binary":
        auc = metrics.get("auc_oriented")
        d = abs(metrics.get("cohens_d")) if metrics.get("cohens_d") is not None else None
        strong = config.get("strong", {})
        moderate = config.get("moderate", {})
        if _meets(auc, strong.get("auc_oriented_min")) and _meets(d, strong.get("abs_cohens_d_min")):
            return "strong"
        if _meets(auc, moderate.get("auc_oriented_min")) and _meets(d, moderate.get("abs_cohens_d_min")):
            return "moderate"
        return "weak"
    if target_type == "continuous":
        corr = metrics.get("spearman_corr")
        abs_corr = abs(corr) if corr is not None else None
        if _meets(abs_corr, config.get("strong", {}).get("abs_spearman_min")):
            return "strong"
        if _meets(abs_corr, config.get("moderate", {}).get("abs_spearman_min")):
            return "moderate"
        return "weak"
    return "not_applicable"


def assign_overall_label(stability_label: str, transfer_label: str) -> str:
    if stability_label == "insufficient":
        return "insufficient"
    transfer_good = transfer_label in {"strong", "moderate"}
    if stability_label == "stable" and transfer_good:
        return "stable_and_transferable"
    if stability_label == "stable":
        return "stable_not_transferable"
    if stability_label in {"moderate", "fragile"} and transfer_good:
        return "transferable_but_drifted"
    if stability_label == "fragile":
        return "fragile"
    return "insufficient"

