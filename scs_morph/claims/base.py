from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha1
from typing import Any

import numpy as np
import pandas as pd

from scs_morph.analysis.design_matrix import prepare_analysis_table
from scs_morph.analysis.settings import AnalysisSetting, setting_to_dict, source_setting_from_analysis
from scs_morph.claims.cards import ClaimCard
from scs_morph.claims.family import ClaimFamily
from scs_morph.claims.filters import (
    ClaimCriteria,
    composite_passes_criteria,
    direction_summary_passes_criteria,
    filter_atomic_rois,
    group_passes_criteria,
    ranking_passes_criteria,
)
from scs_morph.claims.grouping import (
    annotate_roi_statistics,
    format_group_label,
    make_group_label,
    make_region_summary,
    summarize_roi_groups,
)
from scs_morph.claims.statistics import add_fdr, fit_roiwise_ols
from scs_morph.claims.support import compute_support_metrics, select_support_regions
from scs_morph.claims.text_generation import (
    composite_direction,
    covariate_text,
    direction_from_effect,
    lower_mmse_interpretation,
    render_template,
    target_label_for_claim,
    validate_claim_text,
)


def _family_id(setting: AnalysisSetting) -> str:
    digest = sha1(f"{setting.setting_id}:{setting.dataset}:{setting.feature_space}:{setting.claim_type}".encode()).hexdigest()[:10]
    return f"{setting.setting_id}_{digest}"


def _feature_family_label(setting: AnalysisSetting, labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    return labels.get(setting.feature_space) or labels.get(str(setting.feature_family)) or str(setting.feature_family or "morphometry")


def _row_float(row, key: str) -> float | None:
    value = row.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _claim_id(family_id: str, level: str, suffix: str) -> str:
    safe = str(suffix).replace("Yale code-group ", "yale_").replace(" ", "_").replace("/", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})
    safe = safe.strip("_") or "unclassified"
    return f"{family_id}_{level}_{safe}"


def _finalize_claim(card: ClaimCard) -> ClaimCard:
    warnings = validate_claim_text(card.claim_text)
    if warnings:
        card.notes.extend(warnings)
        if card.criteria_passed and card.status == "supported":
            card.status = "weak_support"
    return card


class BaseClaimExtractor(ABC):
    """Base extractor that returns a full claim family."""

    def __init__(self, criteria: ClaimCriteria | None = None, feature_family_labels: dict[str, str] | None = None):
        self.criteria = criteria or ClaimCriteria()
        self.feature_family_labels = feature_family_labels or {}

    @abstractmethod
    def extract_family(self, X: pd.DataFrame, metadata: pd.DataFrame, setting: AnalysisSetting) -> tuple[ClaimFamily, pd.DataFrame | None]:
        """Extract a claim family and optional ROI statistics."""

    def extract(self, X: pd.DataFrame, metadata: pd.DataFrame, setting: AnalysisSetting) -> ClaimCard:
        family, _ = self.extract_family(X, metadata, setting)
        if not family.all_claims:
            raise ValueError(f"No claims generated for {setting.setting_id}")
        return family.all_claims[0]


class SupervisedClaimFamilyBuilder(BaseClaimExtractor):
    """Shared supervised claim-family generation."""

    def extract_family(self, X: pd.DataFrame, metadata: pd.DataFrame, setting: AnalysisSetting) -> tuple[ClaimFamily, pd.DataFrame]:
        criteria = ClaimCriteria.from_dict({**self.criteria.__dict__, **setting.criteria, "min_n": setting.min_n, "top_k": setting.top_k})
        X_clean, y, covariates, metadata_clean = prepare_analysis_table(X, metadata, setting.target or "", setting.covariates)
        results = fit_roiwise_ols(
            X_clean,
            y,
            covariates,
            standardize_features=setting.standardize_features,
            standardize_target=setting.standardize_target,
        )
        results = annotate_roi_statistics(add_fdr(results), setting.feature_space)
        family = self._build_family(results, metadata_clean, setting, criteria)
        return family, results

    def _base_card_kwargs(self, setting: AnalysisSetting, family_id: str, n_samples: int, n_features: int) -> dict[str, Any]:
        return {
            "claim_family_id": family_id,
            "source_setting": source_setting_from_analysis(setting).to_dict(),
            "dataset": setting.dataset,
            "cohort": setting.cohort,
            "feature_space": setting.feature_space,
            "feature_family": setting.feature_family,
            "pipeline": setting.pipeline,
            "atlas": setting.atlas,
            "target": setting.target,
            "covariates": list(setting.covariates),
            "n_samples": n_samples,
            "n_features": n_features,
        }

    def _build_family(
        self,
        results: pd.DataFrame,
        metadata: pd.DataFrame,
        setting: AnalysisSetting,
        criteria: ClaimCriteria,
    ) -> ClaimFamily:
        family_id = _family_id(setting)
        n_samples = int(len(metadata))
        n_features = int(len(results))
        family = ClaimFamily(
            claim_family_id=family_id,
            setting_id=setting.setting_id,
            setting_metadata=setting_to_dict(setting),
            extraction_summary={"n_samples": n_samples, "n_features": n_features},
        )
        base = self._base_card_kwargs(setting, family_id, n_samples, n_features)
        feature_label = _feature_family_label(setting, self.feature_family_labels)
        target_label = target_label_for_claim(setting.claim_type, setting.target, setting.positive_label, setting.negative_label)

        support_regions = select_support_regions(results, criteria.top_k, "t")
        support = results.set_index("roi").reindex(support_regions)
        support_effects = {str(k): float(v) for k, v in pd.to_numeric(support["coef"], errors="coerce").dropna().items()}
        support_metrics = compute_support_metrics(results, support_regions)
        direction = composite_direction(support_effects)
        region_summary = make_region_summary(support_regions, setting.feature_space)

        if setting.generate_composite_claim:
            passed, details = composite_passes_criteria(results, support_regions, criteria)
            template_id = f"{setting.claim_type}_composite" if f"{setting.claim_type}_composite" in self._template_ids() else "continuous_association_composite"
            text = render_template(template_id, positive_label=setting.positive_label, negative_label=setting.negative_label, direction=direction, feature_family_label=feature_label, region_summary=region_summary, covariate_text=covariate_text(setting.covariates), target_label=target_label)
            family.composite_claims.append(ClaimCard(
                claim_id=_claim_id(family_id, "composite", "main"),
                claim_level="composite",
                claim_type=setting.claim_type,
                direction=direction,
                effect_size=float(np.nanmean(list(support_effects.values()))) if support_effects else None,
                evidence_metric="mean_abs_t",
                evidence_value=support_metrics.get("mean_abs_t_support"),
                support_regions=support_regions,
                support_region_effects=support_effects,
                support_metrics=support_metrics,
                criteria_passed=passed,
                criteria_details=details,
                claim_text=text,
                short_claim_text=text,
                machine_readable_statement={"target": setting.target, "direction": direction, "support_regions": support_regions},
                status="supported" if passed else "weak_support",
                **base,
            ))

        if setting.generate_atomic_claims:
            atomic = filter_atomic_rois(results, criteria)
            for _, row in atomic.iterrows():
                roi = str(row["roi"])
                coef = _row_float(row, "coef")
                atom_direction = direction_from_effect(coef, feature_label)
                template_id = f"{setting.claim_type}_atomic_roi" if f"{setting.claim_type}_atomic_roi" in self._template_ids() else "continuous_association_atomic_roi"
                text = render_template(template_id, positive_label=setting.positive_label, negative_label=setting.negative_label, direction=atom_direction, roi=roi, feature_family_label=feature_label, covariate_text=covariate_text(setting.covariates), target_label=target_label)
                machine = {"roi": roi, "direction": atom_direction}
                if setting.claim_type == "mmse_association":
                    machine["lower_mmse_interpretation"] = lower_mmse_interpretation(coef)
                family.atomic_claims.append(ClaimCard(
                    claim_id=_claim_id(family_id, "atomic_roi", roi),
                    claim_level="atomic_roi",
                    claim_type=setting.claim_type,
                    roi=roi,
                    hemisphere=row.get("hemisphere"),
                    roi_group=row.get("roi_group"),
                    anatomical_group=row.get("anatomical_group"),
                    network=row.get("network"),
                    lobe=row.get("lobe"),
                    direction=atom_direction,
                    effect_size=coef,
                    evidence_metric="t",
                    evidence_value=_row_float(row, "t"),
                    coefficient=coef,
                    standard_error=_row_float(row, "se"),
                    t_value=_row_float(row, "t"),
                    p_value=_row_float(row, "p"),
                    q_value=_row_float(row, "q"),
                    criteria_passed=True,
                    criteria_details={"atomic_criteria": True},
                    claim_text=text,
                    short_claim_text=text,
                    machine_readable_statement=machine,
                    status="supported",
                    **base,
                ))

        group_summary = summarize_roi_groups(results, support_regions, setting.feature_space)
        if setting.generate_group_claims:
            for _, row in group_summary.iterrows():
                passed = group_passes_criteria(row, criteria)
                if not passed:
                    continue
                raw_group_label = make_group_label(row, setting.feature_space)
                formatted_group_label = format_group_label(raw_group_label, setting.feature_space)
                template_id = f"{setting.claim_type}_group_summary" if f"{setting.claim_type}_group_summary" in self._template_ids() else "continuous_association_group_summary"
                text = render_template(template_id, positive_label=setting.positive_label, negative_label=setting.negative_label, group_label=formatted_group_label, n_support_rois=int(row["n_support_rois"]), n_group_rois=int(row["n_group_rois"]), target_label=target_label)
                family.group_claims.append(_finalize_claim(ClaimCard(
                    claim_id=_claim_id(family_id, "group_summary", raw_group_label),
                    claim_level="group_summary",
                    claim_type="group_enrichment",
                    roi_group=row.get("roi_group"),
                    anatomical_group=row.get("anatomical_group"),
                    network=row.get("network"),
                    lobe=row.get("lobe"),
                    direction=row.get("direction"),
                    evidence_metric="mean_abs_t",
                    evidence_value=_row_float(row, "mean_abs_t"),
                    coefficient=_row_float(row, "mean_coef"),
                    support_regions=str(row.get("support_rois", "")).split("; ") if row.get("support_rois") else [],
                    support_metrics={"group_label": raw_group_label, "formatted_group_label": formatted_group_label},
                    criteria_passed=True,
                    criteria_details=row.to_dict(),
                    claim_text=text,
                    short_claim_text=text,
                    machine_readable_statement={
                        "group": raw_group_label,
                        "group_label": raw_group_label,
                        "formatted_group_label": formatted_group_label,
                        "network_7_like": row.get("network_7_like"),
                    },
                    status="supported",
                    **base,
                )))

        if setting.generate_ranking_claim:
            passed, details = ranking_passes_criteria(results, criteria)
            ranked_regions = select_support_regions(results, criteria.ranking_top_k, "t")
            template_id = f"{setting.claim_type}_ranking" if f"{setting.claim_type}_ranking" in self._template_ids() else "feature_ranking"
            text = render_template(template_id, positive_label=setting.positive_label, negative_label=setting.negative_label, top_regions=", ".join(ranked_regions[:10]), target_label=target_label)
            family.ranking_claims.append(ClaimCard(
                claim_id=_claim_id(family_id, "ranking", "top"),
                claim_level="ranking",
                claim_type="feature_ranking",
                evidence_metric="abs_t_rank",
                support_regions=ranked_regions,
                support_region_effects={roi: support_effects.get(roi) for roi in ranked_regions if roi in support_effects},
                criteria_passed=passed,
                criteria_details=details,
                claim_text=text,
                short_claim_text=text,
                machine_readable_statement={"ranked_regions": ranked_regions},
                status="supported" if passed else "weak_support",
                **base,
            ))

        if setting.generate_direction_claim:
            passed, details = direction_summary_passes_criteria(support_effects, criteria)
            frac_pos = details.get("positive_fraction", 0.0)
            frac_neg = details.get("negative_fraction", 0.0)
            if frac_neg >= criteria.direction_min_fraction_same_sign:
                template_id, fraction = "direction_summary_lower", frac_neg
            elif frac_pos >= criteria.direction_min_fraction_same_sign:
                template_id, fraction = "direction_summary_higher", frac_pos
            else:
                template_id, fraction = "direction_summary_mixed", max(frac_pos, frac_neg)
            text = render_template(template_id, fraction_pct=round(100 * fraction, 1), feature_family_label=feature_label, target_label=target_label)
            family.direction_claims.append(ClaimCard(
                claim_id=_claim_id(family_id, "direction_summary", "support"),
                claim_level="direction_summary",
                claim_type="direction_summary",
                direction=direction,
                evidence_metric="fraction_same_sign",
                evidence_value=float(details.get("fraction_same_sign", 0.0)),
                support_regions=support_regions,
                support_region_effects=support_effects,
                criteria_passed=passed,
                criteria_details=details,
                claim_text=text,
                short_claim_text=text,
                machine_readable_statement={"direction": direction},
                status="supported" if passed else "weak_support",
                **base,
            ))
        return family

    @staticmethod
    def _template_ids() -> set[str]:
        from scs_morph.claims.templates import CLAIM_TEMPLATES
        return set(CLAIM_TEMPLATES)


def get_extractor_for_claim_type(claim_type: str, criteria: ClaimCriteria | None = None, feature_family_labels: dict[str, str] | None = None) -> BaseClaimExtractor:
    from scs_morph.claims.continuous import ContinuousAssociationClaimExtractor
    from scs_morph.claims.contrast import DiagnosisContrastClaimExtractor
    from scs_morph.claims.normative import NormativeDeviationClaimExtractor
    from scs_morph.claims.unsupervised import LatentMorphometryClaimExtractor

    if claim_type in {"diagnosis_contrast", "cdr_impairment_contrast", "demographic_contrast", "acquisition_contrast"}:
        return DiagnosisContrastClaimExtractor(criteria, feature_family_labels)
    if claim_type in {"mmse_association", "age_association", "continuous_association"}:
        return ContinuousAssociationClaimExtractor(criteria, feature_family_labels)
    if claim_type == "latent_morphometry_axis":
        return LatentMorphometryClaimExtractor(criteria, feature_family_labels)
    if claim_type == "normative_deviation":
        return NormativeDeviationClaimExtractor(criteria, feature_family_labels)
    raise ValueError(f"Unsupported claim type: {claim_type}")
