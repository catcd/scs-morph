from __future__ import annotations

import numpy as np
import pandas as pd

from scs_morph.analysis.design_matrix import encode_covariates
from scs_morph.analysis.settings import AnalysisSetting, setting_to_dict, source_setting_from_analysis
from scs_morph.claims.base import BaseClaimExtractor, _claim_id, _family_id, _feature_family_label
from scs_morph.claims.cards import ClaimCard
from scs_morph.claims.family import ClaimFamily
from scs_morph.claims.filters import ClaimCriteria, normative_claim_passes_criteria, group_passes_criteria
from scs_morph.claims.grouping import (
    annotate_roi_statistics,
    format_group_label,
    make_group_label,
    make_region_summary,
    summarize_roi_groups,
)
from scs_morph.claims.text_generation import direction_from_effect, render_template


def _fit_reference_predictions(X_ref: pd.DataFrame, cov_ref: pd.DataFrame, cov_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    design_ref = encode_covariates(cov_ref)
    design_all = encode_covariates(cov_all).reindex(columns=design_ref.columns, fill_value=0.0)
    m_ref = np.column_stack([np.ones(len(design_ref)), design_ref.to_numpy(dtype=float)])
    m_all = np.column_stack([np.ones(len(design_all)), design_all.to_numpy(dtype=float)])
    preds = {}
    stds = {}
    for roi in X_ref.columns:
        y = X_ref[roi].to_numpy(dtype=float)
        beta = np.linalg.pinv(m_ref) @ y
        ref_resid = y - m_ref @ beta
        std = float(np.std(ref_resid, ddof=max(1, m_ref.shape[1])))
        preds[roi] = m_all @ beta
        stds[roi] = std if std > 0 and np.isfinite(std) else np.nan
    return pd.DataFrame(preds, index=cov_all.index), pd.Series(stds)


class NormativeDeviationClaimExtractor(BaseClaimExtractor):
    """Generate normative deviation claim families."""

    def extract_family(self, X: pd.DataFrame, metadata: pd.DataFrame, setting: AnalysisSetting) -> tuple[ClaimFamily, pd.DataFrame]:
        criteria = ClaimCriteria.from_dict({**self.criteria.__dict__, **setting.criteria, "min_n": setting.min_n, "normative_z_threshold": setting.z_threshold})
        if not setting.reference_filter or not setting.target_group_column or not setting.target_group_values:
            raise ValueError("Normative settings require reference_filter, target_group_column, and target_group_values.")
        covariates = [c for c in setting.covariates if c in metadata.columns and metadata[c].notna().mean() > 0.2]
        meta = metadata.reset_index(drop=True)
        X_use = X.reset_index(drop=True).astype(float)
        ref_mask = meta.eval(setting.reference_filter)
        if int(ref_mask.sum()) < setting.min_n:
            raise ValueError(f"reference group n={int(ref_mask.sum())} below min_n={setting.min_n}")
        cov_all = meta[covariates].copy() if covariates else pd.DataFrame(index=meta.index)
        complete = cov_all.notna().all(axis=1) if covariates else pd.Series(True, index=meta.index)
        ref_mask &= complete
        preds, ref_stds = _fit_reference_predictions(X_use.loc[ref_mask], cov_all.loc[ref_mask], cov_all.loc[complete])
        z = (X_use.loc[complete] - preds) / ref_stds
        meta_complete = meta.loc[complete].reset_index(drop=True)
        z = z.reset_index(drop=True)
        family_id = _family_id(setting)
        family = ClaimFamily(
            claim_family_id=family_id,
            setting_id=setting.setting_id,
            setting_metadata=setting_to_dict(setting),
            extraction_summary={"n_samples": len(z), "n_features": X.shape[1]},
        )
        feature_label = _feature_family_label(setting, self.feature_family_labels)
        all_stats = []
        for target_group in setting.target_group_values:
            target_mask = meta_complete[setting.target_group_column].astype(str) == str(target_group)
            if int(target_mask.sum()) < setting.min_n:
                family.warnings.append(f"target group {target_group} n={int(target_mask.sum())} below min_n={setting.min_n}")
                continue
            mean_z = z.loc[target_mask].mean(axis=0)
            stats = pd.DataFrame({"roi": mean_z.index.astype(str), "coef": mean_z.values, "t": mean_z.values, "mean_z": mean_z.values, "n": int(target_mask.sum())})
            stats = annotate_roi_statistics(stats, setting.feature_space)
            passing = stats[stats["mean_z"].abs() >= criteria.normative_z_threshold].copy()
            support_regions = passing.sort_values("mean_z", key=lambda s: s.abs(), ascending=False)["roi"].head(criteria.normative_max_atomic_claims).tolist()
            support_effects = {row["roi"]: float(row["mean_z"]) for _, row in passing.iterrows()}
            summary = {"n_support_rois": len(support_regions), "mean_abs_z": float(passing["mean_z"].abs().mean()) if len(passing) else 0.0}
            if normative_claim_passes_criteria(summary, criteria):
                direction = direction_from_effect(float(passing["mean_z"].mean()))
                text = render_template("normative_deviation_composite", target_group=target_group, feature_family_label=feature_label, region_summary=make_region_summary(support_regions, setting.feature_space), reference_group=setting.reference_group_label or "reference")
                family.normative_claims.append(ClaimCard(
                    claim_id=_claim_id(family_id, "normative_deviation", str(target_group)),
                    claim_family_id=family_id,
                    claim_level="normative_deviation",
                    claim_type="normative_deviation",
                    source_setting=source_setting_from_analysis(setting).to_dict(),
                    dataset=setting.dataset,
                    cohort=setting.cohort,
                    feature_space=setting.feature_space,
                    feature_family=setting.feature_family,
                    pipeline=setting.pipeline,
                    atlas=setting.atlas,
                    target=setting.target_group_column,
                    covariates=covariates,
                    n_samples=int(target_mask.sum()),
                    n_features=X.shape[1],
                    direction=direction,
                    evidence_metric="mean_abs_z",
                    evidence_value=summary["mean_abs_z"],
                    support_regions=support_regions,
                    support_region_effects=support_effects,
                    criteria_passed=True,
                    criteria_details=summary,
                    claim_text=text,
                    short_claim_text=text,
                    machine_readable_statement={"target_group": target_group, "reference_group": setting.reference_group_label},
                    status="supported",
                ))
            for _, row in passing.head(criteria.normative_max_atomic_claims).iterrows():
                roi = str(row["roi"])
                text = render_template("normative_deviation_atomic_roi", target_group=target_group, roi=roi, feature_family_label=feature_label, reference_group=setting.reference_group_label or "reference")
                family.normative_claims.append(ClaimCard(
                    claim_id=_claim_id(family_id, "normative_atomic_roi", f"{target_group}_{roi}"),
                    claim_family_id=family_id,
                    claim_level="normative_atomic_roi",
                    claim_type="normative_deviation",
                    source_setting=source_setting_from_analysis(setting).to_dict(),
                    dataset=setting.dataset,
                    cohort=setting.cohort,
                    feature_space=setting.feature_space,
                    feature_family=setting.feature_family,
                    pipeline=setting.pipeline,
                    atlas=setting.atlas,
                    target=setting.target_group_column,
                    covariates=covariates,
                    n_samples=int(target_mask.sum()),
                    n_features=X.shape[1],
                    roi=roi,
                    hemisphere=row.get("hemisphere"),
                    roi_group=row.get("roi_group"),
                    anatomical_group=row.get("anatomical_group"),
                    network=row.get("network"),
                    lobe=row.get("lobe"),
                    direction=direction_from_effect(row["mean_z"]),
                    evidence_metric="mean_z",
                    evidence_value=float(row["mean_z"]),
                    coefficient=float(row["mean_z"]),
                    criteria_passed=True,
                    criteria_details={"abs_z_threshold": criteria.normative_z_threshold},
                    claim_text=text,
                    short_claim_text=text,
                    machine_readable_statement={"target_group": target_group, "roi": roi},
                    status="supported",
                ))
            group_summary = summarize_roi_groups(stats, support_regions, setting.feature_space)
            for _, row in group_summary.iterrows():
                if group_passes_criteria(row, criteria):
                    raw_group_label = make_group_label(row, setting.feature_space)
                    formatted_group_label = format_group_label(raw_group_label, setting.feature_space)
                    text = render_template("normative_group_deviation", target_group=target_group, group_label=formatted_group_label, n_support_rois=int(row["n_support_rois"]), n_group_rois=int(row["n_group_rois"]))
                    family.normative_claims.append(ClaimCard(
                        claim_id=_claim_id(family_id, "normative_deviation", f"{target_group}_{raw_group_label}"),
                        claim_family_id=family_id,
                        claim_level="group_summary",
                        claim_type="normative_group_deviation",
                        source_setting=source_setting_from_analysis(setting).to_dict(),
                        dataset=setting.dataset,
                        cohort=setting.cohort,
                        feature_space=setting.feature_space,
                        feature_family=setting.feature_family,
                        pipeline=setting.pipeline,
                        atlas=setting.atlas,
                        target=setting.target_group_column,
                        covariates=covariates,
                        n_samples=int(target_mask.sum()),
                        n_features=X.shape[1],
                        roi_group=row.get("roi_group"),
                        anatomical_group=row.get("anatomical_group"),
                        network=row.get("network"),
                        lobe=row.get("lobe"),
                        evidence_metric="mean_abs_z",
                        evidence_value=float(row["mean_abs_t"]),
                        criteria_passed=True,
                        criteria_details=row.to_dict(),
                        claim_text=text,
                        short_claim_text=text,
                        support_metrics={"group_label": raw_group_label, "formatted_group_label": formatted_group_label},
                        machine_readable_statement={
                            "target_group": target_group,
                            "group": raw_group_label,
                            "group_label": raw_group_label,
                            "formatted_group_label": formatted_group_label,
                            "network_7_like": row.get("network_7_like"),
                        },
                        status="supported",
                    ))
            stats["target_group"] = target_group
            all_stats.append(stats)
        return family, pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
