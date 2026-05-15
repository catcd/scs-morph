from __future__ import annotations

import numpy as np
import pandas as pd

from scs_morph.analysis.design_matrix import encode_covariates, standardize_feature_matrix
from scs_morph.analysis.settings import AnalysisSetting, setting_to_dict, source_setting_from_analysis
from scs_morph.claims.base import BaseClaimExtractor, _claim_id, _family_id, _feature_family_label
from scs_morph.claims.cards import ClaimCard
from scs_morph.claims.family import ClaimFamily
from scs_morph.claims.filters import ClaimCriteria, latent_axis_passes_criteria
from scs_morph.claims.grouping import make_region_summary
from scs_morph.claims.templates import CLAIM_TEMPLATES
from scs_morph.claims.text_generation import render_template


def _residualize(X: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    if covariates.empty:
        return X.astype(float)
    design = encode_covariates(covariates)
    if design.empty:
        return X.astype(float)
    matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    residuals = {}
    for column in X.columns:
        y = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
        beta = np.linalg.pinv(matrix) @ y
        residuals[column] = y - matrix @ beta
    return pd.DataFrame(residuals, index=X.index)


def _pca(X: pd.DataFrame, n_components: int):
    matrix = X.to_numpy(dtype=float)
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(matrix)
        return scores, pca.components_, pca.explained_variance_ratio_
    except Exception:
        _, s, vt = np.linalg.svd(matrix, full_matrices=False)
        components = vt[:n_components]
        scores = matrix @ components.T
        explained = (s ** 2) / np.sum(s ** 2)
        return scores, components, explained[:n_components]


def _spearman(x, y) -> float:
    xr = pd.Series(x).rank().to_numpy(dtype=float)
    yr = pd.Series(y).rank().to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _cohens_d(scores, groups) -> float:
    values = pd.Series(groups).dropna().unique()
    if len(values) != 2:
        return np.nan
    s = pd.Series(scores)
    g = pd.Series(groups)
    a = s[g == values[1]].astype(float)
    b = s[g == values[0]].astype(float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled and np.isfinite(pooled) else np.nan


class LatentMorphometryClaimExtractor(BaseClaimExtractor):
    """Generate unsupervised latent-axis claim families."""

    def extract_family(self, X: pd.DataFrame, metadata: pd.DataFrame, setting: AnalysisSetting) -> tuple[ClaimFamily, pd.DataFrame]:
        criteria = ClaimCriteria.from_dict({**self.criteria.__dict__, **setting.criteria, "min_n": setting.min_n})
        missing_covariates = [c for c in setting.covariates if c not in metadata.columns]
        covariates = [c for c in setting.covariates if c in metadata.columns]
        data = metadata[covariates].copy() if covariates else pd.DataFrame(index=metadata.index)
        keep = data.notna().all(axis=1) if covariates else pd.Series(True, index=metadata.index)
        X_use = X.loc[keep].reset_index(drop=True).astype(float)
        meta_use = metadata.loc[keep].reset_index(drop=True)
        cov_use = data.loc[keep].reset_index(drop=True)
        if len(X_use) < setting.min_n:
            raise ValueError(f"n={len(X_use)} below min_n={setting.min_n}")
        residual = _residualize(X_use, cov_use)
        standardized = standardize_feature_matrix(residual)
        scores, components, explained = _pca(standardized, setting.n_components)
        family_id = _family_id(setting)
        family = ClaimFamily(
            claim_family_id=family_id,
            setting_id=setting.setting_id,
            setting_metadata=setting_to_dict(setting),
            extraction_summary={"n_samples": len(X_use), "n_features": X_use.shape[1]},
            warnings=[f"Dropped missing covariates: {missing_covariates}"] if missing_covariates else [],
        )
        feature_label = _feature_family_label(setting, self.feature_family_labels)
        loading_rows = []
        for comp_idx, variance in enumerate(explained, start=1):
            loadings = pd.Series(components[comp_idx - 1], index=X_use.columns)
            top_abs = loadings.abs().sort_values(ascending=False).head(criteria.latent_top_k).index.astype(str).tolist()
            loading_rows.extend([{"component": comp_idx, "roi": roi, "loading": float(loadings[roi]), "explained_variance_ratio": float(variance)} for roi in X_use.columns])
            axis_summary = {"explained_variance_ratio": float(variance)}
            if latent_axis_passes_criteria(axis_summary, criteria):
                text = render_template("latent_morphometry_axis", explained_variance_pct=round(100 * float(variance), 1), top_regions=make_region_summary(top_abs, setting.feature_space))
                family.latent_claims.append(ClaimCard(
                    claim_id=_claim_id(family_id, "latent_axis", f"PC{comp_idx}"),
                    claim_family_id=family_id,
                    claim_level="latent_axis",
                    claim_type="latent_morphometry_axis",
                    source_setting=source_setting_from_analysis(setting).to_dict(),
                    dataset=setting.dataset,
                    cohort=setting.cohort,
                    feature_space=setting.feature_space,
                    feature_family=setting.feature_family,
                    pipeline=setting.pipeline,
                    atlas=setting.atlas,
                    target=None,
                    covariates=covariates,
                    n_samples=len(X_use),
                    n_features=X_use.shape[1],
                    evidence_metric="explained_variance_ratio",
                    evidence_value=float(variance),
                    support_regions=top_abs,
                    support_region_effects={roi: float(loadings[roi]) for roi in top_abs},
                    criteria_passed=True,
                    criteria_details=axis_summary,
                    claim_text=text,
                    short_claim_text=text,
                    machine_readable_statement={"component": comp_idx, "explained_variance_ratio": float(variance)},
                    status="supported",
                ))
            for target in setting.metadata_association_targets:
                if target not in meta_use.columns:
                    continue
                series = meta_use[target]
                numeric = pd.to_numeric(series, errors="coerce")
                mask = numeric.notna()
                association = np.nan
                template_id = "latent_axis_association_continuous"
                labels = {"positive_label": "group 1", "negative_label": "group 0"}
                if mask.sum() >= setting.min_n and numeric.dropna().nunique() > 2:
                    association = _spearman(scores[:, comp_idx - 1][mask], numeric[mask])
                    passed = abs(association) >= criteria.latent_min_metadata_abs_corr if np.isfinite(association) else False
                    text = render_template(template_id, metadata_variable=target, association_value=round(float(association), 3))
                else:
                    clean = series.dropna()
                    association = _cohens_d(scores[:, comp_idx - 1], series)
                    passed = abs(association) >= criteria.latent_min_metadata_abs_effect if np.isfinite(association) else False
                    text = render_template("latent_axis_association_binary", positive_label=labels["positive_label"], negative_label=labels["negative_label"], association_value=round(float(association), 3) if np.isfinite(association) else "NA")
                if passed:
                    family.latent_claims.append(ClaimCard(
                        claim_id=_claim_id(family_id, "latent_axis_association", f"PC{comp_idx}_{target}"),
                        claim_family_id=family_id,
                        claim_level="latent_axis_association",
                        claim_type="latent_axis_association",
                        source_setting=source_setting_from_analysis(setting).to_dict(),
                        dataset=setting.dataset,
                        cohort=setting.cohort,
                        feature_space=setting.feature_space,
                        feature_family=setting.feature_family,
                        pipeline=setting.pipeline,
                        atlas=setting.atlas,
                        target=target,
                        covariates=covariates,
                        n_samples=len(X_use),
                        n_features=X_use.shape[1],
                        evidence_metric="spearman_or_cohens_d",
                        evidence_value=float(association),
                        criteria_passed=True,
                        criteria_details={"component": comp_idx, "association": float(association)},
                        claim_text=text,
                        short_claim_text=text,
                        machine_readable_statement={"component": comp_idx, "metadata_variable": target, "association": float(association)},
                        status="supported",
                    ))
        return family, pd.DataFrame(loading_rows)

