from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scs_morph.analysis.design_matrix import (
    build_design_matrix,
    standardize_feature_matrix,
    standardize_numeric_series,
)

try:
    from scipy import stats
except Exception:  # pragma: no cover - fallback for minimal environments
    stats = None


def _two_sided_t_pvalue(t_value: float, df: int) -> float:
    if not np.isfinite(t_value) or df <= 0:
        return np.nan
    if stats is not None:
        return float(2.0 * stats.t.sf(abs(t_value), df))
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def fit_roiwise_ols(
    X: pd.DataFrame,
    y: pd.Series,
    covariates: pd.DataFrame | None,
    standardize_features: bool = True,
    standardize_target: bool = False,
) -> pd.DataFrame:
    """Fit ROI-wise OLS models: ROI ~ target + covariates."""
    X_base = X.reset_index(drop=True).copy()
    y_work = pd.to_numeric(y.reset_index(drop=True), errors="coerce").astype(float)
    if standardize_target:
        y_work = standardize_numeric_series(y_work)

    if covariates is None:
        covariates_work = pd.DataFrame(index=X_base.index)
    elif len(covariates) == len(X) and covariates.index.equals(X.index):
        covariates_work = covariates.reset_index(drop=True).copy()
    else:
        covariates_work = covariates.loc[X.index].reset_index(drop=True).copy()

    design = build_design_matrix(y_work, covariates_work, include_intercept=True)
    X_work = standardize_feature_matrix(X_base) if standardize_features else X_base.astype(float)

    rows: list[dict[str, float | int | str]] = []
    for roi in X_work.columns:
        roi_values = pd.to_numeric(X_work[roi], errors="coerce").astype(float)
        fit_table = pd.concat([roi_values.rename("roi_value"), design], axis=1)
        fit_table = fit_table.replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(fit_table))
        p_design = int(design.shape[1])

        result = {
            "roi": str(roi),
            "coef": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p": np.nan,
            "n": n,
            "mean_roi": float(roi_values.mean()) if n > 0 else np.nan,
            "std_roi": float(roi_values.std(ddof=0)) if n > 0 else np.nan,
        }

        if n <= p_design:
            rows.append(result)
            continue

        y_roi = fit_table["roi_value"].to_numpy(dtype=float)
        matrix = fit_table[design.columns].to_numpy(dtype=float)

        try:
            beta = np.linalg.lstsq(matrix, y_roi, rcond=None)[0]
            residuals = y_roi - matrix @ beta
            dof = n - matrix.shape[1]
            sigma2 = float((residuals @ residuals) / dof)
            xtx_inv = np.linalg.pinv(matrix.T @ matrix)
            se_variance = np.diag(xtx_inv) * sigma2
            se_variance = np.where(se_variance >= 0, se_variance, np.nan)
            se_all = np.sqrt(se_variance)
            target_idx = list(design.columns).index("__target__")
            coef = float(beta[target_idx])
            se = float(se_all[target_idx])
            t_value = coef / se if se > 0 else np.nan
            result.update(
                {
                    "coef": coef,
                    "se": se,
                    "t": float(t_value),
                    "p": _two_sided_t_pvalue(float(t_value), dof),
                }
            )
        except Exception:
            pass

        rows.append(result)

    return pd.DataFrame(rows)


def benjamini_hochberg(p_values) -> np.ndarray:
    """Compute Benjamini-Hochberg q-values in original order."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q

    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    valid_q = np.empty_like(valid_p)
    valid_q[order] = adjusted
    q[valid] = valid_q
    return q


def add_fdr(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values to ROI statistics."""
    results = results_df.copy()
    results["q"] = benjamini_hochberg(results["p"].to_numpy(dtype=float))
    return results
