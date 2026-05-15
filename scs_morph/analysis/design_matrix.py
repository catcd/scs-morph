from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _align_X_metadata(X: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X) == len(metadata) and X.index.equals(metadata.index):
        return X.reset_index(drop=True).copy(), metadata.reset_index(drop=True).copy()

    if X.index.is_unique and metadata.index.is_unique:
        common_index = X.index.intersection(metadata.index)
        return X.loc[common_index].reset_index(drop=True).copy(), metadata.loc[common_index].reset_index(drop=True).copy()

    if len(X) == len(metadata):
        warnings.warn(
            "X and metadata have duplicate or nonmatching labels; using row-position alignment.",
            RuntimeWarning,
            stacklevel=2,
        )
        return X.reset_index(drop=True).copy(), metadata.reset_index(drop=True).copy()

    raise ValueError("Cannot align X and metadata with duplicate labels and different row counts.")


def prepare_analysis_table(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    target: str,
    covariates: list[str],
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Align feature and metadata tables and remove unusable rows."""
    X_aligned, metadata_aligned = _align_X_metadata(X, metadata)

    required = [target, *covariates]
    missing = [column for column in required if column not in metadata_aligned.columns]
    if missing:
        raise KeyError(f"Missing required metadata columns: {missing}")

    y = metadata_aligned[target].copy()
    covariate_df = metadata_aligned[covariates].copy()
    keep = pd.Series(True, index=X_aligned.index)

    if drop_missing:
        keep &= y.notna()
        if covariates:
            keep &= covariate_df.notna().all(axis=1)

    finite_features = np.isfinite(X_aligned.to_numpy(dtype=float)).all(axis=1)
    keep &= pd.Series(finite_features, index=X_aligned.index)

    X_clean = X_aligned.loc[keep].astype(float)
    y_clean = pd.to_numeric(y.loc[keep], errors="coerce")
    covariate_df_clean = covariate_df.loc[keep].copy()
    metadata_clean = metadata_aligned.loc[keep].copy()

    if drop_missing:
        finite_target = np.isfinite(y_clean.to_numpy(dtype=float))
        X_clean = X_clean.loc[finite_target].reset_index(drop=True)
        y_clean = y_clean.loc[finite_target].reset_index(drop=True)
        covariate_df_clean = covariate_df_clean.loc[finite_target].reset_index(drop=True)
        metadata_clean = metadata_clean.loc[finite_target].reset_index(drop=True)

    return X_clean, y_clean, covariate_df_clean, metadata_clean


def encode_covariates(covariate_df: pd.DataFrame) -> pd.DataFrame:
    """Encode covariates as a numeric design matrix."""
    if covariate_df.empty:
        return pd.DataFrame(index=covariate_df.index)

    numeric = covariate_df.copy()
    for column in numeric.columns:
        converted = pd.to_numeric(numeric[column], errors="coerce")
        if converted.notna().all():
            numeric[column] = converted.astype(float)

    encoded = pd.get_dummies(numeric, drop_first=True, dtype=float)
    return encoded.astype(float)


def build_design_matrix(
    target_series: pd.Series,
    covariate_df: pd.DataFrame,
    include_intercept: bool = True,
) -> pd.DataFrame:
    """Build regression design matrix with target named '__target__'."""
    parts: list[pd.DataFrame] = []
    if include_intercept:
        parts.append(pd.DataFrame({"intercept": 1.0}, index=target_series.index))

    parts.append(pd.DataFrame({"__target__": target_series.astype(float)}, index=target_series.index))
    encoded_covariates = encode_covariates(covariate_df)
    if not encoded_covariates.empty:
        parts.append(encoded_covariates)

    return pd.concat(parts, axis=1).astype(float)


def standardize_numeric_series(series: pd.Series) -> pd.Series:
    """Return z-scored numeric series, using zeros for zero variance."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, name=series.name)
    return (values - mean) / std


def standardize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature column, replacing zero-variance columns with zeros."""
    values = X.astype(float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    zero_variance = (~np.isfinite(stds)) | (stds == 0)
    if zero_variance.any():
        warnings.warn(
            f"{int(zero_variance.sum())} zero-variance feature columns were set to zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    stds = stds.mask(zero_variance, 1.0)
    standardized = (values - means) / stds
    standardized.loc[:, zero_variance] = 0.0
    return standardized
