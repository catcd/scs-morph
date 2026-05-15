from __future__ import annotations

import numpy as np
import pandas as pd


def select_support_regions(
    results_df: pd.DataFrame,
    top_k: int = 20,
    evidence_col: str = "t",
) -> list[str]:
    """Select top regions by absolute evidence metric."""
    valid = results_df[["roi", evidence_col]].dropna()
    if valid.empty:
        return []
    ranked = valid.assign(abs_metric=valid[evidence_col].abs()).sort_values(
        "abs_metric", ascending=False
    )
    return ranked.head(top_k)["roi"].astype(str).tolist()


def compute_support_metrics(results_df: pd.DataFrame, support_regions: list[str]) -> dict:
    """Compute simple support metrics for a claim."""
    support = results_df[results_df["roi"].isin(support_regions)].copy()
    q = pd.to_numeric(results_df["q"], errors="coerce") if "q" in results_df.columns else pd.Series(dtype=float)
    t_all = pd.to_numeric(results_df["t"], errors="coerce") if "t" in results_df.columns else pd.Series(dtype=float)
    coef_support = (
        pd.to_numeric(support["coef"], errors="coerce") if "coef" in support.columns else pd.Series(dtype=float)
    )
    t_support = pd.to_numeric(support["t"], errors="coerce") if "t" in support.columns else pd.Series(dtype=float)

    n_support = int(len(support))
    positive_fraction = float((coef_support > 0).mean()) if n_support else np.nan
    negative_fraction = float((coef_support < 0).mean()) if n_support else np.nan

    return {
        "n_features": int(len(results_df)),
        "n_support": n_support,
        "n_fdr_0_05": int((q <= 0.05).sum()),
        "n_fdr_0_10": int((q <= 0.10).sum()),
        "mean_abs_t_support": float(t_support.abs().mean()) if n_support else np.nan,
        "max_abs_t": float(t_all.abs().max()) if len(results_df) else np.nan,
        "mean_abs_coef_support": float(coef_support.abs().mean()) if n_support else np.nan,
        "support_positive_fraction": positive_fraction,
        "support_negative_fraction": negative_fraction,
    }


def summarize_support_regions(support_regions: list[str], support_effects: dict[str, float]) -> str:
    """Return a short text summary of support regions."""
    if not support_regions:
        return "no regions"
    pieces = []
    for region in support_regions[:5]:
        effect = support_effects.get(region)
        if effect is None or not np.isfinite(effect):
            pieces.append(region)
        else:
            pieces.append(f"{region} ({effect:.3g})")
    return ", ".join(pieces)
