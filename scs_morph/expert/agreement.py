from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy import stats


def pairwise_reviewer_correlations(scores_wide: pd.DataFrame, dimensions) -> pd.DataFrame:
    rows = []
    for dim in dimensions:
        pivot = scores_wide.pivot_table(index="claim_id", columns="reviewer_id", values=dim, aggfunc="mean")
        for a, b in itertools.combinations(pivot.columns, 2):
            pair = pivot[[a, b]].dropna()
            pearson = pair[a].corr(pair[b], method="pearson") if len(pair) >= 2 else np.nan
            spearman = pair[a].corr(pair[b], method="spearman") if len(pair) >= 2 else np.nan
            rows.append({"dimension": dim, "reviewer_a": a, "reviewer_b": b, "n_overlap": len(pair), "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


def _icc_components(score_matrix) -> tuple[int, int, float, float, float] | None:
    matrix = pd.DataFrame(score_matrix).dropna()
    n, k = matrix.shape
    if n < 2 or k < 2:
        return None
    values = matrix.to_numpy(dtype=float)
    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ssr = k * ((row_means - grand) ** 2).sum()
    ssc = n * ((col_means - grand) ** 2).sum()
    sse = ((values - row_means[:, None] - col_means[None, :] + grand) ** 2).sum()
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    return n, k, msr, msc, mse


def compute_icc_2_1(score_matrix) -> float | None:
    comp = _icc_components(score_matrix)
    if comp is None:
        return None
    n, k, msr, msc, mse = comp
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom else None


def compute_icc_2_k(score_matrix) -> float | None:
    comp = _icc_components(score_matrix)
    if comp is None:
        return None
    n, _, msr, msc, mse = comp
    denom = msr + (msc - mse) / n
    return float((msr - mse) / denom) if denom else None


def compute_agreement_by_dimension(scores_long: pd.DataFrame, reviewer_subset: str = "human") -> pd.DataFrame:
    subset = scores_long if reviewer_subset == "all" else scores_long[scores_long["reviewer_type"] == reviewer_subset]
    rows = []
    for dim, group in subset.groupby("dimension"):
        pivot = group.pivot_table(index="claim_id", columns="reviewer_id", values="score", aggfunc="mean")
        corr = pairwise_reviewer_correlations(group.pivot_table(index=["claim_id", "reviewer_id", "reviewer_type"], columns="dimension", values="score", aggfunc="mean").reset_index(), [dim])
        rows.append({
            "reviewer_subset": reviewer_subset,
            "dimension": dim,
            "n_claims_complete": int(pivot.dropna().shape[0]),
            "n_reviewers": int(pivot.shape[1]),
            "icc_2_1": compute_icc_2_1(pivot),
            "icc_2_k": compute_icc_2_k(pivot),
            "mean_pairwise_pearson": corr["pearson"].mean() if not corr.empty else np.nan,
            "mean_pairwise_spearman": corr["spearman"].mean() if not corr.empty else np.nan,
        })
    return pd.DataFrame(rows)


def compute_human_llm_agreement(scores_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    humans = scores_long[scores_long["reviewer_type"] == "human"]
    llms = scores_long[scores_long["reviewer_type"] == "llm"]
    human_mean = humans.groupby(["claim_id", "dimension"])["score"].mean().rename("human_mean").reset_index()
    for (llm, dim), group in llms.groupby(["reviewer_id", "dimension"]):
        merged = group.merge(human_mean[human_mean["dimension"] == dim], on=["claim_id", "dimension"])
        merged = merged.dropna(subset=["score", "human_mean"])
        rows.append({
            "llm_reviewer_id": llm,
            "dimension": dim,
            "n_overlap": int(len(merged)),
            "correlation_with_human_mean": merged["score"].corr(merged["human_mean"], method="spearman") if len(merged) >= 2 else np.nan,
            "mean_absolute_difference": (merged["score"] - merged["human_mean"]).abs().mean() if len(merged) else np.nan,
            "bias": (merged["score"] - merged["human_mean"]).mean() if len(merged) else np.nan,
        })
    return pd.DataFrame(rows)

