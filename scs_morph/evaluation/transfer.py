from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scs_morph.evaluation.metrics import safe_pearson, safe_spearman
from scs_morph.evaluation.pairs import EvaluationPair


def build_claim_weight_vector(source_stats: pd.DataFrame, matched_feature_names, weight_metric: str = "coefficient") -> pd.Series:
    if source_stats is None or source_stats.empty:
        return pd.Series(dtype=float)
    stats = source_stats.set_index("roi")
    if weight_metric == "coefficient":
        col = "coef" if "coef" in stats.columns else "t"
    elif weight_metric == "t_value":
        col = "t"
    elif weight_metric == "evidence_value":
        col = "evidence_value" if "evidence_value" in stats.columns else "t"
    else:
        col = "coef" if "coef" in stats.columns else "t"
    weights = pd.to_numeric(stats.reindex(list(matched_feature_names))[col], errors="coerce").fillna(0.0)
    norm = float(np.linalg.norm(weights.to_numpy(dtype=float)))
    return weights if norm == 0 else weights / norm


def compute_claim_score(X_target: pd.DataFrame, weights: pd.Series, standardize: bool = True) -> pd.Series:
    common = [col for col in weights.index if col in X_target.columns]
    if not common:
        return pd.Series(dtype=float)
    X = X_target[common].astype(float)
    w = weights.reindex(common).fillna(0.0)
    if standardize:
        std = X.std(axis=0, ddof=0).replace(0, np.nan)
        X = (X - X.mean(axis=0)) / std
        X = X.fillna(0.0)
    return pd.Series(X.to_numpy(dtype=float) @ w.to_numpy(dtype=float), index=X.index, name="claim_score")


def _rank_auc(score: np.ndarray, y: np.ndarray) -> float | None:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def evaluate_binary_transfer(score: pd.Series, y_binary: pd.Series) -> dict:
    aligned = pd.concat([score.rename("score"), y_binary.rename("y")], axis=1).dropna()
    aligned["y"] = pd.to_numeric(aligned["y"], errors="coerce")
    aligned = aligned[aligned["y"].isin([0, 1])]
    if aligned.empty:
        return {"n": 0, "n_positive": 0, "n_negative": 0, "auc_raw": None, "auc_oriented": None, "cohens_d": None, "point_biserial_corr": None}
    s = aligned["score"].to_numpy(dtype=float)
    y = aligned["y"].to_numpy(dtype=int)
    auc = _rank_auc(s, y)
    pos = s[y == 1]
    neg = s[y == 0]
    pooled = np.sqrt(((len(pos) - 1) * np.var(pos, ddof=1) + (len(neg) - 1) * np.var(neg, ddof=1)) / max(len(pos) + len(neg) - 2, 1)) if len(pos) > 1 and len(neg) > 1 else np.nan
    d = (np.mean(pos) - np.mean(neg)) / pooled if pooled and np.isfinite(pooled) and pooled > 0 else None
    return {
        "n": int(len(aligned)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "auc_raw": auc,
        "auc_oriented": max(auc, 1 - auc) if auc is not None else None,
        "cohens_d": float(d) if d is not None and np.isfinite(d) else None,
        "point_biserial_corr": safe_pearson(s, y),
    }


def evaluate_continuous_transfer(score: pd.Series, y: pd.Series) -> dict:
    aligned = pd.concat([score.rename("score"), y.rename("y")], axis=1).dropna()
    aligned["y"] = pd.to_numeric(aligned["y"], errors="coerce")
    aligned = aligned.dropna()
    if aligned.empty:
        return {"n": 0, "spearman_corr": None, "pearson_corr": None, "slope_simple": None}
    score_values = aligned["score"].to_numpy(dtype=float)
    y_values = aligned["y"].to_numpy(dtype=float)
    denom = np.var(score_values)
    slope = np.cov(score_values, y_values, ddof=0)[0, 1] / denom if denom > 0 else None
    return {
        "n": int(len(aligned)),
        "spearman_corr": safe_spearman(score_values, y_values),
        "pearson_corr": safe_pearson(score_values, y_values),
        "slope_simple": float(slope) if slope is not None and np.isfinite(slope) else None,
    }


def evaluate_claim_score_transfer(
    pair: EvaluationPair,
    source_stats: pd.DataFrame,
    target_X: pd.DataFrame,
    target_meta: pd.DataFrame,
    matched_features,
) -> dict:
    if not pair.transfer_target or pair.transfer_target_type in {None, "none"}:
        return {}
    if pair.transfer_target not in target_meta.columns:
        return {"error": f"transfer target {pair.transfer_target} missing"}
    weights = build_claim_weight_vector(source_stats, matched_features, pair.use_weight_metric)
    score = compute_claim_score(target_X, weights, pair.standardize_target_features)
    if score.empty:
        return {"error": "no matched target features for claim score"}
    y = target_meta.loc[score.index, pair.transfer_target] if score.index.isin(target_meta.index).all() else target_meta[pair.transfer_target].reindex(score.index)
    if pair.transfer_target_type == "binary":
        return evaluate_binary_transfer(score, y)
    if pair.transfer_target_type == "continuous":
        return evaluate_continuous_transfer(score, y)
    return {}

