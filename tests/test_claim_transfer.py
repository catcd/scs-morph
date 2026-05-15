import pandas as pd

from scs_morph.evaluation.transfer import (
    build_claim_weight_vector,
    compute_claim_score,
    evaluate_binary_transfer,
    evaluate_continuous_transfer,
)


def test_claim_score_and_transfer_metrics():
    stats = pd.DataFrame({"roi": ["a", "b"], "coef": [1.0, -1.0], "t": [2.0, -2.0]})
    weights = build_claim_weight_vector(stats, ["a", "b"], "coefficient")
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    score = compute_claim_score(X, weights)
    assert len(score) == 4
    binary = evaluate_binary_transfer(score, pd.Series([0, 0, 1, 1]))
    continuous = evaluate_continuous_transfer(score, pd.Series([1, 2, 3, 4]))
    assert binary["auc_oriented"] is not None
    assert binary["cohens_d"] is not None
    assert continuous["spearman_corr"] is not None

