import numpy as np
import pandas as pd

from scs_morph.evaluation.metrics import (
    compute_evidence_stability,
    compute_support_stability,
    direction_retention,
    jaccard,
    magnitude_drift,
    rank_overlap_topk,
    safe_pearson,
    safe_spearman,
    sign_flip_fraction,
    weighted_jaccard_from_abs_values,
)


def test_basic_stability_metrics():
    x = [1, 2, 3, 4]
    y = [1, 3, 2, 4]
    assert safe_spearman(x, y) is not None
    assert safe_pearson(x, y) is not None
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert direction_retention([1, -1, 2], [2, -3, -1]) == 2 / 3
    assert np.isclose(sign_flip_fraction([1, -1, 2], [2, -3, -1]), 1 / 3)
    assert magnitude_drift(x, y) is not None
    assert rank_overlap_topk(x, y, k=2) is not None
    assert weighted_jaccard_from_abs_values([1, 2], [1, 4]) == 3 / 5


def test_compute_evidence_and_support_stability():
    matched = pd.DataFrame({"roi": ["a", "b", "c"], "source_t": [3, 2, 1], "target_t": [2.5, 2, -1]})
    source = pd.DataFrame({"roi": ["a", "b", "c"], "t": [3, 2, 1], "q": [0.01, 0.2, 0.3]})
    target = pd.DataFrame({"roi": ["a", "b", "c"], "t": [2.5, 2, -1], "q": [0.01, 0.05, 0.3]})
    evidence = compute_evidence_stability(matched, top_k=2)
    support = compute_support_stability(source, target, matched, top_k=2)
    assert evidence["n_matched"] == 3
    assert support["support_overlap_count"] >= 1
