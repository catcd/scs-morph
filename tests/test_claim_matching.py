import pandas as pd

from scs_morph.evaluation.aggregation import aggregate_roi_statistics
from scs_morph.evaluation.matching import get_support_set, match_roi_features


def test_exact_roi_matching_works():
    source = pd.DataFrame({"roi": ["a", "b"], "coef": [1, -1], "t": [3, -2], "p": [0.01, 0.02], "q": [0.02, 0.04]})
    target = pd.DataFrame({"roi": ["b", "c"], "coef": [-2, 1], "t": [-3, 1], "p": [0.01, 0.5], "q": [0.02, 0.8]})
    matched, warnings = match_roi_features(source, target, "exact_roi")
    assert warnings == []
    assert matched["roi"].tolist() == ["b"]
    assert "source_coef" in matched


def test_group_aggregation_and_support_sets_work():
    stats = pd.DataFrame({
        "roi": ["a", "b", "c"],
        "roi_group": ["g1", "g1", "g2"],
        "coef": [1, 2, -1],
        "t": [3, 4, -2],
        "q": [0.01, 0.02, 0.2],
    })
    grouped = aggregate_roi_statistics(stats, "roi_group")
    support = get_support_set(stats, top_k=2)
    assert set(grouped["group_id"]) == {"g1", "g2"}
    assert support == {"a", "b"}

