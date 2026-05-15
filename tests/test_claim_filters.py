import pandas as pd

from scs_morph.claims.filters import (
    ClaimCriteria,
    composite_passes_criteria,
    direction_summary_passes_criteria,
    filter_atomic_rois,
    group_passes_criteria,
    normative_claim_passes_criteria,
    roi_passes_atomic_criteria,
)


def test_claim_filter_criteria():
    criteria = ClaimCriteria()
    row = {"q": 0.05, "p": 0.01, "t": 2.5, "coef": 0.3}
    assert roi_passes_atomic_criteria(row, criteria)
    df = pd.DataFrame({"roi": ["a", "b"], "q": [0.05, 0.2], "p": [0.01, 0.2], "t": [3.0, 1.0], "coef": [0.2, 0.1], "n": [40, 40]})
    assert len(filter_atomic_rois(df, criteria)) == 1
    passed, details = composite_passes_criteria(df, ["a", "b", "c"], criteria)
    assert "n_support_rois" in details
    assert isinstance(passed, bool)
    assert group_passes_criteria({"n_group_rois": 5, "n_support_rois": 2, "fraction_support": 0.4, "mean_abs_t": 2.0}, criteria)
    passed, _ = direction_summary_passes_criteria({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, criteria)
    assert passed
    assert normative_claim_passes_criteria({"n_support_rois": 3, "mean_abs_z": 1.2}, criteria)

