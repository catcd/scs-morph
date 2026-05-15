import pandas as pd

from scs_morph.expert.export import create_reviewer_table


def test_reviewer_table_context_and_blank_scores():
    df = pd.DataFrame([{"claim_id": "c1", "claim_family_id": "f1", "review_set": "main", "claim_level": "composite", "claim_type": "mmse_association", "claim_text": "A claim text.", "overall_label": "fragile", "stability_label": "fragile", "transfer_label": "weak"}])
    full = create_reviewer_table(df, "r1", "human", "neuro", context="full")
    minimal = create_reviewer_table(df, "r1", "human", "neuro", context="minimal")
    assert "overall_label" in full.columns
    assert "overall_label" not in minimal.columns
    assert full.loc[0, "plausibility_1_5"] == ""
    assert "review_id" in full.columns

