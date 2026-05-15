import pandas as pd

from scs_morph.expert.import_scores import combine_reviews


def _config(tmp_path):
    return {
        "paths": {"analyzed_dir": str(tmp_path / "analyzed")},
        "scoring": {
            "likert_min": 1,
            "likert_max": 5,
            "required_score_columns": ["plausibility_1_5", "clarity_1_5", "faithfulness_1_5", "usefulness_1_5", "reviewer_confidence_1_5"],
            "categorical_columns": {"caution_level": ["stable"], "recommended_action": ["accept"]},
        },
    }


def test_import_validates_likert_and_duplicates(tmp_path):
    returned = tmp_path / "returned"
    returned.mkdir()
    pd.DataFrame([
        {"review_id": "REV0001", "claim_id": "c1", "claim_family_id": "f1", "reviewer_id": "r1", "reviewer_type": "human", "plausibility_1_5": 6, "clarity_1_5": 5, "faithfulness_1_5": 4, "usefulness_1_5": 4, "reviewer_confidence_1_5": 5, "caution_level": "bad", "recommended_action": "accept"},
        {"review_id": "REV0001", "claim_id": "c1", "claim_family_id": "f1", "reviewer_id": "r1", "reviewer_type": "human", "plausibility_1_5": 4, "clarity_1_5": 5, "faithfulness_1_5": 4, "usefulness_1_5": 4, "reviewer_confidence_1_5": 5, "caution_level": "stable", "recommended_action": "accept"},
    ]).to_csv(returned / "r1_review_main_minimal_context.csv", index=False)
    long, warnings = combine_reviews(returned, _config(tmp_path))
    assert not long.empty
    assert {"invalid_likert", "invalid_category", "duplicate_review"}.issubset(set(warnings["warning_type"]))
    assert (tmp_path / "analyzed" / "expert_scores_wide.csv").exists()


def test_import_reads_excel_cp1252_csv(tmp_path):
    returned = tmp_path / "returned"
    returned.mkdir()
    path = returned / "geriatrics_expert_review_main_minimal_context.csv"
    content = (
        "review_id,claim_id,claim_family_id,reviewer_id,reviewer_type,"
        "plausibility_1_5,clarity_1_5,faithfulness_1_5,usefulness_1_5,"
        "reviewer_confidence_1_5,caution_level,recommended_action,reviewer_comments\n"
        'REV0001,c1,f1,,human,4,5,4,4,5,stable,accept,"Looks “reasonable”"\n'
    )
    path.write_bytes(content.encode("cp1252"))

    long, warnings = combine_reviews(returned, _config(tmp_path))

    assert not long.empty
    assert warnings.empty
    assert set(long["reviewer_id"]) == {"geriatrics"}
