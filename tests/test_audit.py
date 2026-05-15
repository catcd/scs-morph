import pandas as pd

from scs_morph.data.audit import audit_dataset_encodings


def test_audit_dataset_encodings_reports_top_values():
    df = pd.DataFrame(
        {
            "A": [1, 2, 2, None],
            "B": ["x", "x", "y", ""],
        }
    )
    report = audit_dataset_encodings("TEST", df, max_values=2)
    assert report.loc[report["column"] == "A", "non_missing"].iloc[0] == 3
    assert report.loc[report["column"] == "B", "top_values"].iloc[0].startswith("x:2")
