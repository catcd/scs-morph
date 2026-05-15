import pandas as pd

from scripts.build_feature_matrices import match_feature_matrix_to_metadata


def test_match_feature_matrix_to_metadata_preserves_matched_order_and_drops_missing():
    X = pd.DataFrame({"roi1": [1.0, 2.0, 3.0], "roi2": [4.0, 5.0, 6.0]}, index=["a", "b", "c"])
    meta = pd.DataFrame({"image_uid": ["101", "102", "103"]}, index=["a", "b", "c"])
    matched_meta = pd.DataFrame({"image_uid": ["103", "999", "101"], "scan_id": ["103", "999", "101"]})

    X_out, meta_out, warnings = match_feature_matrix_to_metadata(X, meta, matched_meta)

    assert X_out["roi1"].tolist() == [3.0, 1.0]
    assert meta_out["scan_id"].tolist() == ["103", "101"]
    assert any("without source FS8 image_uid" in warning for warning in warnings)

