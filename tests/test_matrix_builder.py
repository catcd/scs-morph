import pandas as pd

from scs_morph.features.feature_space import FeatureSpace
from scs_morph.features.matrix_builder import build_feature_matrix


def test_build_feature_matrix(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    feature_names_file = raw_dir / "example_feats.txt"
    feature_names_file.write_text("region1\nregion2\nregion3\n", encoding="utf-8")

    df = pd.DataFrame(
        {
            "scan_id": ["s1", "s2"],
            "cortical_volume": ["[1,2,3]", "[4,5,6]"],
            "subject_id": ["subj1", "subj2"],
        }
    )
    feature_space = FeatureSpace(
        name="test_space",
        dataset_column="cortical_volume",
        feature_names_file="example_feats.txt",
        family="volume",
        pipeline="freesurfer8",
        atlas="cortical_regions",
    )

    X, meta = build_feature_matrix(df, feature_space, str(raw_dir), index_col="scan_id")
    assert X.shape == (2, 3)
    assert list(X.columns) == ["region1", "region2", "region3"]
    assert meta.loc["s1", "subject_id"] == "subj1"


def test_build_feature_matrix_keeps_missing_roi_alignment(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    feature_names_file = raw_dir / "example_feats.txt"
    feature_names_file.write_text("region1\nregion2\nregion3\n", encoding="utf-8")

    df = pd.DataFrame(
        {
            "scan_id": ["s1"],
            "cortical_volume": ["[1,nan,3]"],
            "subject_id": ["subj1"],
        }
    )
    feature_space = FeatureSpace(
        name="test_space",
        dataset_column="cortical_volume",
        feature_names_file="example_feats.txt",
        family="volume",
        pipeline="freesurfer8",
        atlas="cortical_regions",
    )

    X, _ = build_feature_matrix(df, feature_space, str(raw_dir), index_col="scan_id")
    assert X.loc["s1", "region2"] == -1.0


def test_build_feature_matrix_treats_bad_dimension_as_all_missing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    feature_names_file = raw_dir / "example_feats.txt"
    feature_names_file.write_text("region1\nregion2\nregion3\n", encoding="utf-8")

    df = pd.DataFrame(
        {
            "scan_id": ["s1"],
            "cortical_volume": ["[1,2]"],
            "subject_id": ["subj1"],
        }
    )
    feature_space = FeatureSpace(
        name="test_space",
        dataset_column="cortical_volume",
        feature_names_file="example_feats.txt",
        family="volume",
        pipeline="freesurfer8",
        atlas="cortical_regions",
    )

    X, meta = build_feature_matrix(df, feature_space, str(raw_dir), index_col="scan_id")
    assert X.loc["s1"].tolist() == [-1.0, -1.0, -1.0]
    assert meta.loc["s1", "test_space_vector_status"] == "dimension_mismatch_all_missing:2->3"
