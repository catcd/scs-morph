import pandas as pd

from scs_morph.harmonization.normalizers import (
    normalize_field_strength,
    normalize_image_uid,
    normalize_numeric,
    normalize_sex,
    normalize_subject_id,
)


def test_normalize_sex_mappings():
    assert normalize_sex("Male") == "M"
    assert normalize_sex("female") == "F"
    assert normalize_sex("X") == "X"
    assert normalize_sex("2.0") == "F"
    assert normalize_sex(1) == "M"


def test_normalize_field_strength_from_text():
    assert normalize_field_strength("1.5 Tesla MRI") == "1.5T"
    assert normalize_field_strength("3 Tesla MRI") == "3T"
    assert normalize_field_strength("Field Strength=3.0") == "3T"
    assert normalize_field_strength("3T") == "3T"


def test_normalize_image_uid_and_subject_id():
    assert normalize_image_uid("I35475") == "35475"
    assert normalize_image_uid(35475.0) == "35475"
    assert normalize_subject_id(123.0) == "123"
    assert normalize_subject_id("456.0") == "456"


def test_normalize_numeric_handles_strings():
    assert normalize_numeric("72") == 72.0
    assert normalize_numeric("72.0") == 72.0
    assert pd.isna(normalize_numeric("unknown"))
