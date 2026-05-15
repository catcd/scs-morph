import pandas as pd

from scs_morph.harmonization.adni import harmonize_adni
from scs_morph.harmonization.aibl import harmonize_aibl
from scs_morph.harmonization.oasis import harmonize_oasis1


def test_harmonize_adni():
    raw = pd.DataFrame(
        {
            "PTID": ["P1"],
            "COLPROT": ["ADNI1"],
            "AGE": [72],
            "PTGENDER": ["M"],
            "PTEDUCAT": [16],
            "PTETHCAT": ["Not Hispanic"],
            "PTRACCAT": ["White"],
            "PTMARRY": ["Married"],
            "FLDSTRENG": [3],
            "DX": ["AD"],
            "MMSE": [28],
            "ICV": [1500],
            "file_path": ["/P1/preproc/1/I123456/image.nii"],
        }
    )
    harmonized = harmonize_adni(raw)
    assert harmonized.loc[0, "dataset"] == "ADNI"
    assert harmonized.loc[0, "diagnosis_3class"] == "AD"
    assert harmonized.loc[0, "diagnosis_binary_ad_cn"] == 1
    assert harmonized.loc[0, "image_uid"] == "123456"


def test_harmonize_aibl():
    raw = pd.DataFrame(
        {
            "Image ID": ["A1"],
            "Sex": ["F"],
            "Age": [68],
            "Imaging Protocol": ["3T MPRAGE"],
            "DXNORM": [1],
            "DXMCI": [0],
            "DXAD": [0],
            "MMSCORE": [30],
        }
    )
    harmonized = harmonize_aibl(raw)
    assert harmonized.loc[0, "dataset"] == "AIBL"
    assert harmonized.loc[0, "diagnosis_3class"] == "CN"
    assert harmonized.loc[0, "field_strength"] == "3T"


def test_harmonize_oasis1():
    raw = pd.DataFrame(
        {"ID": ["O1"], "Age": [75], "M/F": ["M"], "Educ": [14], "MMSE": [29], "CDR": [0.5]}
    )
    harmonized = harmonize_oasis1(raw)
    assert harmonized.loc[0, "dataset"] == "OASIS1"
    assert harmonized.loc[0, "diagnosis_3class"] == "impaired_like"
    assert harmonized.loc[0, "diagnosis_binary_impairment_cdr"] == 1


def test_harmonize_adni_normalizes_raw_values():
    raw = pd.DataFrame(
        {
            "PTID": ["P1"],
            "COLPROT": ["ADNI1"],
            "AGE": [72],
            "PTGENDER": ["Male"],
            "PTEDUCAT": [16],
            "PTETHCAT": ["Not Hisp/Latino"],
            "PTRACCAT": ["White"],
            "PTMARRY": ["Married"],
            "FLDSTRENG": ["1.5 Tesla MRI"],
            "DX": ["AD"],
            "MMSE": [28],
            "ICV": [1500],
            "file_path": ["/011_S_0002/MPR__GradWarp__B1_Correction__N3__Scaled/2005-08-26_08_45_00.0/I35475/ADNI_image.nii"],
            "VISCODE": ["bl"],
        }
    )
    harmonized = harmonize_adni(raw)
    assert harmonized.loc[0, "sex"] == "M"
    assert harmonized.loc[0, "field_strength"] == "1.5T"
    assert harmonized.loc[0, "image_uid"] == "35475"
    assert harmonized.loc[0, "visit_code"] == "bl"


def test_harmonize_aibl_parses_protocol_field_strength():
    raw = pd.DataFrame(
        {
            "Image ID": [123.0],
            "Sex": ["F"],
            "Age": [68],
            "Imaging Protocol": [
                "Acquisition Plane=SAGITTAL;Mfg Model=TrioTim;Slice Thickness=1.2;Matrix Z=160.0;Acquisition Type=3D;Field Strength=3.0;Manufacturer=SIEMENS;Weighting=T1"
            ],
            "DXNORM": [1],
            "DXMCI": [0],
            "DXAD": [0],
            "MMSCORE": [30],
        }
    )
    harmonized = harmonize_aibl(raw)
    assert harmonized.loc[0, "subject_id"] == "123"
    assert harmonized.loc[0, "field_strength"] == "3T"
    assert harmonized.loc[0, "sex"] == "F"


def test_harmonize_oasis1_falls_back_to_etiv_for_icv():
    raw = pd.DataFrame(
        {"ID": ["O1"], "Age": [75], "M/F": ["F"], "Educ": [14], "MMSE": [29], "CDR": [0.5], "eTIV": [1500]}
    )
    harmonized = harmonize_oasis1(raw)
    assert harmonized.loc[0, "icv"] == 1500.0
    assert harmonized.loc[0, "sex"] == "F"
