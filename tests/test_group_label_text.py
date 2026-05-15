from scs_morph.claims.grouping import format_group_label, make_group_label


def test_cortical_group_label_uses_lobe():
    label = make_group_label({"lobe": "temporal", "roi_group": "temporal"}, "fs8_cortical_volume")
    assert format_group_label(label, "fs8_cortical_volume") == "temporal regions"


def test_subcortical_group_label_uses_roi_group():
    label = make_group_label({"roi_group": "medial_temporal"}, "fs8_subcortical_volume")
    assert format_group_label(label, "fs8_subcortical_volume") == "medial temporal structures"


def test_schaefer_group_label_uses_network():
    label = make_group_label({"network": "Default", "roi_group": "Default"}, "fs8_thickness_schaefer200_7")
    assert format_group_label(label, "fs8_thickness_schaefer200_7") == "Default network parcels"


def test_kong_group_label_formats_17_network_suffix():
    label = make_group_label({"network": "DefaultA", "network_7_like": "Default"}, "fs8_thickness_kong200_17")
    assert format_group_label(label, "fs8_thickness_kong200_17") == "Default A network parcels"


def test_yale_group_label_uses_code_group_language():
    label = make_group_label({"roi_group": "temporal_pole"}, "fs8_thickness_yale696")
    assert format_group_label(label, "fs8_thickness_yale696") == "Yale code-group temporal pole regions"


def test_missing_group_label_falls_back_to_unclassified():
    label = make_group_label({"network": "nan", "roi_group": None, "anatomical_group": "unknown"}, "fs8_thickness_schaefer200_7")
    assert format_group_label(label, "fs8_thickness_schaefer200_7") == "unclassified regions"
