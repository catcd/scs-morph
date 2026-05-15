from scs_morph.claims.templates import CLAIM_TEMPLATES


def test_all_templates_render_without_unresolved_braces():
    assert len(CLAIM_TEMPLATES) == 40
    values = {
        "positive_label": "AD",
        "negative_label": "CN",
        "direction": "lower",
        "feature_family_label": "volume",
        "region_summary": "temporal regions",
        "covariate_text": "age, sex",
        "roi": "Left-Hippocampus",
        "group_label": "medial temporal",
        "n_support_rois": 3,
        "n_group_rois": 5,
        "top_regions": "a, b, c",
        "target_label": "Higher MMSE",
        "fraction_pct": 80,
        "explained_variance_pct": 12.5,
        "metadata_variable": "age",
        "association_value": 0.3,
        "target_group": "AD",
        "reference_group": "CN",
    }
    for template in CLAIM_TEMPLATES.values():
        text = template.format(**values)
        assert "{" not in text and "}" not in text
        assert "nan" not in text.lower()
