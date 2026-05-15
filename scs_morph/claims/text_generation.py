from __future__ import annotations

import numpy as np

from scs_morph.claims.templates import CLAIM_TEMPLATES


def direction_from_effect(effect_value, feature_family_label=None) -> str:
    try:
        value = float(effect_value)
    except (TypeError, ValueError):
        return "altered"
    if not np.isfinite(value) or value == 0:
        return "altered"
    return "lower" if value < 0 else "higher"


def composite_direction(support_effects: dict[str, float]) -> str:
    values = np.asarray([v for v in support_effects.values() if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return "altered"
    if float((values < 0).mean()) >= 0.70:
        return "lower"
    if float((values > 0).mean()) >= 0.70:
        return "higher"
    return "altered"


def target_label_for_claim(claim_type: str, target: str | None, positive_label=None, negative_label=None) -> str:
    if claim_type == "mmse_association":
        return "Higher MMSE"
    if claim_type == "age_association":
        return "older age"
    if claim_type == "cdr_impairment_contrast":
        return "CDR-defined cognitive impairment"
    if positive_label and negative_label:
        return f"{positive_label} versus {negative_label}"
    return str(target or "target")


def covariate_text(covariates: list[str]) -> str:
    return ", ".join(covariates) if covariates else "no covariates"


def render_template(template_id: str, **kwargs) -> str:
    text = CLAIM_TEMPLATES[template_id].format(**kwargs)
    banned = ["biomarker", "causes", "diagnoses"]
    lowered = text.lower()
    if any(word in lowered for word in banned):
        raise ValueError(f"Generated claim text contains banned language: {text}")
    return text


def validate_claim_text(claim_text: str) -> list[str]:
    """Return quality warnings for unsafe or unresolved claim text."""
    warnings: list[str] = []
    if claim_text is None or not str(claim_text).strip():
        return ["empty claim text"]
    text = str(claim_text)
    lowered = text.lower()
    if " nan" in lowered or "nan," in lowered or "nan." in lowered or lowered.strip() == "nan":
        warnings.append("contains nan")
    if "none" in lowered:
        warnings.append("contains None")
    if "unknown_network" in text:
        warnings.append("contains unknown_network")
    if "unknown_subregion" in text:
        warnings.append("contains unknown_subregion")
    if "{" in text or "}" in text:
        warnings.append("contains unresolved braces")
    return warnings


def lower_mmse_interpretation(coef: float | None) -> str | None:
    if coef is None or not np.isfinite(coef):
        return None
    if coef > 0:
        return "Lower MMSE corresponds to lower morphometry"
    if coef < 0:
        return "Lower MMSE corresponds to higher morphometry"
    return "Lower MMSE corresponds to altered morphometry"
