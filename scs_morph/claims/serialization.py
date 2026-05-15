from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs_morph.claims.cards import ClaimCard
from scs_morph.claims.family import ClaimFamily
from scs_morph.claims.text_generation import validate_claim_text
from scs_morph.utils.io import ensure_dir


def _validate_card_text(card: ClaimCard) -> ClaimCard:
    warnings = validate_claim_text(card.claim_text)
    if warnings:
        existing = set(card.notes)
        card.notes.extend([warning for warning in warnings if warning not in existing])
        if card.criteria_passed and card.status == "supported":
            card.status = "weak_support"
    return card


def _validate_family_text(family: ClaimFamily) -> ClaimFamily:
    for card in family.all_claims:
        _validate_card_text(card)
    return family


def save_claim_cards_jsonl(claim_cards: list[ClaimCard], path: str) -> Path:
    output = Path(path)
    ensure_dir(str(output.parent))
    with output.open("w", encoding="utf-8") as handle:
        for card in claim_cards:
            _validate_card_text(card)
            handle.write(card.to_json() + "\n")
    return output


def load_claim_cards_jsonl(path: str) -> list[ClaimCard]:
    cards: list[ClaimCard] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cards.append(ClaimCard.from_json(line))
    return cards


def save_claim_families_jsonl(claim_families: list[ClaimFamily], path: str) -> Path:
    output = Path(path)
    ensure_dir(str(output.parent))
    with output.open("w", encoding="utf-8") as handle:
        for family in claim_families:
            _validate_family_text(family)
            handle.write(family.to_json() + "\n")
    return output


def load_claim_families_jsonl(path: str) -> list[ClaimFamily]:
    families: list[ClaimFamily] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                families.append(ClaimFamily.from_json(line))
    return families


def claim_cards_to_dataframe(claim_cards: list[ClaimCard]) -> pd.DataFrame:
    rows = []
    for card in claim_cards:
        _validate_card_text(card)
        rows.append(
            {
                "claim_id": card.claim_id,
                "claim_family_id": card.claim_family_id,
                "claim_level": card.claim_level,
                "claim_type": card.claim_type,
                "dataset": card.dataset,
                "cohort": card.cohort,
                "feature_space": card.feature_space,
                "feature_family": card.feature_family,
                "pipeline": card.pipeline,
                "atlas": card.atlas,
                "target": card.target,
                "covariates": "; ".join(card.covariates),
                "n_samples": card.n_samples,
                "n_features": card.n_features,
                "roi": card.roi,
                "hemisphere": card.hemisphere,
                "roi_group": card.roi_group,
                "anatomical_group": card.anatomical_group,
                "network": card.network,
                "network_7_like": card.machine_readable_statement.get("network_7_like"),
                "lobe": card.lobe,
                "direction": card.direction,
                "evidence_metric": card.evidence_metric,
                "evidence_value": card.evidence_value,
                "coefficient": card.coefficient,
                "t_value": card.t_value,
                "p_value": card.p_value,
                "q_value": card.q_value,
                "claim_text": card.claim_text,
                "short_claim_text": card.short_claim_text,
                "criteria_passed": card.criteria_passed,
                "status": card.status,
            }
        )
    return pd.DataFrame(rows)


def claim_families_to_dataframe(claim_families: list[ClaimFamily]) -> pd.DataFrame:
    return pd.DataFrame([family.summary_row() for family in claim_families])


def save_claim_cards_summary_csv(claim_cards: list[ClaimCard], path: str) -> Path:
    output = Path(path)
    ensure_dir(str(output.parent))
    claim_cards_to_dataframe(claim_cards).to_csv(output, index=False)
    return output


def save_claim_family_summary_csv(claim_families: list[ClaimFamily], path: str) -> Path:
    output = Path(path)
    ensure_dir(str(output.parent))
    claim_families_to_dataframe(claim_families).to_csv(output, index=False)
    return output
