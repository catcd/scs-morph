from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from scs_morph.claims.cards import ClaimCard, _to_native


@dataclass
class ClaimFamily:
    claim_family_id: str
    setting_id: str
    setting_metadata: dict
    composite_claims: list[ClaimCard] = field(default_factory=list)
    atomic_claims: list[ClaimCard] = field(default_factory=list)
    group_claims: list[ClaimCard] = field(default_factory=list)
    ranking_claims: list[ClaimCard] = field(default_factory=list)
    direction_claims: list[ClaimCard] = field(default_factory=list)
    latent_claims: list[ClaimCard] = field(default_factory=list)
    normative_claims: list[ClaimCard] = field(default_factory=list)
    roi_statistics_path: str | None = None
    extraction_summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_claims(self) -> list[ClaimCard]:
        return [
            *self.composite_claims,
            *self.atomic_claims,
            *self.group_claims,
            *self.ranking_claims,
            *self.direction_claims,
            *self.latent_claims,
            *self.normative_claims,
        ]

    def to_dict(self) -> dict[str, Any]:
        return _to_native(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimFamily":
        converted = dict(data)
        for field_name in [
            "composite_claims",
            "atomic_claims",
            "group_claims",
            "ranking_claims",
            "direction_claims",
            "latent_claims",
            "normative_claims",
        ]:
            converted[field_name] = [ClaimCard.from_dict(item) for item in converted.get(field_name, [])]
        return cls(**converted)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "ClaimFamily":
        return cls.from_dict(json.loads(text))

    def summary_row(self) -> dict[str, Any]:
        return {
            "claim_family_id": self.claim_family_id,
            "setting_id": self.setting_id,
            "dataset": self.setting_metadata.get("dataset"),
            "feature_space": self.setting_metadata.get("feature_space"),
            "claim_type": self.setting_metadata.get("claim_type"),
            "target": self.setting_metadata.get("target"),
            "n_composite": len(self.composite_claims),
            "n_atomic": len(self.atomic_claims),
            "n_group": len(self.group_claims),
            "n_ranking": len(self.ranking_claims),
            "n_direction": len(self.direction_claims),
            "n_latent": len(self.latent_claims),
            "n_normative": len(self.normative_claims),
            "n_total_claims": len(self.all_claims),
            "n_samples": self.extraction_summary.get("n_samples"),
            "n_features": self.extraction_summary.get("n_features"),
            "n_warnings": len(self.warnings),
        }

