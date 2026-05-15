"""Scientific claim extraction primitives."""

from scs_morph.claims.base import BaseClaimExtractor, get_extractor_for_claim_type
from scs_morph.claims.cards import ClaimCard
from scs_morph.claims.family import ClaimFamily

__all__ = ["BaseClaimExtractor", "ClaimCard", "ClaimFamily", "get_extractor_for_claim_type"]
