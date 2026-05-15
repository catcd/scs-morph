from scs_morph.claims.text_generation import validate_claim_text


def test_validate_claim_text_catches_nan():
    assert "contains nan" in validate_claim_text("Effects are concentrated in nan, where 3/4 regions pass.")


def test_validate_claim_text_catches_none():
    assert "contains None" in validate_claim_text("Effects are concentrated in None.")


def test_validate_claim_text_catches_unknown_network():
    assert "contains unknown_network" in validate_claim_text("Effects are concentrated in unknown_network.")


def test_validate_claim_text_catches_unresolved_braces():
    assert "contains unresolved braces" in validate_claim_text("The claim uses {group_label}.")
