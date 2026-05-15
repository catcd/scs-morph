from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs_morph.claims.family import ClaimFamily
from scs_morph.utils.io import read_table


def load_claim_families(path: str = "data/processed/claim_cards/claim_families.jsonl") -> list[ClaimFamily]:
    families: list[ClaimFamily] = []
    input_path = Path(path)
    if not input_path.exists():
        return families
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                families.append(ClaimFamily.from_json(line))
    return families


def load_claim_cards_summary(path: str = "data/processed/claim_cards/claim_cards_summary.csv") -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def load_claim_family_summary(path: str = "data/processed/claim_cards/claim_family_summary.csv") -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def load_roi_statistics_for_setting(setting_id: str, base_dir: str = "data/processed/claim_cards/per_setting") -> pd.DataFrame | None:
    path = Path(base_dir) / f"{setting_id}_roi_statistics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_existing_table(folder: Path, stem: str) -> pd.DataFrame:
    for suffix in [".parquet", ".csv"]:
        path = folder / f"{stem}{suffix}"
        if path.exists():
            return read_table(str(path))
    raise FileNotFoundError(f"Missing {stem}.parquet/csv in {folder}")


def load_feature_matrix(dataset: str, feature_space: str, base_dir: str = "data/processed/feature_matrices") -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path(base_dir) / dataset / feature_space
    return _read_existing_table(folder, "X"), _read_existing_table(folder, "meta")


def find_claim_family(
    families: list[ClaimFamily],
    setting_id: str | None = None,
    dataset: str | None = None,
    feature_space: str | None = None,
    claim_type: str | None = None,
    target: str | None = None,
) -> ClaimFamily | None:
    if setting_id:
        for family in families:
            if family.setting_id == setting_id:
                return family
    matches: list[ClaimFamily] = []
    for family in families:
        meta = family.setting_metadata
        if dataset is not None and meta.get("dataset") != dataset:
            continue
        if feature_space is not None and meta.get("feature_space") != feature_space:
            continue
        if claim_type is not None and meta.get("claim_type") != claim_type:
            continue
        if target is not None and meta.get("target") != target:
            continue
        matches.append(family)
    return matches[0] if matches else None

