from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.data.loaders import load_raw_csv
from scs_morph.data.spm_matching import match_adni_spm
from scs_morph.harmonization.adni import harmonize_adni
from scs_morph.harmonization.aibl import harmonize_aibl
from scs_morph.harmonization.oasis import harmonize_oasis1
from scs_morph.utils.config import load_configs
from scs_morph.utils.io import ensure_dir, save_table
from scs_morph.utils.logging import get_logger


def _load_raw(path: Path, datasets: dict[str, str]) -> dict[str, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {}
    for dataset_id, file_name in datasets.items():
        raw[dataset_id] = load_raw_csv(str(path / file_name))
    return raw


def _save_dataset(df: pd.DataFrame, output_dir: Path, name: str) -> Path:
    ensure_dir(str(output_dir))
    return save_table(df, str(output_dir / name))


WARNING_COLUMNS = ["dataset", "warning_type", "column", "message", "n_affected", "example_values"]


def _counts_json(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "{}"
    counts = df[column].fillna("NA").astype(str).value_counts(dropna=False).to_dict()
    return json.dumps(counts, sort_keys=True)


def _summarize_harmonized_dataset(name: str, df: pd.DataFrame) -> dict:
    age = pd.to_numeric(df["age"], errors="coerce") if "age" in df.columns else pd.Series(dtype=float)
    return {
        "dataset": name,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique(dropna=True)) if "subject_id" in df.columns else 0,
        "diagnosis_3class_counts": _counts_json(df, "diagnosis_3class"),
        "sex_counts": _counts_json(df, "sex"),
        "field_strength_counts": _counts_json(df, "field_strength"),
        "cohort_counts": _counts_json(df, "cohort"),
        "age_min": float(age.min()) if len(age) and pd.notna(age.min()) else None,
        "age_max": float(age.max()) if len(age) and pd.notna(age.max()) else None,
        "mmse_missing": int(df["mmse"].isna().sum()) if "mmse" in df.columns else int(len(df)),
        "icv_missing": int(df["icv"].isna().sum()) if "icv" in df.columns else int(len(df)),
    }


def _collect_harmonization_warnings(name: str, df: pd.DataFrame) -> list[dict]:
    warnings: list[dict] = []
    important_columns = ["dataset", "cohort", "subject_id", "scan_id", "age", "sex", "diagnosis_3class", "mmse", "icv", "field_strength"]
    for column in important_columns:
        if column not in df.columns:
            warnings.append({
                "dataset": name,
                "warning_type": "missing_column",
                "column": column,
                "message": f"Expected harmonized column {column} is absent.",
                "n_affected": int(len(df)),
                "example_values": "",
            })
            continue
        missing = int(df[column].isna().sum())
        if missing:
            examples = df.loc[df[column].isna(), column].head(5).astype(str).tolist()
            warnings.append({
                "dataset": name,
                "warning_type": "missing_values",
                "column": column,
                "message": f"{missing} rows have missing {column}.",
                "n_affected": missing,
                "example_values": "; ".join(examples),
            })
    if "diagnosis_3class" in df.columns:
        other = int(df["diagnosis_3class"].astype(str).eq("OTHER").sum())
        if other:
            examples = df.loc[df["diagnosis_3class"].astype(str).eq("OTHER"), "diagnosis_raw"].head(5).astype(str).tolist() if "diagnosis_raw" in df.columns else []
            warnings.append({
                "dataset": name,
                "warning_type": "other_diagnosis",
                "column": "diagnosis_3class",
                "message": f"{other} rows map to OTHER diagnosis.",
                "n_affected": other,
                "example_values": "; ".join(examples),
            })
    return warnings


def _write_harmonization_reports(harmonized: dict[str, pd.DataFrame], output_dir: Path) -> None:
    ensure_dir(str(output_dir))
    report = pd.DataFrame([_summarize_harmonized_dataset(name, df) for name, df in harmonized.items()])
    report.to_csv(output_dir / "harmonization_report.csv", index=False)
    warnings: list[dict] = []
    for name, df in harmonized.items():
        warnings.extend(_collect_harmonization_warnings(name, df))
    pd.DataFrame(warnings, columns=WARNING_COLUMNS).to_csv(output_dir / "harmonization_warnings.csv", index=False)


def main() -> None:
    logger = get_logger("harmonize_data")
    config = load_configs("configs")
    paths = config["paths"]
    datasets = config["datasets"]
    raw_dir = Path(paths["raw_dir"])
    interim_dir = Path(paths["interim_dir"])

    raw_files = _load_raw(raw_dir, datasets)

    adni_raw = raw_files["ADNI"]
    aibl_raw = raw_files["AIBL"]
    oasis_raw = raw_files["OASIS1"]
    spm_raw = raw_files["ADNI_SPM"]

    adni_harmonized = harmonize_adni(adni_raw)
    aibl_harmonized = harmonize_aibl(aibl_raw)
    oasis_harmonized = harmonize_oasis1(oasis_raw)

    harmonized_dir = interim_dir / "harmonized"
    _save_dataset(adni_harmonized, harmonized_dir, "adni_harmonized")
    _save_dataset(aibl_harmonized, harmonized_dir, "aibl_harmonized")
    _save_dataset(oasis_harmonized, harmonized_dir, "oasis1_harmonized")
    _write_harmonization_reports(
        {"ADNI": adni_harmonized, "AIBL": aibl_harmonized, "OASIS1": oasis_harmonized},
        harmonized_dir,
    )

    matched_dir = interim_dir / "matched"
    adni_spm = match_adni_spm(adni_harmonized, spm_raw, join_type="inner")
    _save_dataset(adni_spm, matched_dir, "adni_spm_matched")

    for name, df in [
        ("ADNI", adni_harmonized),
        ("AIBL", aibl_harmonized),
        ("OASIS1", oasis_harmonized),
    ]:
        logger.info("%s harmonized rows: %d", name, len(df))
        counts = df.groupby(["cohort", "diagnosis_3class"], dropna=False).size()
        logger.info("%s counts by cohort and diagnosis:\n%s", name, counts.to_string())

    logger.info("SPM matched ADNI rows: %d", len(adni_spm))


if __name__ == "__main__":
    main()
