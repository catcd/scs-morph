from __future__ import annotations

from pathlib import Path

import pandas as pd

from scs_morph.utils.io import ensure_dir


def _infer_reviewer(path: Path) -> str:
    name = path.name
    for suffix in ["_expert_review", "_review"]:
        if suffix in name:
            return name.split(suffix)[0]
    return path.stem


def _read_returned_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def load_returned_reviews(returned_dir) -> pd.DataFrame:
    rows = []
    for path in Path(returned_dir).glob("*.csv"):
        df = _read_returned_csv(path)
        df["_source_file"] = path.name
        if "reviewer_id" not in df or df["reviewer_id"].fillna("").eq("").all():
            df["reviewer_id"] = _infer_reviewer(path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def validate_review_scores(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings = []
    cleaned = df.copy()
    scoring = config["scoring"]
    for col in ["review_id", "claim_id", "reviewer_id", *scoring["required_score_columns"]]:
        if col not in cleaned:
            cleaned[col] = pd.NA
            warnings.append({"warning_type": "missing_column", "column": col, "message": f"Missing column {col}", "n_affected": len(cleaned)})
    lo, hi = scoring["likert_min"], scoring["likert_max"]
    for col in scoring["required_score_columns"]:
        values = pd.to_numeric(cleaned[col], errors="coerce")
        invalid = cleaned[col].notna() & ~cleaned[col].astype(str).str.strip().eq("") & (values.isna() | (values < lo) | (values > hi))
        if invalid.any():
            warnings.append({"warning_type": "invalid_likert", "column": col, "message": f"Scores must be {lo}-{hi}", "n_affected": int(invalid.sum())})
        cleaned[col] = values.where(values.between(lo, hi))
    for col, allowed in scoring.get("categorical_columns", {}).items():
        if col not in cleaned:
            cleaned[col] = ""
        invalid = cleaned[col].notna() & ~cleaned[col].astype(str).str.strip().isin(["", *allowed])
        if invalid.any():
            warnings.append({"warning_type": "invalid_category", "column": col, "message": f"Allowed: {allowed}", "n_affected": int(invalid.sum())})
    duplicate = cleaned.duplicated(["reviewer_id", "review_id"], keep=False)
    if duplicate.any():
        warnings.append({"warning_type": "duplicate_review", "column": "reviewer_id+review_id", "message": "Duplicate reviewer/review rows", "n_affected": int(duplicate.sum())})

    long_rows = []
    for _, row in cleaned.iterrows():
        for dimension in scoring["required_score_columns"]:
            long_rows.append({
                "review_id": row.get("review_id"),
                "claim_id": row.get("claim_id"),
                "claim_family_id": row.get("claim_family_id"),
                "reviewer_id": row.get("reviewer_id"),
                "reviewer_type": row.get("reviewer_type"),
                "reviewer_specialty": row.get("reviewer_specialty"),
                "dimension": dimension,
                "score": row.get(dimension),
                "caution_level": row.get("caution_level"),
                "recommended_action": row.get("recommended_action"),
                "corrected_claim_text": row.get("corrected_claim_text"),
                "reviewer_comments": row.get("reviewer_comments"),
            })
    return pd.DataFrame(long_rows), pd.DataFrame(warnings)


def combine_reviews(returned_dir, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_returned_reviews(returned_dir)
    analyzed = ensure_dir(config["paths"]["analyzed_dir"])
    if raw.empty:
        empty_long = pd.DataFrame(columns=["review_id", "claim_id", "claim_family_id", "reviewer_id", "reviewer_type", "reviewer_specialty", "dimension", "score", "caution_level", "recommended_action", "corrected_claim_text", "reviewer_comments"])
        warnings = pd.DataFrame([{"warning_type": "no_returned_files", "column": "", "message": "No returned review files found.", "n_affected": 0}])
        empty_long.to_csv(Path(analyzed) / "expert_scores_long.csv", index=False)
        pd.DataFrame().to_csv(Path(analyzed) / "expert_scores_wide.csv", index=False)
        warnings.to_csv(Path(analyzed) / "expert_score_validation_warnings.csv", index=False)
        return empty_long, warnings
    long, warnings = validate_review_scores(raw, config)
    wide = long.pivot_table(index=["review_id", "claim_id", "claim_family_id", "reviewer_id", "reviewer_type", "reviewer_specialty"], columns="dimension", values="score", aggfunc="first").reset_index()
    long.to_csv(Path(analyzed) / "expert_scores_long.csv", index=False)
    wide.to_csv(Path(analyzed) / "expert_scores_wide.csv", index=False)
    warnings.to_csv(Path(analyzed) / "expert_score_validation_warnings.csv", index=False)
    return long, warnings
