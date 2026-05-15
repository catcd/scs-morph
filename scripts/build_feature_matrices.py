from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.features.feature_space import load_feature_spaces
from scs_morph.features.matrix_builder import build_feature_matrix, save_feature_matrix
from scs_morph.harmonization.groups import select_first_scan_per_subject
from scs_morph.utils.config import load_configs
from scs_morph.utils.io import read_table
from scs_morph.utils.logging import get_logger


FS8_SCHAEFER_MATCHED_ALIAS = "schaefer200_7network"
FS8_SCHAEFER_EXPECTED_COLUMN = "schafer200_7network"


def _load_harmonized_table(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").exists():
        return read_table(str(path.with_suffix(".parquet")))
    return read_table(str(path.with_suffix(".csv")))


def _prepare_adni_spm_matched_fs8_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Expose FS8 Schaefer vectors in the matched table under the configured column name."""
    if FS8_SCHAEFER_EXPECTED_COLUMN in df.columns:
        logger.info("ADNI_SPM_MATCHED FS8 strategy: direct from matched table column %s", FS8_SCHAEFER_EXPECTED_COLUMN)
        print(f"ADNI_SPM_MATCHED FS8 strategy: direct from matched table column {FS8_SCHAEFER_EXPECTED_COLUMN}")
        return df
    if FS8_SCHAEFER_MATCHED_ALIAS in df.columns:
        out = df.copy()
        out[FS8_SCHAEFER_EXPECTED_COLUMN] = out[FS8_SCHAEFER_MATCHED_ALIAS]
        logger.info(
            "ADNI_SPM_MATCHED FS8 strategy: direct from matched table alias %s -> %s",
            FS8_SCHAEFER_MATCHED_ALIAS,
            FS8_SCHAEFER_EXPECTED_COLUMN,
        )
        print(f"ADNI_SPM_MATCHED FS8 strategy: direct from matched table alias {FS8_SCHAEFER_MATCHED_ALIAS} -> {FS8_SCHAEFER_EXPECTED_COLUMN}")
        return out
    logger.info("ADNI_SPM_MATCHED FS8 strategy: no raw FS8 Schaefer vector column found in matched table")
    print("ADNI_SPM_MATCHED FS8 strategy: no raw FS8 Schaefer vector column found in matched table")
    return df


def match_feature_matrix_to_metadata(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    matched_meta: pd.DataFrame,
    uid_col: str = "image_uid",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align an existing feature matrix to matched metadata by image UID."""
    warnings: list[str] = []
    if uid_col not in meta.columns or uid_col not in matched_meta.columns:
        raise KeyError(f"{uid_col} must exist in both metadata tables.")
    source = meta.copy()
    source["_match_uid"] = source[uid_col].astype(str).str.replace(r"^I", "", regex=True)
    target = matched_meta.copy()
    target["_match_uid"] = target[uid_col].astype(str).str.replace(r"^I", "", regex=True)
    duplicate_mask = source["_match_uid"].duplicated(keep="first")
    if duplicate_mask.any():
        warnings.append(f"Dropped {int(duplicate_mask.sum())} duplicate source image_uid rows.")
    source = source.loc[~duplicate_mask]
    source_lookup = source.reset_index().set_index("_match_uid")
    keep_rows = []
    keep_meta = []
    missing = 0
    for _, row in target.iterrows():
        uid = row["_match_uid"]
        if uid not in source_lookup.index:
            missing += 1
            continue
        source_index = source_lookup.loc[uid, "index"]
        keep_rows.append(X.loc[source_index])
        keep_meta.append(row.drop(labels=["_match_uid"]).to_dict())
    if missing:
        warnings.append(f"Dropped {missing} matched metadata rows without source FS8 image_uid.")
    X_out = pd.DataFrame(keep_rows).reset_index(drop=True)
    meta_out = pd.DataFrame(keep_meta).reset_index(drop=True)
    return X_out, meta_out, warnings


def main() -> None:
    logger = get_logger("build_feature_matrices")
    config = load_configs("configs")
    paths = config["paths"]
    feature_spaces = load_feature_spaces("configs/feature_spaces.yaml")
    interim_dir = Path(paths["interim_dir"])
    processed_dir = Path(paths["processed_dir"])
    raw_dir = paths["raw_dir"]

    harmonized_files = {
        "ADNI": interim_dir / "harmonized" / "adni_harmonized",
        "AIBL": interim_dir / "harmonized" / "aibl_harmonized",
        "OASIS1": interim_dir / "harmonized" / "oasis1_harmonized",
        "ADNI_SPM": interim_dir / "matched" / "adni_spm_matched",
    }

    for dataset_name, base_path in harmonized_files.items():
        try:
            df = _load_harmonized_table(base_path)
        except Exception as exc:
            logger.warning("Skipping %s because harmonized file not found: %s", dataset_name, exc)
            continue
        if dataset_name == "ADNI_SPM":
            df = _prepare_adni_spm_matched_fs8_columns(df, logger)
            dataset_name = "ADNI_SPM_MATCHED"

        for feature_space_name, feature_space in feature_spaces.items():
            if feature_space.dataset_column not in df.columns:
                continue

            X, meta = build_feature_matrix(
                df,
                feature_space,
                raw_dir,
                index_col="scan_id",
                drop_missing=True,
            )
            save_feature_matrix(X, meta, str(processed_dir / "feature_matrices"), dataset_name, feature_space_name)
            logger.info(
                "Saved feature matrix for %s %s: %d rows, %d features",
                dataset_name,
                feature_space_name,
                X.shape[0],
                X.shape[1],
            )
            if dataset_name == "ADNI_SPM_MATCHED" and feature_space_name in {"spm_thickness_schaefer200_7", "spm_thickness_schaefer200_17", "fs8_thickness_schaefer200_7"}:
                print(
                    f"ADNI_SPM_MATCHED {feature_space_name}: "
                    f"SPM matched rows={len(df)}, rows saved={X.shape[0]}, features={X.shape[1]}, "
                    f"match rate={X.shape[0] / len(df):.3f}"
                )

        baseline = select_first_scan_per_subject(df)
        baseline_name = f"{dataset_name}_baseline"
        for feature_space_name, feature_space in feature_spaces.items():
            if feature_space.dataset_column not in baseline.columns:
                continue

            X, meta = build_feature_matrix(
                baseline,
                feature_space,
                raw_dir,
                index_col="scan_id",
                drop_missing=True,
            )
            save_feature_matrix(
                X,
                meta,
                str(processed_dir / "feature_matrices"),
                baseline_name,
                feature_space_name,
            )
            logger.info(
                "Saved baseline feature matrix for %s %s: %d rows, %d features",
                baseline_name,
                feature_space_name,
                X.shape[0],
                X.shape[1],
            )


if __name__ == "__main__":
    main()
