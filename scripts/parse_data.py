from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.data.loaders import load_all_raw_datasets
from scs_morph.data.validators import validate_all_feature_spaces
from scs_morph.utils.config import load_configs
from scs_morph.utils.io import ensure_dir, save_table
from scs_morph.utils.logging import get_logger


def main() -> None:
    logger = get_logger("parse_data")
    config = load_configs("configs")
    paths = config["paths"]
    datasets = config["datasets"]
    feature_spaces = config["feature_spaces"]

    raw_datasets = load_all_raw_datasets(paths, datasets)
    report = validate_all_feature_spaces(raw_datasets, feature_spaces, paths["raw_dir"])

    output_dir = Path(paths["interim_dir"]) / "parsed"
    ensure_dir(str(output_dir))
    save_path = str(output_dir / "feature_validation_report")
    saved_path = save_table(report, save_path)

    logger.info("Feature validation report saved to %s", saved_path)
    logger.info("Parsed datasets: %s", ", ".join(sorted(raw_datasets.keys())))
    logger.info(
        "Validation summary: %d rows, %d feature spaces checked",
        len(report),
        report["feature_space"].nunique(),
    )


if __name__ == "__main__":
    main()
