from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.data.audit import audit_all_raw_datasets
from scs_morph.data.loaders import load_all_raw_datasets
from scs_morph.utils.config import load_configs
from scs_morph.utils.io import ensure_dir, save_table
from scs_morph.utils.logging import get_logger


def main() -> None:
    logger = get_logger("audit_data")
    config = load_configs("configs")
    paths = config["paths"]
    datasets = config["datasets"]

    raw_datasets = load_all_raw_datasets(paths, datasets)
    audit_report = audit_all_raw_datasets(raw_datasets)

    parsed_dir = Path(paths["interim_dir"]) / "parsed"
    ensure_dir(str(parsed_dir))
    save_path = str(parsed_dir / "raw_encoding_audit")
    saved_path = save_table(audit_report, save_path)

    logger.info("Raw encoding audit saved to %s", saved_path)
    logger.info("Audited datasets: %s", ", ".join(sorted(raw_datasets.keys())))


if __name__ == "__main__":
    main()
