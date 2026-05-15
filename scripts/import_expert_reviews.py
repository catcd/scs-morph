from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.expert.import_scores import combine_reviews
from scs_morph.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expert_review.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    long, warnings = combine_reviews(config["paths"]["returned_dir"], config)
    if long.empty:
        print("No returned review files found. Run export_expert_review.py and collect completed review CSVs first.")
    print(f"Score rows: {len(long)}")
    print(f"Validation warnings: {len(warnings)}")


if __name__ == "__main__":
    main()

