from pathlib import Path
from typing import List


def load_feature_names(path: str) -> List[str]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle.readlines()]
    return [name for name in names if name]
