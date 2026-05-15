from __future__ import annotations

from pathlib import Path
import pandas as pd

from scs_morph.utils.io import ensure_dir


def save_table(df: pd.DataFrame, path: str) -> Path:
    output = Path(path)
    ensure_dir(str(output.parent))
    df.to_csv(output, index=False)
    return output

