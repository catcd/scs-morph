import re

import numpy as np
import pandas as pd


MISSING_VECTOR_VALUE = -1.0
MISSING_VALUES = {"", "nan", "none", "na", "n/a", "null", "'NA'", '"NA"'}


def _normalize_string(cell: str) -> str:
    text = cell.strip()
    if text.startswith("array(") and text.endswith(")"):
        text = text[6:-1]
    return text


def _is_missing_token(token: str) -> bool:
    return token.strip().lower() in MISSING_VALUES


def _parse_vector_token(token: str, missing_value: float) -> float:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1].strip()
    if _is_missing_token(token):
        return missing_value

    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"Unable to parse vector token: {token!r}") from exc


def _split_vector_tokens(text: str) -> list[str]:
    if "," in text or ";" in text:
        normalized = text.replace(";", ",")
        return [token.strip() for token in normalized.split(",")]
    return re.split(r"\s+", text.strip())


def parse_vector_cell(cell, missing_value: float = MISSING_VECTOR_VALUE) -> np.ndarray:
    """Parse a single vector cell into a 1D numpy float array."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        array = np.asarray(cell, dtype=float).ravel()
        array = np.nan_to_num(array, nan=missing_value)
        return array

    if isinstance(cell, pd.Series):
        return parse_vector_cell(cell.item(), missing_value=missing_value)

    if isinstance(cell, str):
        text = _normalize_string(cell)
        if _is_missing_token(text):
            return np.array([], dtype=float)

        text = text.strip("[]() ")
        text = re.sub(r"\s+", " ", text)
        if text == "":
            return np.array([], dtype=float)

        tokens = _split_vector_tokens(text)
        if not tokens:
            raise ValueError(f"Unable to parse vector string: {cell!r}")

        try:
            return np.asarray(
                [_parse_vector_token(token, missing_value) for token in tokens],
                dtype=float,
            )
        except ValueError as exc:
            raise ValueError(f"Unable to parse vector string {cell!r}: {exc}") from exc

    if pd.isna(cell):
        return np.array([], dtype=float)

    if isinstance(cell, (int, float, np.floating, np.integer)):
        return np.asarray([float(cell)], dtype=float)

    raise ValueError(f"Unable to parse vector cell of type {type(cell).__name__}: {cell}")


def parse_vector_series(
    series: pd.Series,
    column_name: str | None = None,
    expected_dim: int | None = None,
    missing_value: float = MISSING_VECTOR_VALUE,
) -> list[np.ndarray]:
    """Parse a pandas Series of vector cells, validating dimension if requested."""
    parsed: list[np.ndarray] = []
    for idx, cell in series.items():
        name = f" '{column_name}'" if column_name else ""
        try:
            array = parse_vector_cell(cell, missing_value=missing_value)
        except ValueError as exc:
            raise ValueError(f"Error parsing row {idx}{name}: {exc}") from exc

        if expected_dim is not None and array.size > 0 and array.shape[0] != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch for row {idx}{name}: got {array.shape[0]}, expected {expected_dim}"
            )

        parsed.append(array)

    return parsed
