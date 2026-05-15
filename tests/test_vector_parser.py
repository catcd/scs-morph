import numpy as np
import pandas as pd

from scs_morph.data.vector_parser import parse_vector_cell, parse_vector_series


def test_parse_vector_cell_formats():
    assert np.array_equal(parse_vector_cell("[1, 2, 3]"), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(parse_vector_cell("1,2,3"), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(parse_vector_cell("[1 2 3]"), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(parse_vector_cell("array([1, 2, 3])"), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(parse_vector_cell("1; 2; 3"), np.array([1.0, 2.0, 3.0]))
    assert parse_vector_cell("").size == 0
    assert parse_vector_cell(None).size == 0


def test_parse_vector_cell_preserves_missing_positions():
    assert np.array_equal(parse_vector_cell("[1, nan, 3]"), np.array([1.0, -1.0, 3.0]))
    assert np.array_equal(parse_vector_cell("[1,,3]"), np.array([1.0, -1.0, 3.0]))
    assert np.array_equal(parse_vector_cell("[1 NA 3]"), np.array([1.0, -1.0, 3.0]))
    assert np.array_equal(parse_vector_cell("['NA', 2, 'NA']"), np.array([-1.0, 2.0, -1.0]))
    assert np.array_equal(parse_vector_cell([1, np.nan, 3]), np.array([1.0, -1.0, 3.0]))


def test_parse_vector_series_dimension_check():
    series = pd.Series(["[1,2]", "[3,4]"])
    parsed = parse_vector_series(series, column_name="test", expected_dim=2)
    assert len(parsed) == 2
    assert parsed[0].tolist() == [1.0, 2.0]
    assert parsed[1].tolist() == [3.0, 4.0]
