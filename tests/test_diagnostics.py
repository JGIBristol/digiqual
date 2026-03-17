import pytest
import pandas as pd
from digiqual.diagnostics import (
    ValidationError,
    _check_input_coverage,
    sample_sufficiency,
    validate_simulation,
)
from unittest.mock import patch

def test_validate_simulation_not_dataframe():
    with pytest.raises(
        ValidationError, match="Input is not a valid pandas DataFrame or is empty"
    ):
        validate_simulation([], input_cols=['A'], outcome_col='B')

def test_validate_simulation_empty_dataframe():
    with pytest.raises(
        ValidationError, match="Input is not a valid pandas DataFrame or is empty"
    ):
        validate_simulation(pd.DataFrame(), input_cols=['A'], outcome_col='B')

def test_check_input_coverage_zero_range():
    # If data range is 0, max_gap_ratio = 1.0
    df = pd.DataFrame({'A': [5, 5, 5, 5]})
    res = _check_input_coverage(df, ['A'])
    assert res['A']['max_gap_ratio'] == 1.0
    assert not res['A']['sufficient_coverage']

def test_sample_sufficiency_drops_invalid():
    # Provide data that has 10 valid rows and 1 invalid row
    valid_data = {'A': [1]*10, 'B': [2]*10, 'Signal': [10]*10}
    df = pd.DataFrame(valid_data)
    df.loc[10] = ['bad', 'data', 'here']
    # This will drop the bad row and still have 10 valid so it proceeds
    res = sample_sufficiency(df, ['A', 'B'], 'Signal')
    assert not res.empty



@patch("digiqual.diagnostics.validate_simulation")
def test_sample_sufficiency_insufficient_data(mock_validate):
    # Mock validate_simulation to yield < 10 rows without throwing
    mock_validate.return_value = (
        pd.DataFrame({'A': [1]*5, 'B': [2]*5, 'Signal': [10]*5}),
        pd.DataFrame()
    )
    with pytest.raises(ValidationError, match="Insufficient valid data"):
        sample_sufficiency(pd.DataFrame(), ['A', 'B'], 'Signal')
