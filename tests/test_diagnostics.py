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
    assert res['A']['max_gap_ratio'] == 0.0
    assert res['A']['sufficient_coverage']

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


def test_check_collinearity_no_correlation():
    # 2 independent variables
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
    })
    from digiqual.diagnostics import _check_collinearity
    vifs = _check_collinearity(df, ['A', 'B'])
    assert vifs['A'] < 2.0
    assert vifs['B'] < 2.0


def test_check_collinearity_perfect_correlation():
    # B is exactly 2 * A (perfect collinearity)
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    })
    from digiqual.diagnostics import _check_collinearity
    vifs = _check_collinearity(df, ['A', 'B'])
    assert vifs['A'] == float('inf')
    assert vifs['B'] == float('inf')


def test_sample_sufficiency_collinearity_integration():
    # 2 highly correlated variables, check if sample_sufficiency fails the check
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [1.01, 2.02, 3.01, 4.02, 5.01, 6.02, 7.01, 8.02, 9.01, 10.02],
        'Signal': [2.1, 4.0, 6.2, 8.1, 9.9, 12.0, 14.1, 15.9, 18.2, 20.0]
    })
    res = sample_sufficiency(df, ['A', 'B'], 'Signal', max_allowed_vif=5.0)
    collinearity_rows = res[res['Test'] == 'Collinearity Check']
    assert len(collinearity_rows) == 2
    # They should fail because they are extremely correlated (VIF > 5.0)
    assert not collinearity_rows['Pass'].all()

