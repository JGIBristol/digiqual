import pytest
import pandas as pd
from digiqual.validation import validate_data

# 1. Test the "Happy Path" (Good Data)
def test_validate_data_success():
    # Create 15 rows of good data (mixed floats and ints)
    data = {
        'InputA': [1, 2, 3] * 5,
        'InputB': [10.5, 20.5, 30.5] * 5,
        'Signal': [0.1, 0.5, 0.9] * 5
    }
    df = pd.DataFrame(data)

    result = validate_data(df, input_cols=['InputA', 'InputB'], outcome_col='Signal')

    assert result['valid'] is True
    assert result['n_dropped'] == 0
    assert len(result['data']) == 15
    assert "Data validated" in result['message']

# 2. Test Missing Columns
def test_validate_data_missing_cols():
    df = pd.DataFrame({'InputA': [1]*15}) # Missing InputB and Signal

    result = validate_data(df, input_cols=['InputA', 'InputB'], outcome_col='Signal')

    assert result['valid'] is False
    assert result['message'] == "Missing columns."

# 3. Test Overlap Check (Outcome inside Inputs)
def test_validate_data_overlap():
    df = pd.DataFrame({'InputA': [1], 'Signal': [1]})

    # 'Signal' is listed as both input and outcome
    result = validate_data(df, input_cols=['InputA', 'Signal'], outcome_col='Signal')

    assert result['valid'] is False
    assert "Outcome variable cannot also be an Input" in result['message']

# 4. Test Numeric Cleaning (Text strings becoming NaNs)
def test_validate_data_non_numeric_drop():
    # Create 12 rows: 10 good, 2 bad (strings)
    df = pd.DataFrame({
        'InputA': [1]*10 + ["bad_text", "error"],
        'Signal': [5]*12
    })

    result = validate_data(df, input_cols=['InputA'], outcome_col='Signal')

    assert result['valid'] is True
    assert result['n_dropped'] == 2 # The 2 text rows should be gone
    assert len(result['data']) == 10

# 5. Test Negative/Zero Signal Drop
def test_validate_data_negative_signal():
    # Create 15 rows, but 5 have negative or zero signal
    df = pd.DataFrame({
        'InputA': [1]*15,
        'Signal': [10]*10 + [-5, 0, -1, 0, -10]
    })

    result = validate_data(df, input_cols=['InputA'], outcome_col='Signal')

    assert result['valid'] is True
    assert result['n_dropped'] == 5
    assert len(result['data']) == 10

# 6. Test "Too Few Rows" Failure
def test_validate_data_too_few_rows():
    # Only 5 rows total (below the <10 threshold)
    df = pd.DataFrame({
        'InputA': [1, 2, 3, 4, 5],
        'Signal': [1, 1, 1, 1, 1]
    })

    result = validate_data(df, input_cols=['InputA'], outcome_col='Signal')

    assert result['valid'] is False
    assert "Too few valid rows" in result['message']
