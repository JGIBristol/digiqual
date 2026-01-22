import pytest
import pandas as pd
from digiqual.diagnostics import validate_simulation, ValidationError

# 1. Test the "Happy Path" (Good Data)
def test_validate_simulation_success():
    # Create 15 rows of good data
    data = {
        'InputA': [1, 2, 3] * 5,
        'InputB': [10.5, 20.5, 30.5] * 5,
        'Signal': [0.1, 0.5, 0.9] * 5
    }
    df = pd.DataFrame(data)

    # New API: Returns tuple (clean, removed)
    df_clean, df_removed = validate_simulation(df, input_cols=['InputA', 'InputB'], outcome_col='Signal')

    # Assertions
    assert len(df_clean) == 15
    assert df_removed.empty  # Should be empty for perfect data
    assert list(df_clean.columns) == ['InputA', 'InputB', 'Signal']

# 2. Test Missing Columns (Expect Error)
def test_validate_simulation_missing_cols():
    df = pd.DataFrame({'InputA': [1]*15}) # Missing InputB and Signal

    # We expect the function to raise a ValidationError
    with pytest.raises(ValidationError, match="Missing required columns"):
        validate_simulation(df, input_cols=['InputA', 'InputB'], outcome_col='Signal')

# 3. Test Overlap Check (Expect Error)
def test_validate_simulation_overlap():
    df = pd.DataFrame({'InputA': [1]*15, 'Signal': [1]*15})

    # 'Signal' is listed as both input and outcome
    with pytest.raises(ValidationError, match="Outcome variable.*cannot also be an Input"):
        validate_simulation(df, input_cols=['InputA', 'Signal'], outcome_col='Signal')

# 4. Test Numeric Cleaning (Text strings becoming NaNs)
def test_validate_simulation_non_numeric_drop():
    # Create 12 rows: 10 good, 2 bad (strings)
    df = pd.DataFrame({
        'InputA': [1]*10 + ["bad_text", "error"],
        'Signal': [5]*12
    })

    df_clean, df_removed = validate_simulation(df, input_cols=['InputA'], outcome_col='Signal')

    assert len(df_clean) == 10
    assert len(df_removed) == 2
    # Verify the bad data is preserved in df_removed
    assert "bad_text" in df_removed['InputA'].values

# 5. Test Negative/Zero Signal Drop
def test_validate_simulation_negative_signal():
    # Create 15 rows: 10 good, 5 bad (negative/zero)
    df = pd.DataFrame({
        'InputA': [1]*15,
        'Signal': [10]*10 + [-5, 0, -1, 0, -10]
    })

    df_clean, df_removed = validate_simulation(df, input_cols=['InputA'], outcome_col='Signal')

    assert len(df_clean) == 10
    assert len(df_removed) == 5
    assert (df_clean['Signal'] > 0).all() # Verify all remaining signals are positive

# 6. Test "Too Few Rows" (Expect Error)
def test_validate_simulation_too_few_rows():
    # Only 5 rows total (below the <10 threshold)
    df = pd.DataFrame({
        'InputA': [1, 2, 3, 4, 5],
        'Signal': [1, 1, 1, 1, 1]
    })

    # This is now a critical error, not just a "valid: False" flag
    with pytest.raises(ValidationError, match="Too few valid rows"):
        validate_simulation(df, input_cols=['InputA'], outcome_col='Signal')
