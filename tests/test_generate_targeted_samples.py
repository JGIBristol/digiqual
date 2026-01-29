import pytest
import pandas as pd
from unittest.mock import patch

# Import the functions to test
# We import the private functions (_) to test their logic in isolation
from digiqual.adaptive import (
    generate_targeted_samples,
    _fill_gaps,
    _sample_uncertainty
)

# --- Fixtures ---

@pytest.fixture
def basic_df():
    """Creates a simple, clean dataset for testing bounds and gaps."""
    return pd.DataFrame({
        'Length': [0.0, 1.0, 2.0, 8.0, 9.0, 10.0], # Explicit gap between 2.0 and 8.0
        'Angle': [0, 10, 20, 30, 40, 50],          # No gap
        'Signal': [1, 2, 3, 4, 5, 6]
    })

@pytest.fixture
def mock_pass_report():
    """Simulates a diagnostic report where everything passed."""
    return pd.DataFrame({
        'Test': ['Input Coverage', 'Model Fit (CV)'],
        'Variable': ['Length', 'Signal'],
        'Metric': ['Max Gap', 'R2'],
        'Value': [0.1, 0.9],
        'Pass': [True, True]
    })

@pytest.fixture
def mock_gap_fail_report():
    """Simulates a report where 'Length' has a coverage gap."""
    return pd.DataFrame({
        'Test': ['Input Coverage', 'Model Fit (CV)'],
        'Variable': ['Length', 'Signal'],
        'Metric': ['Max Gap', 'R2'],
        'Value': [0.5, 0.9],
        'Pass': [False, True] # Length Fails
    })

@pytest.fixture
def mock_stability_fail_report():
    """Simulates a report where the model fit is unstable."""
    return pd.DataFrame({
        'Test': ['Input Coverage', 'Model Fit (CV)', 'Bootstrap Convergence'],
        'Variable': ['Length', 'Signal', 'Signal'],
        'Metric': ['Max Gap', 'R2', 'CI'],
        'Value': [0.1, 0.2, 0.5],
        'Pass': [True, False, False] # Both stability metrics fail
    })


# --- PART 1: Helper Function Tests ---

def test_fill_gaps_logic(basic_df):
    """Test that _fill_gaps correctly identifies and samples inside a hole."""
    input_cols = ['Length', 'Angle']
    n = 10

    # The gap in 'Length' is between 2.0 and 8.0
    # The gap in 'Angle' logic shouldn't matter here, we only constrain Length

    new_samples = _fill_gaps(basic_df, target_col='Length', all_inputs=input_cols, n=n)

    assert len(new_samples) == n
    assert list(new_samples.columns) == input_cols

    # CRITICAL: Verify samples fall INSIDE the gap
    assert new_samples['Length'].min() >= 2.0
    assert new_samples['Length'].max() <= 8.0

    # Verify other columns span the full original range
    assert new_samples['Angle'].min() >= 0
    assert new_samples['Angle'].max() <= 50

def test_fill_gaps_boundary_ordering(basic_df):
    """Test that it handles unsorted data correctly."""
    # Shuffle the dataframe
    shuffled_df = basic_df.sample(frac=1).copy()

    new_samples = _fill_gaps(shuffled_df, target_col='Length', all_inputs=['Length'], n=5)

    # Should still find the gap between 2 and 8
    assert new_samples['Length'].min() >= 2.0
    assert new_samples['Length'].max() <= 8.0

def test_sample_uncertainty_bounds(basic_df):
    """Test that uncertainty sampling produces valid candidates within bounds."""
    input_cols = ['Length', 'Angle']
    n = 10

    new_samples = _sample_uncertainty(basic_df, input_cols, 'Signal', n)

    assert len(new_samples) == n
    assert list(new_samples.columns) == input_cols

    # Check bounds (Must respect min/max of input data)
    assert new_samples['Length'].min() >= 0.0
    assert new_samples['Length'].max() <= 10.0
    assert new_samples['Angle'].min() >= 0
    assert new_samples['Angle'].max() <= 50


# --- PART 2: Main Function Logic (Mocked) ---

@patch('digiqual.adaptive.sample_sufficiency')
def test_targeted_samples_all_pass(mock_diag, basic_df, mock_pass_report):
    """If diagnostics pass, return empty DataFrame."""
    mock_diag.return_value = mock_pass_report

    res = generate_targeted_samples(basic_df, ['Length'], 'Signal')

    assert res.empty
    mock_diag.assert_called_once()

@patch('digiqual.adaptive.sample_sufficiency')
def test_targeted_samples_gap_strategy(mock_diag, basic_df, mock_gap_fail_report):
    """If coverage fails, trigger _fill_gaps strategy."""
    mock_diag.return_value = mock_gap_fail_report

    res = generate_targeted_samples(basic_df, ['Length', 'Angle'], 'Signal', n_new_per_fix=5)

    assert len(res) == 5
    # Since 'Length' failed, the samples should be constrained to the 2-8 gap
    # (This confirms _fill_gaps was actually used)
    assert res['Length'].min() >= 2.0
    assert res['Length'].max() <= 8.0

@patch('digiqual.adaptive.sample_sufficiency')
def test_targeted_samples_stability_strategy(mock_diag, basic_df, mock_stability_fail_report):
    """
    If model/bootstrap fails, trigger _sample_uncertainty strategy.
    CRITICAL: Should only run ONCE even if multiple stability tests fail.
    """
    mock_diag.return_value = mock_stability_fail_report

    # We requested 5 samples per fix.
    # The report fails "Model Fit" AND "Bootstrap Convergence".
    # Logic should deduplicate these into ONE "Global_Model" fix.
    res = generate_targeted_samples(basic_df, ['Length', 'Angle'], 'Signal', n_new_per_fix=5)

    assert len(res) == 5 # Should be 5, NOT 10

    expected_cols = ['Length', 'Angle', 'Refinement_Reason']
    assert list(res.columns) == expected_cols

@patch('digiqual.adaptive.sample_sufficiency')
def test_targeted_samples_mixed_strategy(mock_diag, basic_df):
    """Test that multiple DIFFERENT types of failures generate combined samples."""
    # Create a report where Length fails Coverage AND Model fails Fit
    mixed_report = pd.DataFrame({
        'Test': ['Input Coverage', 'Model Fit (CV)'],
        'Variable': ['Length', 'Signal'],
        'Metric': ['Gap', 'R2'],
        'Value': [0.5, 0.2],
        'Pass': [False, False]
    })
    mock_diag.return_value = mixed_report

    n = 5
    res = generate_targeted_samples(basic_df, ['Length', 'Angle'], 'Signal', n_new_per_fix=n)

    # Should contain 5 (for gap) + 5 (for model) = 10 total
    assert len(res) == 10

def test_targeted_samples_empty_report(basic_df):
    """Edge Case: If sample_sufficiency returns empty DF (e.g. error handled), return empty."""
    with patch('digiqual.adaptive.sample_sufficiency') as mock_diag:
        mock_diag.return_value = pd.DataFrame() # Empty

        res = generate_targeted_samples(basic_df, ['Length'], 'Signal')
        assert res.empty

def test_fill_gaps_zero_samples(basic_df):
    """Edge Case: Requesting 0 samples should return an empty DataFrame with correct columns."""
    input_cols = ['Length', 'Angle']
    # This ensures qmc.LatinHypercube doesn't crash on n=0
    res = _fill_gaps(basic_df, 'Length', input_cols, n=0)

    assert res.empty
    # CRITICAL: Even if empty, it must have the correct columns to prevent
    # pd.concat failures later in the pipeline.
    assert list(res.columns) == input_cols

@patch('digiqual.adaptive.sample_sufficiency')
def test_targeted_samples_duplicate_coverage_failure(mock_diag, basic_df):
    """
    Future-Proofing: If a variable fails multiple coverage metrics,
    we should only trigger gap-filling ONCE per variable.
    """
    # Report has TWO failures for 'Length'
    duplicate_report = pd.DataFrame({
        'Test': ['Input Coverage', 'Input Coverage'],
        'Variable': ['Length', 'Length'],
        'Pass': [False, False]
    })
    mock_diag.return_value = duplicate_report

    res = generate_targeted_samples(basic_df, ['Length'], 'Signal', n_new_per_fix=5)

    # Current behavior check:
    # If your code DOES NOT de-duplicate input coverage, this will be 10.
    # If your code IS robust, this should be 5.

    # Recommendation: It is safer if this returns 5.
    # If it returns 10, consider adding `if var_name in handled_vars: continue`
    # to your coverage block in adaptive.py.
    assert len(res) == 5

def test_sample_uncertainty_zero_samples(basic_df):
    res = _sample_uncertainty(basic_df, ['Length'], 'Signal', n=0)
    assert res.empty
