import pytest
import pandas as pd
from unittest.mock import patch
from digiqual.adaptive import run_adaptive_search

# --- Fixtures ---

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Length': [1.0, 2.0],
        'Angle': [10.0, 20.0],
        'Signal': [0.5, 0.8]
    })

@pytest.fixture
def ranges():
    return {'Length': (0, 10), 'Angle': (-45, 45)}

# --- Tests for run_adaptive_search (The Manager) ---

@patch("digiqual.executors.CLIExecutor.run")
@patch("digiqual.adaptive.generate_lhs")
@patch("digiqual.adaptive.sample_sufficiency")
@patch("digiqual.adaptive.validate_simulation")
def test_adaptive_cold_start(mock_validate, mock_sufficiency, mock_lhs, mock_exec, ranges):
    """Test iteration 0: Generating LHS when no data exists."""
    # Setup Mocks
    mock_lhs.return_value = pd.DataFrame({'Length': [1], 'Angle': [1]}) # 1 LHS point
    mock_exec.return_value = pd.DataFrame({'Length': [1], 'Angle': [1], 'Signal': [0.5]})

    # Mock Validation/Sufficiency to pass immediately so loop stops after init
    mock_validate.return_value = (mock_exec.return_value, pd.DataFrame())
    mock_sufficiency.return_value = pd.DataFrame({'Pass': [True, True]}) # All pass

    result = run_adaptive_search(
        executor="cmd", # Testing backwards compatibility wrapping
        input_cols=["Length", "Angle"],
        outcome_col="Signal",
        ranges=ranges,
        existing_data=pd.DataFrame(), # Empty!
        n_start=5,
        max_iter=1
    )

    # Assert LHS was called (Cold Start)
    mock_lhs.assert_called_once_with(5, ranges)
    # Assert Execution happened
    mock_exec.assert_called()
    # Assert result contains the data
    assert len(result) == 1

@patch("digiqual.executors.CLIExecutor.run")
@patch("digiqual.adaptive.generate_lhs")
@patch("digiqual.adaptive.sample_sufficiency")
@patch("digiqual.adaptive.validate_simulation")
def test_adaptive_resume_existing(mock_validate, mock_sufficiency, mock_lhs, mock_exec, ranges, sample_df):
    """Test resuming: Should SKIP generate_lhs if data exists."""
    # Setup Mocks to pass immediately
    mock_validate.return_value = (sample_df, pd.DataFrame())
    mock_sufficiency.return_value = pd.DataFrame({'Pass': [True]})

    run_adaptive_search(
        executor="cmd",
        input_cols=["Length", "Angle"],
        outcome_col="Signal",
        ranges=ranges,
        existing_data=sample_df, # Data exists!
        n_start=5
    )

    # Assert LHS was NOT called
    mock_lhs.assert_not_called()

@patch("digiqual.executors.CLIExecutor.run")
@patch("digiqual.adaptive.generate_targeted_samples")
@patch("digiqual.adaptive.sample_sufficiency")
@patch("digiqual.adaptive.validate_simulation")
def test_adaptive_refinement_loop(mock_validate, mock_sufficiency, mock_target, mock_exec, ranges, sample_df):
    """Test the refinement loop: Fail diagnostics -> Generate new points -> Run -> Repeat."""

    # --- Iteration 1 Setup: FAIL ---
    # Validation passes through
    mock_validate.side_effect = lambda df, *args: (df, pd.DataFrame())
    # Diagnostics fail
    fail_report = pd.DataFrame({'Test': ['Coverage'], 'Pass': [False]})
    pass_report = pd.DataFrame({'Test': ['Coverage'], 'Pass': [True]})

    # Side Effects for the loop:
    # 1. Fail first check
    # 2. Pass second check (Convergence)
    mock_sufficiency.side_effect = [fail_report, pass_report]

    # Refinement generates 1 new point
    new_points = pd.DataFrame({'Length': [5], 'Angle': [5]})
    mock_target.return_value = new_points

    # Execution returns the new point with a result
    new_result = new_points.copy()
    new_result['Signal'] = 0.99
    mock_exec.return_value = new_result

    # --- Run ---
    final_df = run_adaptive_search(
        executor="cmd",
        input_cols=["Length", "Angle"],
        outcome_col="Signal",
        ranges=ranges,
        existing_data=sample_df,
        max_iter=5
    )

    # --- Assertions ---
    # Should have called refine once
    mock_target.assert_called_once()
    # Should have called execute once (for the refinement batch)
    mock_exec.assert_called()
    # Final data should be Initial (2) + Refined (1) = 3 rows
    assert len(final_df) == 3
