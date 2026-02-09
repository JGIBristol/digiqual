import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from digiqual.core import SimulationStudy

# --- Fixtures ---

@pytest.fixture
def clean_df():
    """Creates a basic clean dataframe."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        'Length': np.linspace(0, 10, n),
        'Angle': np.linspace(-90, 90, n),
        'Signal': np.linspace(0, 1, n) + 10 # +10 ensures positive signal
    })

@pytest.fixture
def study():
    """Returns an initialized SimulationStudy object."""
    return SimulationStudy(input_cols=['Length', 'Angle'], outcome_col='Signal')

# --- Existing Tests (Keep these) ---

def test_initialization(study):
    assert study.inputs == ['Length', 'Angle']
    assert study.outcome == 'Signal'
    assert study.data.empty

def test_add_data(study, clean_df):
    study.add_data(clean_df)
    assert len(study.data) == 20
    study.clean_data = clean_df.copy()
    study.add_data(clean_df)
    assert len(study.data) == 40
    assert study.clean_data.empty

def test_validate_explicit(study, clean_df):
    dirty_row = pd.DataFrame({'Length': [1], 'Angle': [0], 'Signal': [-5]})
    mixed_df = pd.concat([clean_df, dirty_row], ignore_index=True)
    study.add_data(mixed_df)
    study.validate()
    assert len(study.clean_data) == 20
    assert len(study.removed_data) == 1

def test_diagnose_implicit_validation(study, clean_df):
    study.add_data(clean_df)
    results = study.diagnose()
    assert not study.clean_data.empty
    assert not results.empty

def test_add_data_filtering(study, clean_df):
    df_with_junk = clean_df.copy()
    df_with_junk["Irrelevant"] = "Delete me"
    study.add_data(df_with_junk)
    assert "Irrelevant" not in study.data.columns
    assert len(study.data.columns) == 3

def test_add_data_missing_column(study):
    incomplete_df = pd.DataFrame({'Length': [1], 'Angle': [10]})
    with pytest.raises(ValueError, match="missing required columns"):
        study.add_data(incomplete_df)

# --- NEW TEST FOR OPTIMIZE ---

@patch("digiqual.core.run_adaptive_search")
def test_optimize_delegation(mock_run_adaptive, study, clean_df):
    """
    Test that study.optimize correctly delegates to run_adaptive_search
    and updates the study's data with the result.
    """
    # 1. Setup the mock to return a dummy dataframe
    # This simulates the adaptive loop finishing and returning a combined dataset
    mock_result_df = clean_df.copy()
    mock_run_adaptive.return_value = mock_result_df

    # 2. Define inputs
    cmd = "echo dummy"
    ranges = {"Length": (0, 10), "Angle": (-90, 90)}

    # 3. Call the method
    study.optimize(
        command=cmd,
        ranges=ranges,
        n_start=10,
        max_iter=2
    )

    # 4. Assertions
    # A) Check that run_adaptive_search was called with the correct arguments
    mock_run_adaptive.assert_called_once()

    # Check specific args passed to the function
    call_kwargs = mock_run_adaptive.call_args.kwargs
    assert call_kwargs['command'] == cmd
    assert call_kwargs['ranges'] == ranges
    assert call_kwargs['input_cols'] == ['Length', 'Angle']
    assert call_kwargs['outcome_col'] == 'Signal'

    # B) Check that the result was loaded into the study
    pd.testing.assert_frame_equal(study.data, mock_result_df)
