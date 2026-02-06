import pytest
import pandas as pd
import numpy as np
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

# --- Tests ---

def test_initialization(study):
    """Test that the object initializes with empty state."""
    assert study.inputs == ['Length', 'Angle']
    assert study.outcome == 'Signal'
    assert study.data.empty
    assert study.clean_data.empty
    assert study.sufficiency_results.empty

def test_add_data(study, clean_df):
    """Test adding data and state resets."""
    # 1. Add first batch
    study.add_data(clean_df)
    assert len(study.data) == 20

    # Simulate existing results to test reset logic
    study.clean_data = clean_df.copy()

    # 2. Add second batch
    study.add_data(clean_df)
    assert len(study.data) == 40
    # Check that downstream results were wiped (Safety Check)
    assert study.clean_data.empty
    assert study.sufficiency_results.empty

def test_validate_explicit(study, clean_df):
    """Test explicitly calling .validate()."""
    # Create dirty data (negative signal)
    dirty_row = pd.DataFrame({'Length': [1], 'Angle': [0], 'Signal': [-5]})
    mixed_df = pd.concat([clean_df, dirty_row], ignore_index=True)

    study.add_data(mixed_df)
    study.validate()

    # Should have removed the 1 dirty row
    assert len(study.clean_data) == 20
    assert len(study.removed_data) == 1
    assert not study.clean_data.empty

def test_diagnose_implicit_validation(study, clean_df):
    """Test that .diagnose() runs validation automatically (Lazy Loading)."""
    study.add_data(clean_df)

    # We skip study.validate() intentionally
    results = study.diagnose()

    # Check that validation happened "under the hood"
    assert not study.clean_data.empty
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "Pass" in results.columns

def test_diagnose_no_data(study):
    """Test graceful failure if user forgets to add data."""
    results = study.diagnose()
    assert results.empty

def test_diagnose_insufficient_data(study):
    """Test graceful failure if validation removes everything."""
    # Add only bad data
    bad_df = pd.DataFrame({'Length': [1]*10, 'Angle': [0]*10, 'Signal': [-1]*10})
    study.add_data(bad_df)

    # Diagnose should try to validate, fail to find enough rows, and return empty
    results = study.diagnose()
    assert results.empty
    assert study.clean_data.empty


def test_add_data_filtering(study, clean_df):
    """
    Test that add_data automatically strips away columns
    that were not defined in __init__.
    """
    # 1. Create data with an extra "Junk" column
    df_with_junk = clean_df.copy()
    df_with_junk["Irrelevant_Meta_Data"] = "Should be deleted"

    # 2. Add it to the study
    study.add_data(df_with_junk)

    # 3. Assertions
    assert "Irrelevant_Meta_Data" not in study.data.columns
    assert "Length" in study.data.columns
    # Ensure we only have the 3 expected columns (Length, Angle, Signal)
    assert len(study.data.columns) == 3

def test_add_data_missing_column(study):
    """
    Test that add_data raises a ValueError immediately
    if the new data lacks a required input or outcome.
    """
    # Create a dataframe that is missing the 'Signal' column
    incomplete_df = pd.DataFrame({
        'Length': [1, 2, 3],
        'Angle': [10, 20, 30]
        # 'Signal' is missing!
    })

    # Assert that this raises a ValueError
    with pytest.raises(ValueError, match="missing required columns"):
        study.add_data(incomplete_df)

def test_add_data_schema_consistency(study, clean_df):
    """
    Test adding two batches: one pure, one with junk.
    They should merge cleanly without creating NaN columns.
    """
    # Batch 1: Pure
    study.add_data(clean_df)

    # Batch 2: With Junk
    junk_df = clean_df.copy()
    junk_df["Junk"] = 123
    study.add_data(junk_df)

    # Should have 40 rows total, and still only 3 columns
    assert len(study.data) == 40
    assert len(study.data.columns) == 3
    assert not study.data.isnull().values.any() # Ensure no NaNs were created
