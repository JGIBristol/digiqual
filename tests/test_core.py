import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from digiqual.core import SimulationStudy
import sys
from unittest.mock import MagicMock

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
    """
    Verifies that numeric negative signals are now treated as valid.
    """
    # This row has a negative signal, which is now allowed
    dirty_row = pd.DataFrame({'Length': [1], 'Angle': [0], 'Signal': [-5]})
    mixed_df = pd.concat([clean_df, dirty_row], ignore_index=True)
    study.add_data(mixed_df)
    study._validate()

    # 20 rows from clean_df + 1 from dirty_row = 21 valid rows
    assert len(study.clean_data) == 21
    # No rows should be removed because -5 is a valid number
    assert len(study.removed_data) == 0

def test_validate_with_removed_data(study, clean_df):
    dirty_row = pd.DataFrame({'Length': ['bad'], 'Angle': ['bad'], 'Signal': ['bad']})
    mixed_df = pd.concat([clean_df, dirty_row], ignore_index=True)
    study.add_data(mixed_df)
    study._validate()
    # It will drop the bad row and print the warning!
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

# --- NEW TEST FOR optimise ---

@patch("digiqual.core.run_adaptive_search")
def test_optimise_delegation(mock_run_adaptive, study, clean_df):
    """
    Test that study.optimise correctly delegates to run_adaptive_search
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
    study.optimise(
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


# --- NEW TESTS FOR CORE ---

def test_validate_error(study):
    # Triggers ValidationError because self.data is empty
    study._validate()
    assert study.clean_data.empty

def test_diagnose_empty_data(study):
    res = study.diagnose()
    assert res.empty

def test_diagnose_failed_validation(study):
    # Provide < 10 rows to fail validation
    df = pd.DataFrame({'Length': [1,2], 'Angle': [0,0], 'Signal': [10,10]})
    study.add_data(df)
    res = study.diagnose()
    assert res.empty

def test_refine_empty_clean_data(study):
    res = study.refine()
    assert res.empty

@patch("digiqual.core.generate_targeted_samples")
def test_refine_success(mock_gen, study, clean_df):
    study.add_data(clean_df)
    study._validate()
    mock_gen.return_value = pd.DataFrame({'Length': [5.0]})
    res = study.refine()
    assert not res.empty
    assert len(res) == 1

def test_pod_invalid_data(study, clean_df):
    # Empty data
    with pytest.raises(ValueError, match="Cannot run PoD analysis"):
         study.pod(poi_col="Length", threshold=0.5)

    # Missing poi_col
    study.add_data(clean_df)
    with pytest.raises(ValueError, match="not found in data columns"):
         study.pod(poi_col="BadCol", threshold=0.5)

@patch("digiqual.core.pod.fit_robust_mean_model")
@patch("digiqual.core.pod.fit_variance_model")
@patch("digiqual.core.pod.infer_best_distribution")
@patch("digiqual.core.pod.compute_pod_curve")
@patch("digiqual.core.pod.bootstrap_pod_ci")
def test_pod_kriging_selection(mock_boot, mock_comp, mock_dist, mock_var, mock_fit, study, clean_df):
    study.add_data(clean_df)
    from sklearn.gaussian_process import GaussianProcessRegressor
    mock_model = GaussianProcessRegressor()
    mock_model.model_type_ = 'Kriging'
    mock_model.model_params_ = None
    mock_model.cv_scores_ = {}
    mock_fit.return_value = mock_model

    mock_var.return_value = (np.array([1]), 0.1, np.array([1]))
    mock_dist.return_value = ("norm", (0,1))
    mock_comp.return_value = (np.array([1]), np.array([1]))
    mock_boot.return_value = (np.array([1]), np.array([1]))

    study.pod(poi_col="Length", threshold=0.5)
    assert study.pod_results["mean_model"].model_type_ == 'Kriging'

@patch("matplotlib.pyplot.show")
@patch("matplotlib.figure.Figure.savefig")
def test_visualise(mock_savefig, mock_show, study, clean_df):
    # Without pod_results (prints a message)
    study.visualise()

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    mean_model = make_pipeline(PolynomialFeatures(1), LinearRegression())
    mean_model.cv_scores_ = {('Polynomial', 1): 0.1}

    study.pod_results = {
        "poi_col": "Length",
        "threshold": 0.5,
        "X": np.array([1,2,3]),
        "y": np.array([1,2,3]),
        "X_eval": np.array([1,2,3]),
        "residuals": np.array([0,0,0]),
        "bandwidth": 0.1,
        "mean_model": mean_model,
        "curves": {
             "mean_response": np.array([1,2,3]),
             "pod": np.array([1,2,3]),
             "ci_lower": np.array([0,0,0]),
             "ci_upper": np.array([1,1,1])
        }
    }
    study.visualise(show=True, save_path="test_save")
    mock_show.assert_called_once()
    assert mock_savefig.call_count == 3


@patch.dict(sys.modules, {'matplotlib.pyplot': None})
def test_visualise_no_matplotlib(study):
    study.pod_results = {"poi_col": "Length", "mean_model": MagicMock(), "X": np.array([]), "residuals": np.array([]), "X_eval": np.array([]), "bandwidth": 1, "y": np.array([]), "threshold": 0.5, "curves": {"mean_response": np.array([]), "pod": np.array([]), "ci_lower": np.array([]), "ci_upper": np.array([])}}
    with patch("digiqual.core.pod.plot_model_selection"), patch("digiqual.core.plot_signal_model", return_value=MagicMock()), patch("digiqual.core.plot_pod_curve", return_value=MagicMock()):
        study.visualise(show=True)  # Should hit except ImportError and pass quietly
