import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from digiqual.core import SimulationStudy
import sys

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
    study_instance = SimulationStudy()
    study_instance.inputs = ['Length', 'Angle']
    study_instance.outcome = 'Signal'
    return study_instance

# --- Existing Initialization & Data Tests ---

def test_initialization(study):
    assert study.inputs == ['Length', 'Angle']
    assert study.outcome == 'Signal'
    assert study.data.empty

def test_add_data(study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED
    assert len(study.data) == 20
    study.clean_data = clean_df.copy()
    study.add_data(clean_df) # Appending
    assert len(study.data) == 40
    assert study.clean_data.empty

def test_validate_explicit(study, clean_df):
    dirty_row = pd.DataFrame({'Length': [1], 'Angle': [0], 'Signal': [-5]})
    mixed_df = pd.concat([clean_df, dirty_row], ignore_index=True)
    study.add_data(mixed_df, outcome_col="Signal") # <--- FIXED
    study._validate()

    assert len(study.clean_data) == 21
    assert len(study.removed_data) == 0

def test_validate_with_removed_data(study, clean_df):
    dirty_row = pd.DataFrame({'Length': ['bad'], 'Angle': ['bad'], 'Signal': ['bad']})
    mixed_df = pd.concat([clean_df, dirty_row], ignore_index=True)
    study.add_data(mixed_df, outcome_col="Signal") # <--- FIXED
    study._validate()
    assert len(study.removed_data) == 1

def test_diagnose_implicit_validation(study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED
    results = study.diagnose()
    assert not study.clean_data.empty
    assert not results.empty

def test_add_data_filtering(study, clean_df):
    df_with_junk = clean_df.copy()
    df_with_junk["Irrelevant"] = "Delete me"
    # FIXED: Explicitly tell it what the inputs are, so it knows to drop "Irrelevant"
    study.add_data(df_with_junk, outcome_col="Signal", input_cols=["Length", "Angle"])
    assert "Irrelevant" not in study.data.columns
    assert len(study.data.columns) == 3

def test_add_data_missing_column(study):
    incomplete_df = pd.DataFrame({'Length': [1], 'Angle': [10]})
    with pytest.raises(ValueError, match="missing required columns"):
        study.add_data(incomplete_df, outcome_col="Signal") # <--- FIXED

# --- NEW: Cache Management Tests ---

def test_add_data_clears_caches(study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED

    study.models_cache = {'dummy': 'model'}
    study.variance_cache = {'dummy': 'var'}
    study.pod_curves_cache = {'dummy': 'curve'}
    study.threshold_spectrum_cache = {'dummy': 'spec'}

    study.add_data(clean_df) # Appending

    assert not study.models_cache
    assert not study.variance_cache
    assert not study.pod_curves_cache
    assert not study.threshold_spectrum_cache

# --- Optimise Tests ---

@patch("digiqual.core.run_adaptive_search")
def test_optimise_delegation(mock_run_adaptive, study, clean_df):
    study.add_data(clean_df, outcome_col="Signal")

    mock_result_df = clean_df.copy()
    mock_run_adaptive.return_value = mock_result_df
    cmd = "echo dummy"
    ranges = {"Length": (0, 10), "Angle": (-90, 90)}

    study.optimise(executor=cmd, ranges=ranges, n_start=10, max_iter=2)

    mock_run_adaptive.assert_called_once()
    call_kwargs = mock_run_adaptive.call_args.kwargs
    assert call_kwargs['executor'] == cmd
    assert call_kwargs['ranges'] == ranges
    pd.testing.assert_frame_equal(study.data, mock_result_df)

def test_optimise_range_mismatch(study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED
    bad_ranges = {"Length": (0, 10)}

    with pytest.raises(ValueError, match="Variable Mismatch!"):
        study.optimise(executor="dummy", ranges=bad_ranges)

# --- Core Refinement & Diagnostics Tests ---

def test_validate_error(study):
    study._validate()
    assert study.clean_data.empty

def test_diagnose_empty_data(study):
    res = study.diagnose()
    assert res.empty

def test_diagnose_failed_validation(study):
    df = pd.DataFrame({'Length': [1,2], 'Angle': [0,0], 'Signal': [10,10]})
    study.add_data(df, outcome_col="Signal") # <--- FIXED
    res = study.diagnose()
    assert res.empty

def test_refine_empty_clean_data(study):
    res = study.refine()
    assert res.empty

@patch("digiqual.core.generate_targeted_samples")
def test_refine_success(mock_gen, study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED
    study._validate()
    mock_gen.return_value = pd.DataFrame({'Length': [5.0]})
    res = study.refine()
    assert not res.empty
    assert len(res) == 1

# --- PoD & Caching Integration Tests ---

def test_pod_invalid_data(study, clean_df):
    with pytest.raises(ValueError, match="Cannot run PoD analysis"):
         study.pod(poi_col="Length", threshold=0.5)
    study.add_data(clean_df, outcome_col="Signal")
    with pytest.raises(ValueError, match="is not in the initialized input_cols"):
         study.pod(poi_col="BadCol", threshold=0.5)

@patch("digiqual.core.pod.fit_all_robust_mean_models")
@patch("digiqual.core.pod.fit_variance_model")
@patch("digiqual.core.pod.infer_best_distribution")
@patch("digiqual.integration.compute_multi_dim_pod")
@patch("digiqual.core.pod.bootstrap_pod_ci")
def test_pod_kriging_selection(mock_boot, mock_comp, mock_dist, mock_var, mock_fit, study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED
    from sklearn.gaussian_process import GaussianProcessRegressor

    mock_model = GaussianProcessRegressor()
    mock_model.model_type_ = 'Kriging'
    mock_model.model_params_ = None
    mock_model.cv_scores_ = {}

    mock_fit.return_value = (
        {('Kriging', None): mock_model},
        {('Kriging', None): 0.1},
        ('Kriging', None)
    )

    mock_var.return_value = (np.ones(len(clean_df)), 0.1)
    mock_dist.return_value = ("norm", (0,1))
    mock_comp.return_value = (np.ones(100), np.ones(100))
    mock_boot.return_value = (np.ones(100), np.ones(100))

    study.pod(poi_col="Length", threshold=0.5)
    assert study.pod_results["mean_model"].model_type_ == 'Kriging'

def test_pod_fast_caching_and_slicing(study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED

    _ = study.pod(poi_col="Length", threshold=10.5, n_boot=0)

    assert study.models_cache
    assert study.variance_cache
    assert study.pod_curves_cache

    res2 = study.update_slice(slice_values={"Angle": 45.0})
    assert res2["slice_values"]["Angle"] == 45.0

# --- NEW: Time Heuristic & Linear PoD Tests ---

def test_estimate_compute_time(study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED
    time_est = study.estimate_compute_time(model_type="polynomial", n_boot=100, n_nuisances=1, n_jobs=1)

    assert isinstance(time_est, float)
    assert time_est > 0

@patch("digiqual.core.bootstrap_linear_pod_ci")
@patch("digiqual.core.compute_linear_pod_curve")
@patch("digiqual.core.fit_linear_a_hat_model")
def test_linear_pod_execution(mock_fit, mock_comp, mock_boot, study, clean_df):
    study.add_data(clean_df, outcome_col="Signal") # <--- FIXED

    mock_fit.return_value = ("dummy_model", 0.1)
    mock_comp.return_value = (np.ones(100), np.ones(100))
    mock_boot.return_value = (np.ones(100), np.ones(100))

    res = study.linear_pod(poi_col="Length", threshold=10.5, n_boot=10)

    assert res["model"] == "dummy_model"
    assert res["tau"] == 0.1
    assert "pod" in res["curves"]
    assert study.linear_pod_results is not None

# --- Visualisation Tests ---

@patch("matplotlib.pyplot.show")
@patch("matplotlib.figure.Figure.savefig")
def test_visualise(mock_savefig, mock_show, study, clean_df):
    study.visualise()

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    mean_model = make_pipeline(PolynomialFeatures(1), LinearRegression())
    mean_model.model_type_ = 'Polynomial'
    mean_model.model_params_ = 1
    mean_model.cv_scores_ = {('Polynomial', 1): 0.1}
    mean_model.cv_winner_ = ('Polynomial', 1)

    study.inputs = ["Length"]
    study.pod_results = {
        "poi_cols": ["Length"],
        "nuisance_cols": [],
        "slice_values": {},
        "threshold": 0.5,
        "X": np.array([[1], [2], [3]]),
        "y": np.array([1,2,3]),
        "X_eval": np.array([[1], [2], [3]]),
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
    study.pod_results = {"poi_cols": ["Length"], "mean_model": MagicMock(), "X": np.empty((0, 1)), "residuals": np.array([]), "X_eval": np.empty((0, 1)), "bandwidth": 1, "y": np.array([]), "threshold": 0.5, "curves": {"mean_response": np.array([]), "pod": np.array([]), "ci_lower": np.array([]), "ci_upper": np.array([])}}
    with patch("digiqual.core.pod.plot_model_selection"), patch("digiqual.core.plot_signal_model", return_value=MagicMock()), patch("digiqual.core.plot_pod_curve", return_value=MagicMock()):
        # <-- FIXED: We now explicitly expect the ImportError to be thrown
        with pytest.raises(ImportError):
            study.visualise(show=True)
