import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from digiqual.core import SimulationStudy
from digiqual.integration import compute_multi_dim_pod

@pytest.fixture
def clean_df():
    """Creates a basic clean dataframe for end-to-end tests."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        'Length': np.linspace(0.1, 10.0, n),
        'Angle': np.linspace(-90.0, 90.0, n),
        'Signal': 5.0 * np.linspace(0.1, 10.0, n) + np.random.normal(0, 0.5, n) + 10.0
    })

@pytest.fixture
def dummy_data():
    """Provides consistent base data for direct integration tests."""
    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    residuals = np.array([0.1, -0.1, 0.0])
    dist_info = ('norm', (0.0, 1.0))
    bandwidth = 1.0
    poi_grid = np.array([[1.0], [2.0], [3.0]])
    feature_names = ["Length", "Angle"]
    poi_names = ["Length"]
    return X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.side_effect = lambda X: np.sum(X, axis=1)
    return model

@patch("digiqual.pod.predict_local_std")
def test_compute_multi_dim_pod_with_norm_dist(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names = dummy_data
    n_mc_samples = 100
    mock_std.return_value = np.ones(n_mc_samples)

    # Nuisance ranges specify min/max from data, but we pass custom Normal distribution
    nuisance_ranges = {"Angle": (-90.0, 90.0)}
    nuisance_dists = {"Angle": ("norm", (45.0, 5.0))} # loc=45, scale=5
    threshold = 50.0

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, threshold, n_mc_samples=n_mc_samples,
        feature_names=feature_names, poi_names=poi_names,
        nuisance_dists=nuisance_dists
    )

    assert len(pod) == len(poi_grid)
    assert len(mean) == len(poi_grid)
    assert np.all((pod >= 0.0) & (pod <= 1.0))

@patch("digiqual.pod.predict_local_std")
def test_compute_multi_dim_pod_with_uniform_dist(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names = dummy_data
    n_mc_samples = 100
    mock_std.return_value = np.ones(n_mc_samples)

    nuisance_ranges = {"Angle": (-90.0, 90.0)}
    nuisance_dists = {"Angle": ("uniform", (20.0, 40.0))} # loc=20, scale=40 (uniform on [20, 60])
    threshold = 50.0

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, threshold, n_mc_samples=n_mc_samples,
        feature_names=feature_names, poi_names=poi_names,
        nuisance_dists=nuisance_dists
    )

    assert len(pod) == len(poi_grid)
    assert len(mean) == len(poi_grid)
    assert np.all((pod >= 0.0) & (pod <= 1.0))

def test_simulation_study_pod_caching_differentiation(clean_df):
    study = SimulationStudy()
    study.add_data(clean_df, outcome_col="Signal", input_cols=["Length", "Angle"])

    # Run PoD with default distribution (Uniform)
    res_default = study.pod(
        poi_col="Length",
        threshold=35.0,
        nuisance_col="Angle",
        n_boot=0
    )

    # Run PoD with Normal distribution
    res_norm = study.pod(
        poi_col="Length",
        threshold=35.0,
        nuisance_col="Angle",
        n_boot=0,
        nuisance_dists={"Angle": ("norm", (45.0, 5.0))}
    )

    # Assert that they are distinct (different cache hits / calculations)
    # The default uniform is [-90, 90], norm is centered at 45 with std=5, so results should differ
    assert not np.array_equal(res_default["curves"]["pod"], res_norm["curves"]["pod"])

    # Run PoD with a DIFFERENT Normal distribution to verify caching invalidates correctly
    res_norm_diff = study.pod(
        poi_col="Length",
        threshold=35.0,
        nuisance_col="Angle",
        n_boot=0,
        nuisance_dists={"Angle": ("norm", (10.0, 2.0))}
    )

    assert not np.array_equal(res_norm["curves"]["pod"], res_norm_diff["curves"]["pod"])
