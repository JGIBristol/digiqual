from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from digiqual.integration import compute_multi_dim_pod

# --- Fixtures ---

@pytest.fixture
def dummy_data():
    """Provides consistent base data for integration tests."""
    X_train = np.array([[1, 10], [2, 20], [3, 30]])
    residuals = np.array([0.1, -0.1, 0.0])
    dist_info = ('norm', (0, 1))
    bandwidth = 1.0
    poi_grid = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    feature_names = ["Length", "Angle"]
    poi_names = ["Length"]

    return X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.side_effect = lambda X: np.sum(X, axis=1)
    return model

# --- 1. Fast Path Tests (Vectorized, No Active Nuisances) ---

@patch("digiqual.cpp_fallback.predict_local_std_scaled")
def test_fast_path_single_threshold(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names = dummy_data
    mock_std.side_effect = lambda X, res, X_eval, bw, **kw: np.ones(len(X_eval))

    nuisance_ranges = {"Angle": (45.0, 45.0)}
    threshold = 50.0

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, threshold,
        feature_names=feature_names, poi_names=poi_names
    )

    assert len(pod) == len(poi_grid)
    assert len(mean) == len(poi_grid)
    assert pod.ndim == 1
    assert np.all((pod >= 0.0) & (pod <= 1.0))
    mock_model.predict.assert_called_once()

@patch("digiqual.cpp_fallback.predict_local_std_scaled")
def test_fast_path_vector_thresholds(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names = dummy_data
    mock_std.side_effect = lambda X, res, X_eval, bw, **kw: np.ones(len(X_eval))

    nuisance_ranges = {"Angle": (45.0, 45.0)}
    thresholds = np.array([40.0, 50.0, 60.0])

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, thresholds,
        feature_names=feature_names, poi_names=poi_names
    )

    assert pod.shape == (len(poi_grid), len(thresholds))
    assert mean.ndim == 1
    assert np.all((pod >= 0.0) & (pod <= 1.0))
    mock_model.predict.assert_called_once()

# --- 2. Slow Path Tests (Monte Carlo Integration) ---

@patch("digiqual.cpp_fallback.predict_local_std_scaled")
def test_slow_path_single_threshold(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names = dummy_data
    n_mc_samples = 100
    mock_std.side_effect = lambda X, res, X_eval, bw, **kw: np.ones(len(X_eval))

    nuisance_ranges = {"Angle": (30.0, 60.0)}
    threshold = 50.0

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, threshold, n_mc_samples=n_mc_samples,
        feature_names=feature_names, poi_names=poi_names
    )

    assert len(pod) == len(poi_grid)
    assert len(mean) == len(poi_grid)
    assert np.all((pod >= 0.0) & (pod <= 1.0))
    assert mock_model.predict.call_count >= 1

@patch("digiqual.cpp_fallback.predict_local_std_scaled")
def test_slow_path_vector_thresholds(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, feature_names, poi_names = dummy_data
    n_mc_samples = 50
    mock_std.side_effect = lambda X, res, X_eval, bw, **kw: np.ones(len(X_eval))

    nuisance_ranges = {"Angle": (30.0, 60.0)}
    thresholds = np.array([40.0, 50.0, 60.0])

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, thresholds, n_mc_samples=n_mc_samples,
        feature_names=feature_names, poi_names=poi_names
    )

    assert pod.shape == (len(poi_grid), len(thresholds))
    assert np.all((pod >= 0.0) & (pod <= 1.0))

# --- 3. Edge Cases ---

@patch("digiqual.cpp_fallback.predict_local_std_scaled")
def test_fallback_column_indexing(mock_std, mock_model, dummy_data):
    X_train, residuals, dist_info, bandwidth, poi_grid, _, _ = dummy_data
    mock_std.side_effect = lambda X, res, X_eval, bw, **kw: np.ones(len(X_eval))

    nuisance_ranges = {"Angle": (45.0, 45.0)}
    threshold = 50.0

    pod, mean = compute_multi_dim_pod(
        poi_grid, nuisance_ranges, mock_model, X_train, residuals,
        bandwidth, dist_info, threshold,
        feature_names=None, poi_names=None
    )

    assert len(pod) == len(poi_grid)
