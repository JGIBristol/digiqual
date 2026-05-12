import pytest
import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process.kernels import RBF
from digiqual.pod import (
    fit_all_robust_mean_models, # <-- Updated Import
    fit_variance_model,
    predict_local_std,
    infer_best_distribution,
    compute_pod_curve,
    bootstrap_pod_ci,
    plot_model_selection
)

from unittest.mock import patch
from digiqual.pod import calculate_reliability_point

# --- FIXTURES: Synthetic Physics Data ---

@pytest.fixture
def linear_data():
    """Generates simple linear data with constant Gaussian noise."""
    np.random.seed(42)
    X = np.linspace(0.1, 5.0, 50)
    y = 2.0 * X + 1.0 + np.random.normal(0, 0.5, size=len(X))
    return X, y

@pytest.fixture
def quadratic_data():
    """Generates quadratic data to test model selection."""
    np.random.seed(42)
    X = np.linspace(0.1, 5.0, 50)
    y = 0.5 * X**2 + np.random.normal(0, 0.2, size=len(X))
    return X, y

@pytest.fixture
def heteroscedastic_data():
    """Generates data where noise increases with X (to test variance model)."""
    np.random.seed(42)
    X = np.linspace(0.1, 5.0, 100)
    noise_level = 0.2 * X
    y = 3.0 * X + np.random.normal(0, 1, size=len(X)) * noise_level
    return X, y

@pytest.fixture
def gumbel_data():
    """Generates data with Gumbel-distributed residuals (non-Gaussian)."""
    np.random.seed(42)
    X = np.linspace(0.1, 3.0, 100)
    noise = stats.gumbel_r.rvs(loc=0, scale=0.5, size=len(X))
    y = 2.0 * X + noise
    return X, y

# --- 1. MEAN MODEL TESTS (Robust Regression) ---

def test_fit_all_robust_mean_models_selection(quadratic_data):
    """Test if Cross-Validation correctly identifies model type and params."""
    X, y = quadratic_data
    models, scores, best_key = fit_all_robust_mean_models(X, y, max_degree=5, n_folds=5)

    model = models[best_key] # Retrieve the winner

    # Check for the dynamic attributes
    assert hasattr(model, 'model_type_')
    assert hasattr(model, 'model_params_')
    assert model.model_type_ in ['Polynomial', 'Kriging']

    # If it chose polynomial for quadratic data, degree should be >= 2
    if model.model_type_ == 'Polynomial':
        assert model.model_params_ >= 2

def test_fit_all_robust_mean_models_shapes(linear_data):
    """Test it handles 1D array shapes correctly and predicts smoothly."""
    X, y = linear_data
    models, scores, best_key = fit_all_robust_mean_models(X, y, max_degree=3)

    model = models[best_key]

    X_eval = np.array([[1.0], [2.0], [3.0]])
    preds = model.predict(X_eval)

    assert len(preds) == 3
    assert isinstance(preds, np.ndarray)

def test_plot_model_selection(linear_data):
    """Test that the standalone model selection plot generates a figure."""
    X, y = linear_data
    models, scores, best_key = fit_all_robust_mean_models(X, y, max_degree=3)

    # Test that the plot generates successfully using the new scores dict
    fig = plot_model_selection(scores, cv_winner_key=best_key)
    assert fig is not None
    assert fig.axes # Checks that it actually contains plot axes

# --- 2. VARIANCE MODEL TESTS (Kernel Smoothing) ---

def test_fit_variance_model_outputs(heteroscedastic_data):
    X, y = heteroscedastic_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)
    mean_model = models[best_key]

    residuals, bw = fit_variance_model(X.reshape(-1, 1), y, mean_model)

    assert len(residuals) == len(y)
    assert isinstance(bw, float)
    assert bw > 0

def test_predict_local_std_heteroscedasticity(heteroscedastic_data):
    X, y = heteroscedastic_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)
    mean_model = models[best_key]

    residuals, bw = fit_variance_model(X.reshape(-1, 1), y, mean_model)

    eval_points = np.array([[0.5], [4.5]])
    predicted_std = predict_local_std(X.reshape(-1, 1), residuals, eval_points, bw)

    assert len(predicted_std) == 2
    assert np.all(predicted_std > 0)
    assert predicted_std[1] > predicted_std[0]

# --- 3. DISTRIBUTION INFERENCE TESTS (AIC Selection) ---

def test_infer_best_distribution_gaussian(linear_data):
    X, y = linear_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)
    mean_model = models[best_key]

    residuals, bw = fit_variance_model(X.reshape(-1, 1), y, mean_model)
    dist_name, params = infer_best_distribution(residuals, X.reshape(-1, 1), bw)

    assert dist_name in ['norm', 't', 'logistic']
    assert len(params) >= 2

def test_infer_best_distribution_gumbel(gumbel_data):
    X, y = gumbel_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)
    mean_model = models[best_key]

    residuals, bw = fit_variance_model(X.reshape(-1, 1), y, mean_model)
    dist_name, params = infer_best_distribution(residuals, X.reshape(-1, 1), bw)

    assert dist_name == 'gumbel_r'
    assert isinstance(params, tuple)

def test_infer_best_distribution_safety():
    np.random.seed(123)
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    residuals = np.random.normal(0, 1, 20)
    dist_name, params = infer_best_distribution(residuals, X, bandwidth=1.0)
    assert isinstance(dist_name, str)

# --- 4. POD COMPUTATION TESTS ---

def test_compute_pod_curve_bounds(linear_data):
    X, y = linear_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)
    mean_model = models[best_key]

    residuals, bw = fit_variance_model(X.reshape(-1, 1), y, mean_model)
    X_eval = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    dist_info = ('norm', (0, 1))

    pod_curve, mean_curve = compute_pod_curve(
        X_eval, mean_model, X.reshape(-1, 1), residuals, bw, dist_info, threshold=5.0
    )

    assert len(pod_curve) == len(X_eval)
    assert np.all(pod_curve >= 0.0)
    assert np.all(pod_curve <= 1.0)
    assert np.mean(mean_curve) > 0

# --- 5. BOOTSTRAP TESTS ---

def test_bootstrap_pod_ci_structure(linear_data):
    """Test that bootstrap returns valid upper/lower bounds using dynamic models."""
    X, y = linear_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)
    mean_model = models[best_key]

    residuals, bw = fit_variance_model(X.reshape(-1, 1), y, mean_model)
    X_eval = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    dist_info = ('norm', (0, 1))

    # Run small bootstrap (n_boot=10) for speed using new signature
    lower, upper = bootstrap_pod_ci(
        X.reshape(-1, 1), y, X_eval,
        threshold=5.0,
        model_type=mean_model.model_type_,
        model_params=mean_model.model_params_,
        bandwidth=bw,
        dist_info=dist_info,
        n_boot=10
    )

    assert len(lower) == len(X_eval)
    assert len(upper) == len(X_eval)
    assert np.all(upper >= lower)
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)

def test_bootstrap_kriging_path(linear_data):
    """Explicitly test the Kriging conditional logic inside the bootstrap loop."""
    X, y = linear_data
    kernel = RBF(1.0)
    X_eval = np.linspace(0.1, 5.0, 10).reshape(-1, 1)

    lower, upper = bootstrap_pod_ci(
        X.reshape(-1, 1), y, X_eval,
        threshold=5.0,
        model_type='Kriging',
        model_params=kernel,
        bandwidth=0.5,
        dist_info=('norm', (0, 1)),
        n_boot=2
    )

    assert len(lower) == 10
    assert np.all(upper >= lower)


# --- 6. MISC TESTS ---

@patch("digiqual.pod.cross_val_score")
def test_fit_all_robust_mean_models_kriging(mock_cv, linear_data):
    X, y = linear_data
    # Force Polynomial scores to be terrible (-1000) and Kriging to be great (-1)
    def side_effect(model, *args, **kwargs):
        if hasattr(model, 'kernel'):
            return np.array([-1, -1])  # Kriging
        else:
            return np.array([-1000, -1000])  # Polynomial
    mock_cv.side_effect = side_effect

    models, scores, best_key = fit_all_robust_mean_models(X, y, max_degree=1, n_folds=2)
    assert best_key == ('Kriging', None)
    assert models[best_key].model_type_ == 'Kriging'

def test_infer_best_distribution_exception():
    X = np.array([1, 2, 3])
    # Pass NaNs to force exception in scipy.stats.*.fit
    residuals = np.array([np.nan, np.nan, np.nan])
    dist_name, params = infer_best_distribution(residuals, X, bandwidth=1.0)
    assert dist_name == 'norm'
    assert params == (0, 1)

def test_calculate_reliability_point_nan():
    X_eval = np.array([[1], [2], [3]])
    ci_lower = np.array([0.1, 0.2, 0.3]) # target is 0.9 by default
    res = calculate_reliability_point(X_eval, ci_lower, target_pod=0.9)
    assert np.isnan(res)

def test_plot_model_selection_highlights_forced(linear_data):
    """Plot should render without error when used_key differs from cv_winner_key."""
    X, y = linear_data
    models, scores, best_key = fit_all_robust_mean_models(X, y)

    # Simulate a user forcing a Polynomial 1, even if it didn't win
    fig = plot_model_selection(
        scores,
        used_key=('Polynomial', 1),
        cv_winner_key=best_key
    )
    assert fig is not None
    assert len(fig.axes) == 2
