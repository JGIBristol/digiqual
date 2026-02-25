import pytest
import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process.kernels import RBF
from digiqual.pod import (
    fit_robust_mean_model,
    fit_variance_model,
    predict_local_std,
    infer_best_distribution,
    compute_pod_curve,
    bootstrap_pod_ci
)
import pandas as pd
from digiqual.core import SimulationStudy

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

def test_fit_robust_mean_model_selection(quadratic_data):
    """Test if Cross-Validation correctly identifies model type and params."""
    X, y = quadratic_data
    model = fit_robust_mean_model(X, y, max_degree=5, n_folds=5)

    # Check for the new dynamic attributes
    assert hasattr(model, 'model_type_')
    assert hasattr(model, 'model_params_')
    assert model.model_type_ in ['Polynomial', 'Kriging']

    # If it chose polynomial for quadratic data, degree should be >= 2
    if model.model_type_ == 'Polynomial':
        assert model.model_params_ >= 2

def test_fit_robust_mean_model_shapes(linear_data):
    """Test it handles 1D array shapes correctly and predicts smoothly."""
    X, y = linear_data
    model = fit_robust_mean_model(X, y, max_degree=3)

    X_eval = np.array([[1.0], [2.0], [3.0]])
    preds = model.predict(X_eval)

    assert len(preds) == 3
    assert isinstance(preds, np.ndarray)

def test_fit_robust_mean_model_plotting(linear_data, monkeypatch):
    """Test that plot_cv=True runs without crashing (mocking plt.show)."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)

    X, y = linear_data
    fit_robust_mean_model(X, y, max_degree=3, plot_cv=True)

# --- 2. VARIANCE MODEL TESTS (Kernel Smoothing) ---

def test_fit_variance_model_outputs(heteroscedastic_data):
    X, y = heteroscedastic_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, X_eval = fit_variance_model(X, y, mean_model, n_eval_points=20)

    assert len(residuals) == len(y)
    assert isinstance(bw, float)
    assert bw > 0
    assert len(X_eval) == 20

def test_predict_local_std_heteroscedasticity(heteroscedastic_data):
    X, y = heteroscedastic_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, _ = fit_variance_model(X, y, mean_model)

    eval_points = np.array([0.5, 4.5])
    predicted_std = predict_local_std(X, residuals, eval_points, bw)

    assert len(predicted_std) == 2
    assert np.all(predicted_std > 0)
    assert predicted_std[1] > predicted_std[0]

# --- 3. DISTRIBUTION INFERENCE TESTS (AIC Selection) ---

def test_infer_best_distribution_gaussian(linear_data):
    X, y = linear_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, _ = fit_variance_model(X, y, mean_model)
    dist_name, params = infer_best_distribution(residuals, X, bw)

    assert dist_name in ['norm', 't', 'logistic']
    assert len(params) >= 2

def test_infer_best_distribution_gumbel(gumbel_data):
    X, y = gumbel_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, _ = fit_variance_model(X, y, mean_model)
    dist_name, params = infer_best_distribution(residuals, X, bw)

    assert dist_name == 'gumbel_r'
    assert isinstance(params, tuple)

def test_infer_best_distribution_safety():
    np.random.seed(123)
    X = np.linspace(0, 10, 20)
    residuals = np.random.normal(0, 1, 20)
    dist_name, params = infer_best_distribution(residuals, X, bandwidth=1.0)
    assert isinstance(dist_name, str)

# --- 4. POD COMPUTATION TESTS ---

def test_compute_pod_curve_bounds(linear_data):
    X, y = linear_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, X_eval = fit_variance_model(X, y, mean_model)
    dist_info = ('norm', (0, 1))

    pod_curve, mean_curve = compute_pod_curve(
        X_eval, mean_model, X, residuals, bw, dist_info, threshold=5.0
    )

    assert len(pod_curve) == len(X_eval)
    assert np.all(pod_curve >= 0.0)
    assert np.all(pod_curve <= 1.0)
    assert np.mean(mean_curve) > 0

# --- 5. BOOTSTRAP TESTS ---

def test_bootstrap_pod_ci_structure(linear_data):
    """Test that bootstrap returns valid upper/lower bounds using dynamic models."""
    X, y = linear_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, X_eval = fit_variance_model(X, y, mean_model)
    dist_info = ('norm', (0, 1))

    # Run small bootstrap (n_boot=10) for speed using new signature
    lower, upper = bootstrap_pod_ci(
        X, y, X_eval,
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
    X_eval = np.linspace(0.1, 5.0, 10)

    lower, upper = bootstrap_pod_ci(
        X, y, X_eval,
        threshold=5.0,
        model_type='Kriging',
        model_params=kernel,
        bandwidth=0.5,
        dist_info=('norm', (0, 1)),
        n_boot=2
    )

    assert len(lower) == 10
    assert np.all(upper >= lower)

# --- 6. INTEGRATION TEST (Core Class Wiring) ---

def test_simulation_study_pod_integration(linear_data):
    """Test the full workflow via the main SimulationStudy class."""
    X, y = linear_data
    df = pd.DataFrame({'Crack_Len': X, 'Signal': y, 'Noise_Factor': np.random.rand(len(X))})

    study = SimulationStudy(input_cols=['Crack_Len', 'Noise_Factor'], outcome_col='Signal')
    study.add_data(df)

    results = study.pod(poi_col='Crack_Len', threshold=5.0, n_boot=20)

    assert isinstance(results, dict)
    assert results['poi_col'] == 'Crack_Len'

    # Check new dynamic attributes
    assert hasattr(results['mean_model'], 'model_type_')
    assert hasattr(results['mean_model'], 'model_params_')

    curves = results['curves']
    assert 'pod' in curves
    assert 'ci_lower' in curves
    assert 'ci_upper' in curves
    assert len(curves['pod']) == 100
