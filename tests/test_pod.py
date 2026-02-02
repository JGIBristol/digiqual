import pytest
import numpy as np
import scipy.stats as stats
from sklearn.pipeline import Pipeline
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
    # y = 2x + 1 + noise
    y = 2.0 * X + 1.0 + np.random.normal(0, 0.5, size=len(X))
    return X, y

@pytest.fixture
def quadratic_data():
    """Generates quadratic data to test model selection (degree detection)."""
    np.random.seed(42)
    X = np.linspace(0.1, 5.0, 50)
    # y = 0.5x^2 + noise
    y = 0.5 * X**2 + np.random.normal(0, 0.2, size=len(X))
    return X, y

@pytest.fixture
def heteroscedastic_data():
    """Generates data where noise increases with X (to test variance model)."""
    np.random.seed(42)
    X = np.linspace(0.1, 5.0, 100)
    # Noise grows from 0.1 to 1.0 as X increases
    noise_level = 0.2 * X
    y = 3.0 * X + np.random.normal(0, 1, size=len(X)) * noise_level
    return X, y

@pytest.fixture
def gumbel_data():
    """Generates data with Gumbel-distributed residuals (non-Gaussian)."""
    np.random.seed(42)
    X = np.linspace(0.1, 3.0, 100)
    # Gumbel (Right Skewed) Noise
    noise = stats.gumbel_r.rvs(loc=0, scale=0.5, size=len(X))
    y = 2.0 * X + noise
    return X, y

# --- 1. MEAN MODEL TESTS (Robust Polynomial) ---

def test_fit_robust_mean_model_selection(quadratic_data):
    """Test if Cross-Validation correctly identifies a non-linear relationship."""
    X, y = quadratic_data

    # We expect the robust selector to pick degree >= 2 (likely 2)
    model = fit_robust_mean_model(X, y, max_degree=5, n_folds=5)

    assert isinstance(model, Pipeline)
    assert hasattr(model, 'best_degree_')
    # It should definitely not pick 1 (linear) for quadratic data
    assert model.best_degree_ >= 2

def test_fit_robust_mean_model_shapes(linear_data):
    """Test it handles 1D array shapes correctly and predicts smoothly."""
    X, y = linear_data
    model = fit_robust_mean_model(X, y, max_degree=3)

    # Predict should work on 2D input (sklearn standard)
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
    """Test that residuals, bandwidth, and grid are generated correctly."""
    X, y = heteroscedastic_data
    mean_model = fit_robust_mean_model(X, y)

    residuals, bw, X_eval = fit_variance_model(X, y, mean_model, n_eval_points=20)

    assert len(residuals) == len(y)
    assert isinstance(bw, float)
    assert bw > 0
    # Grid checks
    assert len(X_eval) == 20
    assert X_eval.min() == X.min()
    assert X_eval.max() == X.max()

def test_predict_local_std_heteroscedasticity(heteroscedastic_data):
    """Test that predicted noise actually increases with X for heteroscedastic data."""
    X, y = heteroscedastic_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, _ = fit_variance_model(X, y, mean_model)

    # Predict noise at small X (should be low) and large X (should be high)
    eval_points = np.array([0.5, 4.5])
    predicted_std = predict_local_std(X, residuals, eval_points, bw)

    assert len(predicted_std) == 2
    assert np.all(predicted_std > 0) # Standard deviation must be positive

    # Noise at 4.5 should be significantly higher than at 0.5
    assert predicted_std[1] > predicted_std[0]

# --- 3. DISTRIBUTION INFERENCE TESTS (AIC Selection) ---

def test_infer_best_distribution_gaussian(linear_data):
    """Test that standard Gaussian noise is correctly identified as 'norm'."""
    X, y = linear_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, _ = fit_variance_model(X, y, mean_model)

    dist_name, params = infer_best_distribution(residuals, X, bw)

    # Depending on random seed, might pick t or logistic (similar shapes),
    # but for pure normal data 'norm' is the expected winner.
    assert dist_name in ['norm', 't', 'logistic']
    assert len(params) >= 2 # loc, scale, (shape)

def test_infer_best_distribution_gumbel(gumbel_data):
    """Test that skewed data is identified as Gumbel."""
    X, y = gumbel_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, _ = fit_variance_model(X, y, mean_model)

    dist_name, params = infer_best_distribution(residuals, X, bw)

    assert dist_name == 'gumbel_r' # Should detect right skew
    assert isinstance(params, tuple)

def test_infer_best_distribution_safety():
    """Test robustness against weird/negative residuals (e.g., if Weibull was still there)."""
    np.random.seed(123)
    X = np.linspace(0, 10, 20)
    residuals = np.random.normal(0, 1, 20) # Contains negatives

    # Should not crash even with negatives
    dist_name, params = infer_best_distribution(residuals, X, bandwidth=1.0)
    assert isinstance(dist_name, str)

# --- 4. POD COMPUTATION TESTS ---

def test_compute_pod_curve_bounds(linear_data):
    """Test that PoD values are strictly bounded between 0 and 1."""
    X, y = linear_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, X_eval = fit_variance_model(X, y, mean_model)

    # Force a distribution for testing
    dist_info = ('norm', (0, 1))

    pod_curve, mean_curve = compute_pod_curve(
        X_eval, mean_model, X, residuals, bw, dist_info, threshold=5.0
    )

    assert len(pod_curve) == len(X_eval)
    assert np.all(pod_curve >= 0.0)
    assert np.all(pod_curve <= 1.0)
    # Mean curve should follow roughly the data trend
    assert np.mean(mean_curve) > 0

def test_compute_pod_curve_integration(gumbel_data):
    """Test full integration: Inference -> Prediction."""
    X, y = gumbel_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, X_eval = fit_variance_model(X, y, mean_model)

    # 1. Infer
    dist_info = infer_best_distribution(residuals, X, bw)

    # 2. Compute
    pod_curve, _ = compute_pod_curve(
        X_eval, mean_model, X, residuals, bw, dist_info, threshold=4.0
    )

    assert not np.any(np.isnan(pod_curve))

# --- 5. BOOTSTRAP TESTS ---

def test_bootstrap_pod_ci_structure(linear_data):
    """Test that bootstrap returns valid upper/lower bounds."""
    X, y = linear_data
    mean_model = fit_robust_mean_model(X, y)
    residuals, bw, X_eval = fit_variance_model(X, y, mean_model)
    dist_info = ('norm', (0, 1))

    # Run small bootstrap (n_boot=10) for speed
    lower, upper = bootstrap_pod_ci(
        X, y, X_eval,
        threshold=5.0,
        degree=mean_model.best_degree_,
        bandwidth=bw,
        dist_info=dist_info,
        n_boot=10
    )

    assert len(lower) == len(X_eval)
    assert len(upper) == len(X_eval)

    # Logical check: Upper bound must be >= Lower bound
    assert np.all(upper >= lower)

    # Logical check: Bounds must be within [0, 1]
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)



# --- 6. INTEGRATION TEST (Core Class Wiring) ---

def test_simulation_study_pod_integration(linear_data):
    """
    Test the full 'analyze_reliability' workflow via the main SimulationStudy class.
    This ensures core.py correctly connects to pod.py.
    """
    X, y = linear_data

    # 1. Setup: Create a study populated with data
    df = pd.DataFrame({'Crack_Len': X, 'Signal': y, 'Noise_Factor': np.random.rand(len(X))})

    study = SimulationStudy(input_cols=['Crack_Len', 'Noise_Factor'], outcome_col='Signal')
    study.add_data(df)

    # 2. Action: Run the analysis (using low n_boot for speed)
    # We deliberately use a threshold that sits in the middle of the y range
    results = study.pod(
        poi_col='Crack_Len',
        threshold=5.0,
        n_boot=20
    )

    # 3. Assertions: Check the "Contract" (Output Dictionary Structure)
    assert isinstance(results, dict)

    # Check Inputs were preserved
    assert results['poi_col'] == 'Crack_Len'
    assert results['threshold'] == 5.0

    # Check Models
    assert hasattr(results['mean_model'], 'predict')
    assert isinstance(results['bandwidth'], float)

    # Check Curves (The most important output for the GUI/Plotting)
    curves = results['curves']
    assert 'pod' in curves
    assert 'ci_lower' in curves
    assert 'ci_upper' in curves
    assert 'mean_response' in curves

    # Check Dimensions (Should match default evaluation grid of 100)
    assert len(curves['pod']) == 100
    assert len(curves['ci_lower']) == 100

    # Check Internal State Update
    # The study object should remember the results
    assert study.pod_results is results
