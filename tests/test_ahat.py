import pytest
import numpy as np
import matplotlib.pyplot as plt
from digiqual.ahat import (
    fit_linear_a_hat_model,
    compute_linear_pod_curve,
    bootstrap_linear_pod_ci,
    plot_linear_signal_model
)
from sklearn.linear_model import LinearRegression

# --- Fixtures ---

@pytest.fixture
def linear_data():
    """Generates standard linear data with constant noise."""
    np.random.seed(42)
    X = np.linspace(1.0, 10.0, 50)
    # y = mx + c + noise
    y = 2.0 * X + 1.0 + np.random.normal(0, 0.5, 50)
    return X, y

@pytest.fixture
def log_linear_data():
    """Generates data that requires a log-log transformation to become linear."""
    np.random.seed(42)
    X = np.linspace(1.0, 10.0, 50)
    # log(y) = 2.0 * log(X) + noise -> y = exp(2.0 * log(X) + noise)
    y = np.exp(2.0 * np.log(X) + np.random.normal(0, 0.2, 50))
    return X, y

# --- 1. Fitting Tests ---

def test_fit_linear_model_basic(linear_data):
    X, y = linear_data
    model, tau = fit_linear_a_hat_model(X, y, xlog=False, ylog=False)

    assert isinstance(model, LinearRegression)
    assert tau > 0
    assert isinstance(tau, float)

def test_fit_linear_model_log_transforms(log_linear_data):
    X, y = log_linear_data
    model, tau = fit_linear_a_hat_model(X, y, xlog=True, ylog=True)

    assert isinstance(model, LinearRegression)
    assert tau > 0

# --- 2. Curve Generation Tests ---

def test_compute_linear_pod_curve(linear_data):
    X, y = linear_data
    model, tau = fit_linear_a_hat_model(X, y)

    X_eval = np.linspace(1.0, 10.0, 100)
    threshold = 15.0 # Expected to cross somewhere in the middle

    pod_curve, mean_curve = compute_linear_pod_curve(
        X_eval, model, tau, threshold=threshold, xlog=False, ylog=False
    )

    assert len(pod_curve) == 100
    assert len(mean_curve) == 100
    # Probabilities must be bounded [0, 1]
    assert np.all((pod_curve >= 0.0) & (pod_curve <= 1.0))
    # Mean curve should strictly increase for this specific data
    assert np.all(np.diff(mean_curve) > 0)

def test_compute_linear_pod_curve_log_transforms(log_linear_data):
    X, y = log_linear_data
    model, tau = fit_linear_a_hat_model(X, y, xlog=True, ylog=True)

    X_eval = np.linspace(1.0, 10.0, 100)

    pod_curve, mean_curve = compute_linear_pod_curve(
        X_eval, model, tau, threshold=50.0, xlog=True, ylog=True
    )

    assert len(pod_curve) == 100
    assert len(mean_curve) == 100
    assert np.all((pod_curve >= 0.0) & (pod_curve <= 1.0))

# --- 3. Bootstrap Tests ---

def test_bootstrap_linear_pod_ci(linear_data):
    X, y = linear_data
    X_eval = np.linspace(1.0, 10.0, 20) # Smaller grid for faster testing

    # Run a quick bootstrap with 10 iterations
    lower_ci, upper_ci = bootstrap_linear_pod_ci(
        X, y, X_eval, threshold=15.0, xlog=False, ylog=False, n_boot=10, n_jobs=1
    )

    assert len(lower_ci) == 20
    assert len(upper_ci) == 20
    # Upper bound should be strictly greater than or equal to lower bound
    assert np.all(upper_ci >= lower_ci)
    assert np.all((lower_ci >= 0.0) & (upper_ci <= 1.0))

# --- 4. Plotting Tests ---

def test_plot_linear_signal_model_generation(linear_data):
    X, y = linear_data
    model, tau = fit_linear_a_hat_model(X, y)
    X_eval = np.linspace(1.0, 10.0, 50)

    ax = plot_linear_signal_model(
        X, y, X_eval, model, threshold=15.0, tau=tau,
        xlog=False, ylog=False, poi_name="Flaw Size"
    )

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Flaw Size"
    assert ax.get_ylabel() == "Signal Response"

def test_plot_linear_signal_model_log_scaling(log_linear_data):
    """Ensures the plot axes scale correctly when log transforms are active."""
    X, y = log_linear_data
    model, tau = fit_linear_a_hat_model(X, y, xlog=True, ylog=True)
    X_eval = np.linspace(1.0, 10.0, 50)

    ax = plot_linear_signal_model(
        X, y, X_eval, model, threshold=50.0, tau=tau,
        xlog=True, ylog=True
    )

    assert isinstance(ax, plt.Axes)
    # Check that Matplotlib applied the correct log scaling
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'log'
