import numpy as np
import matplotlib.pyplot as plt
from digiqual.plotting import plot_signal_model, plot_pod_curve, plot_pod_surface, plot_signal_surface

def test_plot_signal_model_no_ax():
    X = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    X_eval = np.array([1, 2, 3])
    mean_curve = np.array([1, 2, 3])

    # without local_std
    ax = plot_signal_model(X, y, X_eval, mean_curve, threshold=2.0)
    assert isinstance(ax, plt.Axes)

    # with local_std
    local_std = np.array([0.1, 0.1, 0.1])
    ax2 = plot_signal_model(X, y, X_eval, mean_curve, threshold=2.0, local_std=local_std)
    assert isinstance(ax2, plt.Axes)

def test_plot_pod_curve_no_ax():
    X_eval = np.array([1, 2, 3])
    pod_curve = np.array([0.1, 0.5, 0.9])

    # without bounds
    ax = plot_pod_curve(X_eval, pod_curve)
    assert isinstance(ax, plt.Axes)

    # with bounds, reaching target
    ci_lower = np.array([0.0, 0.4, 0.95])
    ci_upper = np.array([0.2, 0.6, 1.0])
    ax2 = plot_pod_curve(X_eval, pod_curve, ci_lower=ci_lower, ci_upper=ci_upper, target_pod=0.9)
    assert isinstance(ax2, plt.Axes)

    # with bounds, not reaching target (to hit line `if np.max(ci_lower) >= target_pod:` False path)
    ci_lower_low = np.array([0.0, 0.1, 0.2])
    ax3 = plot_pod_curve(X_eval, pod_curve, ci_lower=ci_lower_low, ci_upper=ci_upper, target_pod=0.9)
    assert isinstance(ax3, plt.Axes)



def test_plot_pod_surface_no_ax():
    """Test the 2D contour plot for multi-dimensional PoD."""
    poi_grids = [np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
    pod_curve = np.random.rand(25) # 5x5 grid flattened
    poi_names = ["Length", "Angle"]

    ax = plot_pod_surface(poi_grids, pod_curve, poi_names)
    assert isinstance(ax, plt.Axes)

def test_plot_signal_surface_no_ax():
    """Test the 3D surface plot for multi-dimensional signals."""
    poi_grids = [np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
    mean_curve = np.random.rand(25)
    X_raw = np.random.rand(10, 2)
    y_raw = np.random.rand(10)
    threshold = 0.5
    poi_names = ["Length", "Angle"]

    ax = plot_signal_surface(poi_grids, mean_curve, X_raw, y_raw, threshold, poi_names)
    assert isinstance(ax, plt.Axes)
