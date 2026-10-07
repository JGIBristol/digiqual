import pytest
import numpy as np
from digiqual.pod import (
    fit_all_robust_mean_models,
    compute_kriging_loo_residuals,
    fit_variance_model,
    plot_model_selection
)
from digiqual.plotting import plot_kriging_diagnostics
from digiqual.pod import build_fixed_kernel_gpr
from sklearn.gaussian_process.kernels import WhiteKernel

def test_kriging_covariance_optimization_and_anisotropy():
    """Test anisotropic Kriging kernel candidate evaluation (Rank 2)."""
    np.random.seed(42)
    N = 80
    # Create 2D input space with different scales and sensitivities per dimension
    x1 = np.linspace(0.1, 5.0, N)
    x2 = np.linspace(-10.0, 10.0, N)
    X = np.column_stack([x1, x2])
    
    # Target signal response with different sensitivities in x1 vs x2
    y = 3.0 * x1 + 0.05 * x2 + np.random.normal(0, 0.2, size=N)

    models, scores, cv_winner_key = fit_all_robust_mean_models(X, y)
    
    # Kriging should be fitted and available in models
    assert ('Kriging', None) in models
    gpr = models[('Kriging', None)]
    
    assert hasattr(gpr, 'best_kernel_name_')
    assert hasattr(gpr, 'kernel_cv_scores_')
    assert len(gpr.kernel_cv_scores_) >= 3
    assert gpr.best_kernel_name_ in ['Matern 3/2', 'Matern 5/2', 'RBF (Gaussian)', 'Rational Quadratic']
    
    # Anisotropic length scales should be present
    assert hasattr(gpr.kernel_, 'k2') or hasattr(gpr.kernel_, 'length_scale')

def test_kriging_standardized_loo_residuals_and_outliers():
    """Test standardized LOO residual calculation and outlier scaling factor (Rank 4)."""
    np.random.seed(42)
    N = 60
    X = np.linspace(0.5, 5.0, N).reshape(-1, 1)
    y = 2.0 * X.flatten() + np.random.normal(0, 0.2, size=N)
    
    # Inject an extreme outlier to trigger gamma > 1.0 calibration
    y[15] += 10.0

    models, scores, cv_winner_key = fit_all_robust_mean_models(X, y)
    gpr = models[('Kriging', None)]

    loo_means, loo_stds, std_residuals, gamma = compute_kriging_loo_residuals(gpr, X, y)

    assert len(std_residuals) == N
    assert hasattr(gpr, 'outlier_scale_factor_')
    assert gpr.outlier_scale_factor_ > 1.0  # Outlier factor should be triggered (> 1.0)
    assert np.isclose(gpr.outlier_scale_factor_, gamma)

    # Test fit_variance_model applies the scaling factor
    residuals, bw = fit_variance_model(X, y, gpr)
    assert len(residuals) == N

def test_plot_kriging_diagnostics():
    """Test plot_kriging_diagnostics generates matplotlib axes."""
    std_residuals = np.random.normal(0, 1, 50)
    # Add an outlier
    std_residuals[5] = 4.5
    gamma = 1.5

    ax = plot_kriging_diagnostics(std_residuals, outlier_scale_factor=gamma, best_kernel_name="Matérn 5/2")
    assert ax is not None


def _hidden_variable_dataset(n=200, seed=0):
    """Response saturates with Area; scatter comes from an unobserved shape
    variable, and Offset has no effect. Mirrors training on (Area, Offset) only."""
    rng = np.random.default_rng(seed)
    radial = rng.uniform(0.1, 4.0, n)
    axial = rng.uniform(0.1, 4.0, n)
    offset = rng.uniform(-0.25, 0.25, n)
    area = 0.785 * radial * axial
    aspect = axial / (radial + axial)
    y = 0.5 * (1 - np.exp(-area / 2.0)) * (0.7 + 0.6 * aspect)
    return np.column_stack([area, offset]), y


def test_kriging_learns_noise_instead_of_overfitting_hidden_scatter():
    """Regression: with a fixed, too-small nugget the GP collapsed the Offset length
    scale onto scatter caused by an unobserved variable. Slices then fell back to 0
    away from data, and fixed-kernel bootstrap refits were unstable."""
    X, y = _hidden_variable_dataset()
    models, _, _ = fit_all_robust_mean_models(X, y)
    gpr = models[('Kriging', None)]

    # Noise is learned as part of the kernel, and is non-negligible here
    assert isinstance(gpr.kernel_.k2, WhiteKernel)
    assert gpr.kernel_.k2.noise_level > 1e-3
    assert gpr.normalize_y

    # No length scale has collapsed relative to its input's range
    length_scales = np.atleast_1d(gpr.kernel_.k1.k2.length_scale)
    ranges = np.ptp(X, axis=0)
    assert np.all(length_scales / ranges[:len(length_scales)] > 0.05)

    # A slice at a fixed Offset tracks the data at large Area instead of reverting to 0
    hi = X[:, 0] > np.percentile(X[:, 0], 90)
    slice_pred = gpr.predict(np.column_stack([X[hi, 0], np.full(hi.sum(), np.median(X[:, 1]))]))
    assert np.all(np.abs(slice_pred - y[hi].mean()) < 0.1)

    # Fixed-kernel bootstrap refits (as used for PoD CIs and the GUI trace) are stable
    probe = np.percentile(X, [10, 50, 90], axis=0)
    rng = np.random.default_rng(42)
    preds = []
    for _ in range(30):
        idx = rng.choice(len(y), len(y), replace=True)
        preds.append(build_fixed_kernel_gpr(gpr.kernel_).fit(X[idx], y[idx]).predict(probe))
    preds = np.array(preds)
    rel_sd = preds.std(axis=0) / np.abs(preds.mean(axis=0))
    assert rel_sd.max() < 0.3


def test_kriging_all_candidate_kernels_fit():
    """Every candidate kernel (including the isotropic Rational Quadratic) is scored."""
    X, y = _hidden_variable_dataset(n=60)
    models, _, _ = fit_all_robust_mean_models(X, y)
    assert set(models[('Kriging', None)].kernel_cv_scores_) == {
        'Matern 3/2', 'Matern 5/2', 'RBF (Gaussian)', 'Rational Quadratic'
    }


def test_kriging_loo_residuals_in_original_units():
    """LOO means/stds are returned in the response's units even with normalize_y=True."""
    X, y = _hidden_variable_dataset(n=80)
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    loo_means, loo_stds, _, _ = compute_kriging_loo_residuals(gpr, X, y)
    assert abs(np.mean(loo_means) - np.mean(y)) < 0.05
    assert np.sqrt(np.mean((y - loo_means) ** 2)) < np.std(y)
    assert np.all(loo_stds < np.std(y))
