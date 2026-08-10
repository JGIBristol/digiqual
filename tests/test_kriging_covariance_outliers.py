import pytest
import numpy as np
from digiqual.pod import (
    fit_all_robust_mean_models,
    compute_kriging_loo_residuals,
    fit_variance_model,
    plot_model_selection
)
from digiqual.plotting import plot_kriging_diagnostics

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
