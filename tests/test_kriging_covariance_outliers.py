import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel

from digiqual.plotting import plot_kriging_diagnostics
from digiqual.pod import (
    build_fixed_kernel_gpr,
    compute_kriging_loo_residuals,
    fit_all_robust_mean_models,
    fit_variance_model,
    plot_model_selection,
)

KERNEL_NAMES = {'Exponential (Matern 1/2)', 'Matern 3/2', 'Matern 5/2', 'Gaussian (RBF)'}

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
    assert hasattr(gpr, 'kernel_loo_scores_')
    assert len(gpr.kernel_loo_scores_) >= 3
    assert gpr.best_kernel_name_ in KERNEL_NAMES

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

    # No length scale has collapsed relative to its input's range. Length scales are
    # in standardised units (inputs divided by their std), so compare like with like.
    length_scales = np.atleast_1d(gpr.kernel_.k1.k2.length_scale)
    ranges = np.ptp(X, axis=0) / gpr.x_std_
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
        preds.append(build_fixed_kernel_gpr(gpr.model_params_).fit(X[idx], y[idx]).predict(probe))
    preds = np.array(preds)
    rel_sd = preds.std(axis=0) / np.abs(preds.mean(axis=0))
    assert rel_sd.max() < 0.3


def test_kriging_all_candidate_kernels_fit():
    """Every candidate kernel from [M26] Sec. 4.3.1 (minus 'linear') is scored by LOO MSE."""
    X, y = _hidden_variable_dataset(n=60)
    models, _, _ = fit_all_robust_mean_models(X, y)
    gpr = models[('Kriging', None)]
    assert set(gpr.kernel_loo_scores_) == KERNEL_NAMES
    # The kept kernel is the LOO-MSE minimiser ([M26] Sec. 2.2.2)
    assert gpr.best_kernel_name_ == min(gpr.kernel_loo_scores_, key=gpr.kernel_loo_scores_.get)
    # Backwards-compatible alias
    assert gpr.kernel_cv_scores_ is gpr.kernel_loo_scores_


def test_kriging_loo_residuals_in_original_units():
    """LOO means/stds are returned in the response's units even with normalize_y=True."""
    X, y = _hidden_variable_dataset(n=80)
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    loo_means, loo_stds, _, _ = compute_kriging_loo_residuals(gpr, X, y)
    assert abs(np.mean(loo_means) - np.mean(y)) < 0.05
    assert np.sqrt(np.mean((y - loo_means) ** 2)) < np.std(y)
    assert np.all(loo_stds < np.std(y))


# --- Provenance-driven behaviour: [UQLab] scaling, [M26] LOO, [M25] 1-SE rule ---

def test_kriging_is_invariant_to_input_units():
    """[UQLab] Scaling=true: standardising X makes the fit independent of input units."""
    X, y = _hidden_variable_dataset(n=80)
    gpr_a = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    gpr_b = fit_all_robust_mean_models(X * 1e-3, y)[0][('Kriging', None)]

    assert gpr_a.best_kernel_name_ == gpr_b.best_kernel_name_
    for k in gpr_a.kernel_loo_scores_:
        assert np.isclose(gpr_a.kernel_loo_scores_[k], gpr_b.kernel_loo_scores_[k], rtol=1e-3)
    probe = np.percentile(X, [10, 50, 90], axis=0)
    assert np.allclose(gpr_a.predict(probe), gpr_b.predict(probe * 1e-3), rtol=1e-3, atol=1e-6)


def test_kriging_loo_matches_explicit_refits_without_nugget_effects():
    """[M26] Eq. 10: the matrix LOO mean equals an ordinary-Kriging refit without point i
    (same hyperparameters). Checked against an explicit GLS ordinary-Kriging predictor."""
    X, y = _hidden_variable_dataset(n=40)
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    loo_means, _, _, _ = compute_kriging_loo_residuals(gpr, X, y)

    U = gpr.transform_X(X)
    K = gpr.kernel_(U)
    y_n = (y - gpr._y_train_mean) / gpr._y_train_std
    for i in [0, 7, 23]:
        keep = np.arange(len(y)) != i
        Ki = K[np.ix_(keep, keep)]
        ones = np.ones(keep.sum())
        Kinv_y = np.linalg.solve(Ki, y_n[keep])
        Kinv_1 = np.linalg.solve(Ki, ones)
        beta = ones @ Kinv_y / (ones @ Kinv_1)                     # [M26] Eq. 6
        k0 = gpr.kernel_.k1(U[[i]], U[keep]).ravel()               # cross-covariance (no nugget off-diagonal)
        mu = beta + k0 @ np.linalg.solve(Ki, y_n[keep] - beta)     # [M26] Eq. 8
        assert np.isclose(mu * gpr._y_train_std + gpr._y_train_mean, loo_means[i], rtol=1e-6, atol=1e-8)


def test_group_loo_reduces_to_loo_for_singletons():
    from digiqual.pod import compute_kriging_group_loo_means
    X, y = _hidden_variable_dataset(n=40)
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    loo_means, _, _, _ = compute_kriging_loo_residuals(gpr, X, y)
    group_means = compute_kriging_group_loo_means(gpr, X, y, groups=np.arange(len(y)))
    assert np.allclose(loo_means, group_means)


def test_gamma_is_diagnostic_only_and_kriging_uses_loo_residuals():
    """gamma ([M26] Sec. 2.2.5.2) no longer scales the variance-model residuals; Kriging
    residuals are LOO residuals ([digiqual])."""
    np.random.seed(42)
    X = np.linspace(0.5, 5.0, 60).reshape(-1, 1)
    y = 2.0 * X.flatten() + np.random.normal(0, 0.2, size=60)
    y[15] += 10.0
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    assert gpr.outlier_scale_factor_ > 1.0

    residuals, _ = fit_variance_model(X, y, gpr)
    assert np.allclose(residuals, y - gpr.loo_means_)
    # LOO residuals are at least as large as in-sample ones on average
    assert np.mean(residuals ** 2) >= np.mean((y - gpr.predict(X)) ** 2)


def test_frozen_refit_reproduces_production_model():
    X, y = _hidden_variable_dataset(n=60)
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    refit = build_fixed_kernel_gpr(gpr.model_params_).fit(X, y)
    probe = np.percentile(X, [5, 50, 95], axis=0)
    assert np.allclose(refit.predict(probe), gpr.predict(probe))
    assert np.allclose(refit.x_std_, gpr.x_std_)


def test_kriging_bootstrap_smoke():
    from digiqual.pod import bootstrap_pod_ci, infer_best_distribution
    X, y = _hidden_variable_dataset(n=60)
    gpr = fit_all_robust_mean_models(X, y)[0][('Kriging', None)]
    residuals, bw = fit_variance_model(X, y, gpr)
    dist = infer_best_distribution(residuals, X, bw)
    X_eval = np.linspace(X[:, 0].min(), X[:, 0].max(), 15).reshape(-1, 1)
    lo, hi = bootstrap_pod_ci(
        X, y, X_eval, threshold=float(np.median(y)), model_type='Kriging',
        model_params=gpr.model_params_, bandwidth=bw, dist_info=dist, n_boot=20,
        nuisance_ranges={'offset': (float(np.median(X[:, 1])),) * 2}, n_jobs=1,
        feature_names=['area', 'offset'], poi_names=['area'], n_mc_samples=10,
    )
    assert lo.shape == hi.shape == (15,)
    assert np.all(lo <= hi + 1e-12)


def test_one_se_rule_prefers_simplest_near_tie():
    """[M25] Fig. 5: among near-equal CV errors the simplest model is chosen."""
    from digiqual.pod import select_cv_winner
    scores = {('Polynomial', 1): 2.0, ('Polynomial', 3): 1.02, ('Polynomial', 5): 1.00, ('Kriging', None): 0.99}
    se = {k: 0.05 for k in scores}
    assert select_cv_winner(scores, se) == ('Polynomial', 3)
    # A clear gap: the strict minimum still wins
    se_small = {k: 0.001 for k in scores}
    assert select_cv_winner(scores, se_small) == ('Kriging', None)
    # Without SEs, fall back to the strict minimum
    assert select_cv_winner(scores) == ('Kriging', None)


def test_cv_standard_errors_attached_to_models():
    X, y = _hidden_variable_dataset(n=60)
    models, scores, winner = fit_all_robust_mean_models(X, y, max_degree=4)
    cv_se = models[winner].cv_se_
    assert set(cv_se) == set(scores)
    assert all(v >= 0 for v in cv_se.values())
    fig = plot_model_selection(scores, cv_winner_key=winner, cv_se=cv_se)
    assert fig.axes
