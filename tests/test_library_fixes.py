"""Regression tests for library bug fixes (equation, n_jobs, caching settings,
adaptive loop, input guards, reliability point, nuisance truncation)."""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from digiqual._parallel import resolve_n_jobs
from digiqual.core import SimulationStudy
from digiqual.cpp_fallback import compute_pod_probs_fast, predict_local_std_fast
from digiqual.integration import compute_multi_dim_pod
from digiqual.pod import (
    calculate_reliability_point,
    fit_all_robust_mean_models,
    generate_latex_equation,
    polynomial_raw_coefficients,
)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


@pytest.fixture
def study_3d():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "Length": rng.uniform(0.5, 5, n),
        "Angle": rng.uniform(-15, 15, n),
        "Rough": rng.uniform(0, 1, n),
    })
    df["Signal"] = 2 * df.Length - 0.2 * df.Length ** 2 - 0.05 * np.abs(df.Angle) + rng.normal(0, 0.2 + 0.3 * df.Rough)
    st = SimulationStudy()
    with _quiet():
        st.add_data(df, input_cols=["Length", "Angle", "Rough"], outcome_col="Signal")
    return st


# --- Polynomial equation in raw units ---

def test_polynomial_raw_coefficients_reproduce_predictions():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(0, 10, 60), rng.uniform(-5, 5, 60)])
    y = 1.5 + 0.8 * X[:, 0] - 0.05 * X[:, 0] ** 2 + 0.3 * X[:, 1] + rng.normal(0, 0.1, 60)
    models, _, _ = fit_all_robust_mean_models(X, y, max_degree=3, n_folds=5)
    model = models[("Polynomial", 3)]

    intercept, coefs, terms = polynomial_raw_coefficients(model, ["a", "b"])
    poly = model.named_steps["polynomialfeatures"]
    feats = poly.transform(X)
    names = list(poly.get_feature_names_out(["a", "b"]))
    manual = intercept + feats[:, [names.index(t) for t in terms]] @ coefs
    assert np.allclose(manual, model.predict(X))

    eq = generate_latex_equation(model, ["a", "b"], "Signal")
    assert eq.startswith("Signal = ")


# --- n_jobs semantics ---

def test_resolve_n_jobs(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert resolve_n_jobs(None) == 1
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(-1) == 6
    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(64) == 8


# --- Settings kept by update_slice; bandwidth_ratio actually used ---

def test_update_slice_keeps_nuisance_distribution(study_3d):
    dists = {"Rough": ("norm", (0.5, 0.1))}
    with _quiet():
        study_3d.pod(poi_col="Length", nuisance_col=["Rough"], slice_values={"Angle": 0.0},
                     threshold=2.0, n_boot=0, nuisance_dists=dists)
        res = study_3d.update_slice({"Angle": 5.0})
    assert res["nuisance_dists"] == dists
    assert res["spectrum_key"][-1] != frozenset()  # distribution is part of the cache key


def test_bandwidth_ratio_fixes_the_bandwidth(study_3d):
    with _quiet():
        auto = study_3d.pod(poi_col="Length", threshold=2.0, n_boot=0)["bandwidth"]
        fixed_a = study_3d.pod(poi_col="Length", threshold=2.0, n_boot=0, bandwidth_ratio=0.05)["bandwidth"]
        fixed_b = study_3d.pod(poi_col="Length", threshold=2.0, n_boot=0, bandwidth_ratio=0.3)["bandwidth"]
    assert fixed_b > fixed_a
    assert not np.isclose(auto, fixed_a) or not np.isclose(auto, fixed_b)


def test_polynomial_override_uses_one_se_rule(study_3d):
    with _quiet():
        study_3d.pod(poi_col="Length", threshold=2.0, n_boot=0)
        res = study_3d.pod(poi_col="Length", threshold=2.0, n_boot=0, model_override="polynomial")
    from digiqual.pod import select_cv_winner
    polys = {k: v for k, v in study_3d.cv_scores_cache.items() if k[0] == "Polynomial"}
    expected = select_cv_winner(polys, res["mean_model"].cv_se_)
    assert ("Polynomial", res["mean_model"].model_params_) == expected


def test_linear_results_store_n_boot(study_3d):
    with _quiet():
        study_3d.linear_pod(poi_col="Length", threshold=2.0, n_boot=20, n_jobs=1)
    assert study_3d.linear_pod_results["n_boot"] == 20


# --- Adaptive loop ---

def test_adaptive_search_survives_too_few_successful_runs():
    """With most of the initial batch failing (<10 successes), the loop tops up the
    design instead of raising."""
    from digiqual.adaptive import run_adaptive_search
    from digiqual.executors import PythonExecutor

    calls = {"n": 0}

    def solver(row):
        calls["n"] += 1
        if calls["n"] <= 8:  # the first 8 runs fail
            raise RuntimeError("mesh error")
        return 2 * row["x"]

    with _quiet():
        out = run_adaptive_search(
            executor=PythonExecutor(solver, "y"), input_cols=["x"], outcome_col="y",
            ranges={"x": (0.0, 10.0)}, n_start=10, n_step=10, max_iter=2, seed=0,
        )
    assert len(out) >= 10


def test_collinearity_failure_is_reported_not_converged():
    from digiqual.adaptive import generate_targeted_samples
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 40)
    df = pd.DataFrame({"a": a, "b": a * 2 + 1e-6 * rng.normal(size=40)})
    df["y"] = df.a + rng.normal(0, 0.01, 40)
    with _quiet():
        out = generate_targeted_samples(df, ["a", "b"], "y", max_gap_ratio=1.0, seed=0)
    assert any("Collinearity" in u for u in out.attrs.get("unresolved", []))


# --- Input guards shared by the C++ and NumPy paths ---

def test_kernel_smoother_rejects_bad_bandwidth():
    X = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        predict_local_std_fast(X, np.ones(10), X, 0.0)
    with pytest.raises(ValueError):
        predict_local_std_fast(X, np.ones(9), X, 0.1)


def test_pod_probs_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        compute_pod_probs_fast(np.zeros(3), np.ones(3), 0.0, ("norm", (0.0, 0.0)))


# --- Reliability point ---

def test_reliability_point_edges():
    x = np.linspace(0, 10, 101)
    curve = 1 / (1 + np.exp(-(x - 5)))
    a = calculate_reliability_point(x, curve, 0.9)
    assert np.isclose(a, 5 + np.log(9), atol=0.02)

    assert np.isnan(calculate_reliability_point(x, curve * 0.5, 0.9))   # never reached
    assert calculate_reliability_point(x, np.ones_like(x), 0.9) == x[0]  # reached immediately

    with_nan = curve.copy()
    with_nan[:20] = np.nan
    assert np.isclose(calculate_reliability_point(x, with_nan, 0.9), a, atol=0.02)

    plateau = np.minimum(curve, 0.95)
    plateau[60:70] = plateau[59]  # flat stretch before the crossing is reached
    assert np.isfinite(calculate_reliability_point(x, plateau, 0.9))


# --- Nuisance sampling stays inside the training range ---

def test_custom_nuisance_draws_are_truncated_to_range():
    class Recorder:
        def __init__(self):
            self.seen = []

        def predict(self, X):
            self.seen.append(np.array(X))
            return np.zeros(len(X))

    rng = np.random.default_rng(0)
    X_train = np.column_stack([rng.uniform(0, 1, 30), rng.uniform(0, 1, 30)])
    model = Recorder()
    compute_multi_dim_pod(
        poi_grid=np.linspace(0, 1, 5).reshape(-1, 1),
        nuisance_ranges={"z": (0.0, 1.0)},
        model=model, X_train=X_train, residuals=np.ones(30), bandwidth=0.3,
        dist_info=("norm", (0.0, 1.0)), thresholds=0.5, n_mc_samples=200,
        feature_names=["x", "z"], poi_names=["x"],
        nuisance_dists={"z": ("norm", (0.5, 2.0))},  # wide: much mass outside [0, 1]
    )
    z = np.concatenate([s[:, 1] for s in model.seen])
    assert z.min() >= 0.0 and z.max() <= 1.0


def test_multi_dim_pod_requires_all_columns():
    with pytest.raises(ValueError):
        compute_multi_dim_pod(
            poi_grid=np.linspace(0, 1, 5).reshape(-1, 1), nuisance_ranges={},
            model=None, X_train=np.zeros((10, 3)), residuals=np.ones(10), bandwidth=0.3,
            dist_info=("norm", (0.0, 1.0)), thresholds=0.5,
        )


# --- Method refinements ---

def test_variance_smoother_is_invariant_to_input_units():
    """Standardised smoother inputs: rescaling one input (m -> mm) changes nothing."""
    from sklearn.linear_model import LinearRegression

    from digiqual.pod import fit_variance_model, predict_local_std

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0, 1, 120), rng.uniform(-30, 30, 120)])
    y = 2 * X[:, 0] + rng.normal(0, 0.1 + 0.5 * X[:, 0])
    X_mm = X * np.array([1000.0, 1.0])

    with _quiet():
        res_a, bw_a = fit_variance_model(X, y, LinearRegression().fit(X, y))
        res_b, bw_b = fit_variance_model(X_mm, y, LinearRegression().fit(X_mm, y))
    assert np.isclose(bw_a, bw_b, rtol=1e-4)
    probe = X[:10]
    assert np.allclose(predict_local_std(X, res_a, probe, bw_a),
                       predict_local_std(X_mm, res_b, probe * np.array([1000.0, 1.0]), bw_b), rtol=1e-4)


def test_distribution_inference_rejects_degenerate_bounded_fits():
    """The chosen distribution's support must contain every z-score with a margin,
    so the PoD can't jump to exactly 0 or 1 just outside the observed residuals."""
    import scipy.stats as stats

    from digiqual.pod import infer_best_distribution, predict_local_std

    rng = np.random.default_rng(3)
    X = np.linspace(0, 10, 300)
    for residuals in (rng.normal(0, 1, 300), rng.uniform(-1, 1, 300), rng.exponential(1, 300)):
        name, params = infer_best_distribution(residuals, X, bandwidth=0.5)
        z = residuals / predict_local_std(X, residuals, X, 0.5)
        margin = 0.05 * np.ptp(z)
        lower, upper = getattr(stats, name).support(*params)
        assert lower < z.min() - margin and upper > z.max() + margin, name


def test_bootstrap_convergence_ignores_response_offset():
    """The convergence measure is relative to the data's scatter, not its mean."""
    from digiqual.diagnostics import _check_bootstrap_convergence

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(0, 1, 60)})
    df["y"] = 0.5 * df.x + rng.normal(0, 0.05, 60)          # near zero, like dB
    shifted = df.assign(y=df.y + 1000.0)                    # same data, big offset

    a = _check_bootstrap_convergence(df, ["x"], "y")
    b = _check_bootstrap_convergence(shifted, ["x"], "y")
    assert np.isclose(a["avg_relative_width"], b["avg_relative_width"], atol=1e-3)
    assert a["converged"]
