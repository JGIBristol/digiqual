import numpy as np
import pytest
import scipy.stats as stats

from digiqual.cpp_fallback import HAS_CPP

# These tests compare the compiled extension against the NumPy fallback, so they
# only make sense when the extension is built (wheels, `setup.py build_ext`).
_digiqual_cpp = pytest.importorskip("digiqual._digiqual_cpp", reason="C++ extension not built")

def test_has_cpp_extension():
    assert HAS_CPP is True, "C++ extension _digiqual_cpp should be compiled and imported successfully."

def test_cpp_predict_local_std_equivalence():
    np.random.seed(42)
    N_train = 150
    D = 3
    N_eval = 80
    bandwidth = 1.25

    X_train = np.random.uniform(0, 10, size=(N_train, D))
    residuals = np.random.normal(0, 0.5, size=N_train)
    X_eval = np.random.uniform(0, 10, size=(N_eval, D))

    # Reference calculation (vectorized SciPy / NumPy)
    sq_residuals = residuals ** 2
    from scipy.spatial.distance import cdist
    dists = cdist(X_eval, X_train, metric='euclidean')
    weights = stats.norm.pdf(dists, loc=0, scale=bandwidth)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    weights = weights / row_sums
    expected = np.sqrt(weights @ sq_residuals)

    # C++ calculation
    cpp_actual = _digiqual_cpp.predict_local_std(X_train, residuals, X_eval, bandwidth)

    # Check numerical equivalence
    np.testing.assert_allclose(cpp_actual, expected, rtol=1e-5, atol=1e-7)

def test_cpp_compute_pod_probs_equivalence():
    np.random.seed(42)
    N = 100
    mean_resp = np.random.uniform(10, 30, size=N)
    sigma_resp = np.random.uniform(0.5, 2.0, size=N)
    threshold = 20.0

    for dist_name in ["norm", "gumbel_r", "gumbel_l", "logistic", "laplace"]:
        sig = np.maximum(sigma_resp, 1e-10)
        z = (threshold - mean_resp) / sig
        dist_obj = getattr(stats, dist_name)
        expected = 1.0 - dist_obj.cdf(z, loc=0, scale=1)

        # C++ calculation
        cpp_actual = _digiqual_cpp.compute_pod_probs(mean_resp, sigma_resp, threshold, dist_name, (0, 1))

        np.testing.assert_allclose(cpp_actual, expected, rtol=1e-5, atol=1e-7, err_msg=f"C++ calculation failed for {dist_name}")



def test_cpp_matches_fallback_far_from_data():
    """Tiny (but non-zero) kernel weight sums must not be clamped in C++ only."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    residuals = np.array([0.5, 1.0, 2.0])
    X_eval = np.array([[10.0], [11.0]])  # 8+ bandwidths away: weight sums below 1e-12
    bandwidth = 1.0

    from scipy.spatial.distance import cdist
    w = stats.norm.pdf(cdist(X_eval, X_train), scale=bandwidth)
    expected = np.sqrt((w / w.sum(axis=1, keepdims=True)) @ residuals ** 2)

    cpp_actual = _digiqual_cpp.predict_local_std(X_train, residuals, X_eval, bandwidth)
    np.testing.assert_allclose(cpp_actual, expected, rtol=1e-6)


def test_cpp_rejects_mismatched_shapes():
    X_train = np.zeros((5, 2))
    with pytest.raises(ValueError):
        _digiqual_cpp.predict_local_std(X_train, np.zeros(5), np.zeros((3, 3)), 1.0)  # wrong columns
    with pytest.raises(ValueError):
        _digiqual_cpp.predict_local_std(X_train, np.zeros(4), np.zeros((3, 2)), 1.0)  # residual count
    with pytest.raises(ValueError):
        _digiqual_cpp.predict_local_std(X_train, np.zeros(5), np.zeros((3, 2)), 1.0, np.zeros(2))  # out too small
    with pytest.raises(ValueError):
        _digiqual_cpp.compute_pod_probs(np.zeros(4), np.ones(3), 0.0, "norm", (0.0, 1.0))


def test_out_argument_is_filled_in_place():
    """A non-contiguous or non-float64 `out` is still filled (by copy), as in the fallback."""
    from digiqual.cpp_fallback import compute_pod_probs_fast, predict_local_std_fast

    X_train = np.linspace(0, 1, 10).reshape(-1, 1)
    residuals = np.linspace(0.1, 1, 10)
    X_eval = np.linspace(0, 1, 4).reshape(-1, 1)
    out = np.zeros((4, 2))[:, 0]  # strided view
    result = predict_local_std_fast(X_train, residuals, X_eval, 0.3, out=out)
    assert result is out
    np.testing.assert_allclose(out, predict_local_std_fast(X_train, residuals, X_eval, 0.3))

    out32 = np.zeros(4, dtype=np.float32)
    compute_pod_probs_fast(np.zeros(4), np.ones(4), 0.0, ("norm", (0.0, 1.0)), out=out32)
    np.testing.assert_allclose(out32, 0.5)


def test_set_num_threads_does_not_change_results():
    rng = np.random.default_rng(0)
    X_train, residuals, X_eval = rng.uniform(size=(200, 2)), rng.normal(size=200), rng.uniform(size=(500, 2))
    full = _digiqual_cpp.predict_local_std(X_train, residuals, X_eval, 0.2)
    _digiqual_cpp.set_num_threads(1)
    try:
        single = _digiqual_cpp.predict_local_std(X_train, residuals, X_eval, 0.2)
    finally:
        _digiqual_cpp.set_num_threads(0)
    np.testing.assert_array_equal(full, single)
