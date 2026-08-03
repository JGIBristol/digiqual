import pytest
import numpy as np
import scipy.stats as stats
from digiqual.cpp_fallback import predict_local_std_fast, compute_pod_probs_fast, HAS_CPP
from digiqual import _digiqual_cpp

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
    dist_info = ('norm', (0, 1))

    # Reference calculation
    sig = np.maximum(sigma_resp, 1e-10)
    z = (threshold - mean_resp) / sig
    expected = 1.0 - stats.norm.cdf(z, 0, 1)

    # C++ calculation
    cpp_actual = _digiqual_cpp.compute_pod_probs(mean_resp, sigma_resp, threshold, "norm", (0, 1))

    np.testing.assert_allclose(cpp_actual, expected, rtol=1e-5, atol=1e-7)
