import warnings
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist

try:
    from . import _digiqual_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    _digiqual_cpp = None

def predict_local_std_fast(
    X: np.ndarray,
    residuals: np.ndarray,
    X_eval: np.ndarray,
    bandwidth: float,
    out: np.ndarray = None
) -> np.ndarray:
    """
    High-performance Nadaraya-Watson kernel smoothing for local standard deviation.

    Uses C++ multi-threading if compiled, or vectorized NumPy array operations if not.
    """
    X_source = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X, dtype=np.float64)
    X_target = np.atleast_2d(X_eval).T if np.asarray(X_eval).ndim == 1 else np.asarray(X_eval, dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64).flatten()

    if HAS_CPP:
        return _digiqual_cpp.predict_local_std(X_source, res, X_target, float(bandwidth), out)

    # Vectorized Python Fallback (batch cdist across all evaluation points)
    sq_residuals = res ** 2
    dists = cdist(X_target, X_source, metric='euclidean')
    weights = stats.norm.pdf(dists, loc=0, scale=bandwidth)

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    weights = weights / row_sums

    result = np.sqrt(weights @ sq_residuals)
    if out is not None:
        np.copyto(out, result)
        return out
    return result

def compute_pod_probs_fast(
    mean_resp: np.ndarray,
    sigma_resp: np.ndarray,
    threshold: float,
    dist_info: tuple,
    out: np.ndarray = None
) -> np.ndarray:
    """
    High-performance PoD survival CDF calculation.

    Uses C++ fast analytical CDF evaluation if compiled, or vectorized SciPy distribution call if not.
    """
    dist_name, dist_params = dist_info
    mean_arr = np.asarray(mean_resp, dtype=np.float64)
    sigma_arr = np.asarray(sigma_resp, dtype=np.float64)

    if HAS_CPP and dist_name in ("norm", "gumbel_r", "logistic", "laplace"):
        return _digiqual_cpp.compute_pod_probs(mean_arr, sigma_arr, float(threshold), dist_name, dist_params, out)

    sig = np.maximum(sigma_arr, 1e-10)
    z_threshold = (threshold - mean_arr) / sig
    dist_obj = getattr(stats, dist_name)
    result = 1.0 - dist_obj.cdf(z_threshold, *dist_params)
    if out is not None:
        np.copyto(out, result)
        return out
    return result
