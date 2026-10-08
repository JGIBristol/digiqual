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

    # Validated here, for both backends: the C++ kernel returns without writing its
    # output for these inputs, which would hand back uninitialised memory.
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"Kernel smoothing bandwidth must be a positive number, got {bandwidth!r}.")
    if X_source.shape[0] == 0 or X_source.shape[0] != res.shape[0]:
        raise ValueError(
            f"Need one residual per training point (got {X_source.shape[0]} points, {res.shape[0]} residuals)."
        )

    if HAS_CPP:
        if out is not None:
            try:
                return _digiqual_cpp.predict_local_std(X_source, res, X_target, float(bandwidth), out)
            except TypeError:
                pass
        return _digiqual_cpp.predict_local_std(X_source, res, X_target, float(bandwidth))

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

def input_scale(X: np.ndarray) -> np.ndarray:
    """Per-input standard deviation of the training inputs (constant inputs get 1)."""
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X, dtype=np.float64)
    std = X_2d.std(axis=0)
    return np.where(std > 0, std, 1.0)


def predict_local_std_scaled(
    X: np.ndarray,
    residuals: np.ndarray,
    X_eval: np.ndarray,
    bandwidth: float,
    out: np.ndarray = None
) -> np.ndarray:
    """
    Kernel-smoothed local standard deviation on *standardised* inputs.

    Each input is divided by its standard deviation in the training data ``X``
    before distances are computed, so ``bandwidth`` is measured in standard
    deviations and an input recorded in small units (e.g. metres) carries the same
    weight as one in large units (e.g. degrees). This mirrors the input scaling of
    the Kriging surrogate. The raw kernel is `predict_local_std_fast`.
    """
    scale = input_scale(X)
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X, dtype=np.float64)
    X_eval_2d = np.atleast_2d(X_eval).T if np.asarray(X_eval).ndim == 1 else np.asarray(X_eval, dtype=np.float64)
    return predict_local_std_fast(X_2d / scale, residuals, X_eval_2d / scale, bandwidth, out=out)


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

    # scipy.stats parameter tuples end with (loc, scale). A non-positive scale is
    # invalid; the C++ path would also return uninitialised memory for it.
    if len(dist_params) >= 2 and not dist_params[-1] > 0:
        raise ValueError(f"Distribution scale must be positive, got {dist_params[-1]!r} for '{dist_name}'.")

    if HAS_CPP and dist_name in ("norm", "gumbel_r", "logistic", "laplace"):
        if out is not None:
            try:
                return _digiqual_cpp.compute_pod_probs(mean_arr, sigma_arr, float(threshold), dist_name, dist_params, out)
            except TypeError:
                pass
        return _digiqual_cpp.compute_pod_probs(mean_arr, sigma_arr, float(threshold), dist_name, dist_params)

    sig = np.maximum(sigma_arr, 1e-10)
    z_threshold = (threshold - mean_arr) / sig
    dist_obj = getattr(stats, dist_name)
    result = 1.0 - dist_obj.cdf(z_threshold, *dist_params)
    if out is not None:
        np.copyto(out, result)
        return out
    return result
