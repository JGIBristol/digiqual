import numpy as np
import scipy.stats as stats
from typing import Any, Tuple, Dict, Union
from scipy.stats import qmc

def compute_multi_dim_pod(
    poi_grid: np.ndarray,
    nuisance_ranges: Dict[str, Tuple[float, float]],
    model: Any,
    X_train: np.ndarray,
    residuals: np.ndarray,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    thresholds: Union[float, np.ndarray, list],
    n_mc_samples: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Monte Carlo integration over continuous nuisance parameters to calculate
    the marginal Probability of Detection (PoD) for a grid of Parameters of Interest (PoI).

    Now supports vectorized 'thresholds', allowing for the simultaneous calculation
    of a PoD spectrum across a range of signal detection levels.

    Args:
        poi_grid (np.ndarray): The evaluation grid of the Parameters of Interest shape (N, n_pois).
        nuisance_ranges (Dict[str, Tuple[float, float]]): The min/max bounds for each nuisance.
        model (Any): Fitted multi-dimensional surrogate model (predicts mean response).
        X_train (np.ndarray): Original training data (N_train, n_total_vars).
        residuals (np.ndarray): Residuals from the model fit.
        bandwidth (float): Local kernel smoothing bandwidth.
        dist_info (Tuple[str, Tuple]): Residual error distribution (name, params).
        thresholds (Union[float, np.ndarray, list]): One or more signal detection thresholds.
        n_mc_samples (int): Number of Monte Carlo draws per PoI grid point.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pod_integrated: Integrated PoD values. If 'thresholds' was a vector,
              this is shape (n_grid_points, n_thresholds). Otherwise, shape (n_grid_points,).
            - mean_integrated: Expected mean response across the poi_grid.
    """
    n_pois = poi_grid.shape[1]
    n_nuisance = len(nuisance_ranges)
    total_vars = n_pois + n_nuisance

    dist_name, dist_params = dist_info
    dist_obj = getattr(stats, dist_name)

    # 1. Handle Threshold Vectorization
    is_vector = isinstance(thresholds, (np.ndarray, list))
    thresh_array = np.atleast_1d(thresholds)
    n_thresholds = len(thresh_array)

    # 2. Pre-generate LHS samples for the nuisance space [0, 1]
    if n_nuisance > 0:
        sampler = qmc.LatinHypercube(d=n_nuisance, seed=42)
        lhs_01 = sampler.random(n=n_mc_samples)

        # Scale to bounds
        nuisance_samples = np.zeros_like(lhs_01)
        for i, (min_val, max_val) in enumerate(nuisance_ranges.values()):
            nuisance_samples[:, i] = lhs_01[:, i] * (max_val - min_val) + min_val
    else:
        n_mc_samples = 1

    from digiqual.pod import predict_local_std

    # Prepare storage based on whether we are calculating a spectrum or a single curve
    if is_vector:
        pod_integrated = np.zeros((len(poi_grid), n_thresholds))
    else:
        pod_integrated = np.zeros(len(poi_grid))

    mean_integrated = np.zeros(len(poi_grid))

    # 3. Main Integration Loop
    for i, poi_point in enumerate(poi_grid):
        # Assemble the full input vectors for this grid point
        X_eval = np.zeros((n_mc_samples, total_vars))
        X_eval[:, :n_pois] = poi_point

        if n_nuisance > 0:
            X_eval[:, n_pois:] = nuisance_samples

        # A) Predict mean response and local noise (The heavy lifting)
        mean_resp = model.predict(X_eval)
        sigma_resp = predict_local_std(X_train, residuals, X_eval, bandwidth)

        # B) Calculate probability of exceedance
        if is_vector:
            # Broadcast thresholds: (N_thresh, 1) - (N_mc,) -> (N_thresh, N_mc)
            z_scores = (thresh_array[:, np.newaxis] - mean_resp) / sigma_resp
            probs = 1 - dist_obj.cdf(z_scores, *dist_params)
            # Take mean across MC samples for each threshold
            pod_integrated[i, :] = np.mean(probs, axis=1)
        else:
            z_scores = (thresholds - mean_resp) / sigma_resp
            probs = 1 - dist_obj.cdf(z_scores, *dist_params)
            pod_integrated[i] = np.mean(probs)

        mean_integrated[i] = np.mean(mean_resp)

    return pod_integrated, mean_integrated
