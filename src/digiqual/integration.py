import numpy as np
import scipy.stats as stats
from typing import Any, Tuple, Dict
from scipy.stats import qmc

def compute_multi_dim_pod(
    poi_grid: np.ndarray,
    nuisance_ranges: Dict[str, Tuple[float, float]],
    model: Any,
    X_train: np.ndarray,
    residuals: np.ndarray,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    threshold: float,
    n_mc_samples: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Monte Carlo integration over continuous nuisance parameters to calculate
    the marginal Probability of Detection (PoD) for a grid of Parameters of Interest (PoI).

    Based on the methodology from Malkiel et al. (2026), this function evaluates the
    flexible multi-dimensional surrogate model across randomly sampled nuisance parameter
    realizations, calculates the probability of detection at each realization using local
    variance standardisation, and aggregates them.

    Args:
        poi_grid (np.ndarray): The evaluation grid of the Parameters of Interest shape (N, n_pois).
        nuisance_ranges (Dict[str, Tuple[float, float]]): The min/max bounds for each nuisance.
        model (Any): Fitted multi-dimensional surrogate model (predicts mean response).
        X_train (np.ndarray): Original training data (N_train, n_total_vars).
        residuals (np.ndarray): Residuals from the model fit.
        bandwidth (float): Local kernel smoothing bandwidth.
        dist_info (Tuple[str, Tuple]): Residual error distribution (name, params).
        threshold (float): Signal detection threshold.
        n_mc_samples (int): Number of Monte Carlo draws per PoI grid point.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pod_integrated: Integrated PoD values across the poi_grid.
            - mean_integrated: Expected mean response across the poi_grid.
    """
    n_pois = poi_grid.shape[1]
    n_nuisance = len(nuisance_ranges)
    total_vars = n_pois + n_nuisance

    dist_name, dist_params = dist_info
    dist_obj = getattr(stats, dist_name)

    # Pre-generate LHS samples for the nuisance space [0, 1]
    # We use a single LHS draw scaled up for every PoI grid point to reduce variance between points
    if n_nuisance > 0:
        sampler = qmc.LatinHypercube(d=n_nuisance, seed=42)
        lhs_01 = sampler.random(n=n_mc_samples)

        # Scale to bounds
        nuisance_samples = np.zeros_like(lhs_01)
        for i, (min_val, max_val) in enumerate(nuisance_ranges.values()):
            nuisance_samples[:, i] = lhs_01[:, i] * (max_val - min_val) + min_val
    else:
        # If no nuisance variables are provided, gracefully degenerate to standard 1D PoD
        n_mc_samples = 1

    from digiqual.pod import predict_local_std

    pod_integrated = np.zeros(len(poi_grid))
    mean_integrated = np.zeros(len(poi_grid))

    for i, poi_point in enumerate(poi_grid):
        # Assemble the full input vectors for this grid point
        X_eval = np.zeros((n_mc_samples, total_vars))
        X_eval[:, :n_pois] = poi_point

        if n_nuisance > 0:
            X_eval[:, n_pois:] = nuisance_samples

        # 1. Predict the mean response from the N-Dimensional model
        mean_resp = model.predict(X_eval)

        # 2. Estimate local noise variance using the kernel smoothing
        # Note: We must predict local std dynamically at these N-dimensional points
        sigma_resp = predict_local_std(X_train, residuals, X_eval, bandwidth)

        # 3. Calculate probability of exceedance
        z_scores = (threshold - mean_resp) / sigma_resp
        probs = 1 - dist_obj.cdf(z_scores, *dist_params)

        # 4. Integrate / Aggregate (Monte Carlo expectation)
        pod_integrated[i] = np.mean(probs)
        mean_integrated[i] = np.mean(mean_resp)

    return pod_integrated, mean_integrated
