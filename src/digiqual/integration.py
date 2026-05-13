import numpy as np
import scipy.stats as stats
import warnings
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
    n_mc_samples: int = 3000,
    feature_names: list = None,
    poi_names: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the marginal Probability of Detection (PoD) across a grid of Parameters of Interest (PoI).

    This function determines the probability that a signal will exceed a given threshold.
    It features a dual-path architecture for maximum efficiency:

    1. **Fast Path (Vectorized)**: If there are no active nuisance parameters (i.e., all extra
       variables are held constant as 'slices'), it calculates the probabilities for the
       entire grid and all threshold vectors simultaneously in a single array operation.
    2. **Slow Path (Monte Carlo)**: If there are active nuisance parameters (ranges where
       min != max), it performs Monte Carlo integration to marginalize out the nuisance
       variance, running thousands of samples per grid point.

    Args:
        poi_grid (np.ndarray): A 2D array of shape (N_grid_points, n_pois) containing the
            evaluation coordinates.
        nuisance_ranges (Dict[str, Tuple[float, float]]): The min and max bounds for each
            nuisance parameter. If min == max, the parameter is treated as a constant slice.
        model (Any): A fitted scikit-learn surrogate model (predicts the mean response).
        X_train (np.ndarray): Original training data matrix (N_train, n_total_vars).
        residuals (np.ndarray): Residuals from the model fit, used for local noise estimation.
        bandwidth (float): The local kernel smoothing bandwidth for the variance model.
        dist_info (Tuple[str, Tuple]): The (name, parameters) of the residual error distribution.
        thresholds (Union[float, np.ndarray, list]): One or more signal detection thresholds.
            Providing an array triggers vectorized multi-threshold calculation.
        n_mc_samples (int, optional): Number of Monte Carlo draws per PoI grid point when
            evaluating active nuisances. Defaults to 3000.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pod_integrated (np.ndarray): The PoD values.
              Shape is `(N_grid_points,)` if a single threshold is provided, or
              `(N_grid_points, N_thresholds)` if an array of thresholds is provided.
            - mean_integrated (np.ndarray): The expected mean signal response across the PoI grid.

    Examples:
        ```python
        # Calculate a PoD spectrum for 100 thresholds instantly (Fast Path)
        pod_matrix, mean_curve = compute_multi_dim_pod(
            poi_grid=X_eval,
            nuisance_ranges={'Angle': (45.0, 45.0)}, # Constant slice
            model=kriging_model,
            X_train=X, residuals=resids, bandwidth=1.5,
            dist_info=('norm', (0, 1)),
            thresholds=np.linspace(10, 50, 100)
        )
        ```
    """

    n_pois = poi_grid.shape[1]
    n_nuisance = len(nuisance_ranges)
    total_vars = n_pois + n_nuisance

    dist_name, dist_params = dist_info
    dist_obj = getattr(stats, dist_name)

    # --- Explicit Column Index Mapping ---
    if feature_names and poi_names:
        poi_indices = [feature_names.index(p) for p in poi_names]
        nuisance_names = list(nuisance_ranges.keys())
        nuisance_indices = [feature_names.index(n) for n in nuisance_names]
    else:
        # Fallback if names aren't provided
        poi_indices = list(range(n_pois))
        nuisance_indices = list(range(n_pois, total_vars))

    # 1. Handle Threshold Vectorization
    is_vector = isinstance(thresholds, (np.ndarray, list))
    thresh_array = np.atleast_1d(thresholds)
    n_thresholds = len(thresh_array)

    # 2. Check for active integration requirements
    active_nuisances = sum(1 for min_val, max_val in nuisance_ranges.values() if min_val != max_val)

    if active_nuisances > 0:
        sampler = qmc.LatinHypercube(d=n_nuisance, seed=42)
        lhs_01 = sampler.random(n=n_mc_samples)
    else:
        n_mc_samples = 1
        lhs_01 = np.zeros((1, n_nuisance))

    # Scale the LHS samples to the physical bounds
    if n_nuisance > 0:
        nuisance_samples = np.zeros_like(lhs_01)
        for i, (min_val, max_val) in enumerate(nuisance_ranges.values()):
            nuisance_samples[:, i] = lhs_01[:, i] * (max_val - min_val) + min_val
    else:
        nuisance_samples = np.empty((n_mc_samples, 0))

    from .pod import predict_local_std

    # ---------------------------------------------------------
    # FAST PATH: Fully Vectorized (No active nuisances)
    # ---------------------------------------------------------
    if active_nuisances == 0:
        X_eval_full = np.zeros((len(poi_grid), total_vars))

        # Map PoIs to their correct physical columns
        for i, idx in enumerate(poi_indices):
            X_eval_full[:, idx] = poi_grid[:, i]

        # Map Slices/Nuisances to their correct physical columns
        if n_nuisance > 0:
            for i, idx in enumerate(nuisance_indices):
                X_eval_full[:, idx] = nuisance_samples[0, i]

        mean_resp = model.predict(X_eval_full).flatten()
        sigma_resp = predict_local_std(X_train, residuals, X_eval_full, bandwidth).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if is_vector:
                z_scores = (thresh_array[np.newaxis, :] - mean_resp[:, np.newaxis]) / sigma_resp[:, np.newaxis]
                pod_integrated = 1 - dist_obj.cdf(z_scores, *dist_params)
            else:
                z_scores = (thresholds - mean_resp) / sigma_resp
                pod_integrated = 1 - dist_obj.cdf(z_scores, *dist_params)

        return pod_integrated, mean_resp

    # ---------------------------------------------------------
    # SLOW PATH: Monte Carlo Integration
    # ---------------------------------------------------------
    if is_vector:
        pod_integrated = np.zeros((len(poi_grid), n_thresholds))
    else:
        pod_integrated = np.zeros(len(poi_grid))

    mean_integrated = np.zeros(len(poi_grid))

    for i, poi_point in enumerate(poi_grid):
        X_eval = np.zeros((n_mc_samples, total_vars))

        # Map PoIs
        for j, idx in enumerate(poi_indices):
            X_eval[:, idx] = poi_point[j]

        # Map Nuisances
        if n_nuisance > 0:
            for j, idx in enumerate(nuisance_indices):
                X_eval[:, idx] = nuisance_samples[:, j]

        mean_resp = model.predict(X_eval)
        sigma_resp = predict_local_std(X_train, residuals, X_eval, bandwidth)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if is_vector:
                z_scores = (thresh_array[:, np.newaxis] - mean_resp) / sigma_resp
                probs = 1 - dist_obj.cdf(z_scores, *dist_params)
                pod_integrated[i, :] = np.mean(probs, axis=1)
            else:
                z_scores = (thresholds - mean_resp) / sigma_resp
                probs = 1 - dist_obj.cdf(z_scores, *dist_params)
                pod_integrated[i] = np.mean(probs)

        mean_integrated[i] = np.mean(mean_resp)

    return pod_integrated, mean_integrated
