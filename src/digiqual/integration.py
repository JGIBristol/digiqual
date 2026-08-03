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
    poi_names: list = None,
    nuisance_dists: Dict[str, Tuple[str, Tuple]] = None
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
        feature_names (list, optional): Names of all feature columns in ``X_train``, in the exact
            same order as the columns appear. Used to correctly map PoIs and nuisances to
            their physical array indices.
        poi_names (list, optional): Names of the parameters of interest (PoIs). Each entry must
            correspond to a name in ``feature_names``.
        nuisance_dists (Dict[str, Tuple[str, Tuple]], optional): The custom distribution name and
            parameters (e.g. ('norm', (mean, std))) for each active nuisance variable.

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
    nuisance_names = list(nuisance_ranges.keys()) if nuisance_ranges else []
    if feature_names and poi_names:
        poi_indices = [feature_names.index(p) for p in poi_names]
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

    # Scale the LHS samples to the physical bounds or specified distributions
    if n_nuisance > 0:
        nuisance_samples = np.zeros_like(lhs_01)
        for i, name in enumerate(nuisance_names):
            min_val, max_val = nuisance_ranges[name]
            if min_val == max_val:
                # Constant slice
                nuisance_samples[:, i] = min_val
            elif nuisance_dists and name in nuisance_dists:
                custom_dist_name, custom_dist_params = nuisance_dists[name]
                custom_dist_obj = getattr(stats, custom_dist_name)
                if not isinstance(custom_dist_params, (tuple, list)):
                    custom_dist_params = (custom_dist_params,)
                # Inverse Transform Sampling
                nuisance_samples[:, i] = custom_dist_obj.ppf(lhs_01[:, i], *custom_dist_params)
            else:
                # Default: Uniform distribution over [min_val, max_val]
                nuisance_samples[:, i] = lhs_01[:, i] * (max_val - min_val) + min_val
    else:
        nuisance_samples = np.empty((n_mc_samples, 0))

    from .cpp_fallback import predict_local_std_fast, compute_pod_probs_fast

    # ---------------------------------------------------------
    # FAST PATH: Fully Vectorized (No active nuisances)
    # ---------------------------------------------------------
    if active_nuisances == 0:
        X_eval_full = np.zeros((len(poi_grid), total_vars))

        for i, idx in enumerate(poi_indices):
            X_eval_full[:, idx] = poi_grid[:, i]

        if n_nuisance > 0:
            for i, idx in enumerate(nuisance_indices):
                X_eval_full[:, idx] = nuisance_samples[0, i]

        mean_resp = model.predict(X_eval_full).flatten()
        sigma_resp = predict_local_std_fast(X_train, residuals, X_eval_full, bandwidth).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if is_vector:
                pod_list = []
                for t in thresh_array:
                    pod_list.append(compute_pod_probs_fast(mean_resp, sigma_resp, t, dist_info))
                pod_integrated = np.column_stack(pod_list)
            else:
                pod_integrated = compute_pod_probs_fast(mean_resp, sigma_resp, thresholds, dist_info)

        return pod_integrated, mean_resp

    # ---------------------------------------------------------
    # VECTORIZED PATH: Monte Carlo Integration Across All Grid Points
    # ---------------------------------------------------------
    N_grid = len(poi_grid)
    N_total_evals = N_grid * n_mc_samples

    X_eval_all = np.zeros((N_total_evals, total_vars))

    # Expand PoI grid points: repeat each grid point n_mc_samples times
    poi_expanded = np.repeat(poi_grid, n_mc_samples, axis=0)
    for j, idx in enumerate(poi_indices):
        X_eval_all[:, idx] = poi_expanded[:, j]

    # Map Nuisance samples: tile nuisance samples for every grid point
    if n_nuisance > 0:
        nuisance_tiled = np.tile(nuisance_samples, (N_grid, 1))
        for j, idx in enumerate(nuisance_indices):
            X_eval_all[:, idx] = nuisance_tiled[:, j]

    # Evaluate surrogate model predictions and local noise in a single batch call
    mean_resp_all = model.predict(X_eval_all).flatten()
    sigma_resp_all = predict_local_std_fast(X_train, residuals, X_eval_all, bandwidth).flatten()

    mean_integrated = mean_resp_all.reshape(N_grid, n_mc_samples).mean(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if is_vector:
            pod_integrated = np.zeros((N_grid, n_thresholds))
            for t_idx, t_val in enumerate(thresh_array):
                probs_all = compute_pod_probs_fast(mean_resp_all, sigma_resp_all, t_val, dist_info)
                pod_integrated[:, t_idx] = probs_all.reshape(N_grid, n_mc_samples).mean(axis=1)
        else:
            probs_all = compute_pod_probs_fast(mean_resp_all, sigma_resp_all, thresholds, dist_info)
            pod_integrated = probs_all.reshape(N_grid, n_mc_samples).mean(axis=1)

    return pod_integrated, mean_integrated
