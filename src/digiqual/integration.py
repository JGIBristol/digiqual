import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple, Dict, List, Any, Optional
from . import pod

def build_integration_space(
    nuisance_cols: List[str],
    reference_data: pd.DataFrame,
    nuisance_dists: Optional[Dict[str, Any]] = None,
    n_mc_samples: int = 5000
) -> np.ndarray:
    """
    Builds the Monte Carlo integration matrix based on user-defined real-world distributions.

    This function is the first step in multi-dimensional reliability assessment.
    It takes the engineering distributions representing the physical reality of
    nuisance parameters (e.g., crack angle, surface roughness) and generates a
    massive randomized matrix. This matrix acts as the "virtual real world" that
    the Kriging model will be evaluated against during the integration phase.

    If a specific distribution is not provided for a parameter, the function contains
    a safety net: it automatically extracts the minimum and maximum values from the
    `reference_data` and defaults to a mathematically safe Uniform distribution across that range.

    Args:
        nuisance_cols (List[str]): A list of the string column names representing
            the nuisance parameters in the dataset (e.g., ['Angle', 'Roughness']).
        reference_data (pd.DataFrame): The validated simulation data used to calculate
            fallback bounds if a distribution is missing.
        nuisance_dists (Optional[Dict[str, Any]]): A dictionary mapping the column names to
            initialized `scipy.stats` continuous distribution objects. Defaults to None.
        n_mc_samples (int, optional): The number of virtual defects to simulate
            for the Monte Carlo integration. Higher numbers yield smoother, more
            accurate curves but increase computation time. Defaults to 5000.

    Returns:
        np.ndarray: A 2D matrix of shape (n_mc_samples, len(nuisance_cols)) containing
            the randomly sampled values for all nuisance parameters.

    Examples:
        ```python
        import scipy.stats as stats
        import pandas as pd

        # Define physical reality: Angle is normal, Roughness is unknown
        dists = {
            'Angle': stats.norm(loc=0, scale=5)
            # We purposely leave out 'Roughness' to test the safety net
        }

        # Build a space of 10,000 virtual defects
        mc_matrix = build_integration_space(
            nuisance_cols=['Angle', 'Roughness'],
            reference_data=study.clean_data,
            nuisance_dists=dists,
            n_mc_samples=10000
        )
        ```
    """
    mc_matrix = np.zeros((n_mc_samples, len(nuisance_cols)))
    nuisance_dists = nuisance_dists or {}

    for idx, col_name in enumerate(nuisance_cols):
        # 1. Check if the user provided a specific distribution
        if col_name in nuisance_dists:
            dist_object = nuisance_dists[col_name]

        # 2. The Safety Net: Automatically build a Uniform distribution
        else:
            c_min = reference_data[col_name].min()
            c_max = reference_data[col_name].max()
            c_range = c_max - c_min

            print(f"Warning: No distribution provided for '{col_name}'. Defaulting to Uniform bounds.")
            dist_object = stats.uniform(loc=c_min, scale=c_range)

        # 3. Generate random samples using the chosen distribution
        mc_matrix[:, idx] = dist_object.rvs(size=n_mc_samples)

    return mc_matrix


def compute_marginal_pod(
    X_eval_grid: np.ndarray,
    mean_model: Any,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    threshold: float,
    mc_samples: np.ndarray,
    X_orig: np.ndarray,
    residuals: np.ndarray
) -> np.ndarray:
    """
    Computes the 1D Marginal PoD by integrating out the nuisance parameters.

    This function utilizes Monte Carlo integration to distill a complex N-dimensional
    Probability of Detection surface down to a single, highly calibrated 1D curve.
    For every specific value of the Parameter of Interest (e.g., a 2mm crack), it
    evaluates the N-dimensional mean model against thousands of random virtual
    nuisance combinations (drawn from `mc_samples`). It computes the conditional
    PoD for each variation and averages them, effectively "integrating out" the
    real-world noise.

    Args:
        X_eval_grid (np.ndarray): The 1D grid of the main Parameter of Interest
            (e.g., a linearly spaced array of crack sizes to evaluate).
        mean_model (Any): The fitted scikit-learn N-dimensional regression model
            (usually Gaussian Process/Kriging) trained on the full feature space.
        bandwidth (float): The smoothing bandwidth used for variance estimation.
        dist_info (Tuple[str, Tuple]): A tuple containing the SciPy distribution
            name (str) and its fitted parameters (Tuple) for the residuals.
        threshold (float): The signal threshold required for a positive detection.
        mc_samples (np.ndarray): The 2D matrix of randomized nuisance parameter
            samples generated by `build_integration_space`.

    Returns:
        np.ndarray: A 1D array of marginal probabilities [0, 1] exactly matching
            the length of `X_eval_grid`.

    Examples:
        ```python
        # Assuming mean_model is trained on [Size, Angle] and mc_samples is built for Angle
        X_eval_poi = np.linspace(0.1, 10.0, 100)

        marginal_curve = compute_marginal_pod(
            X_eval_grid=X_eval_poi,
            mean_model=my_kriging_model,
            bandwidth=1.5,
            dist_info=('norm', (0, 1)),
            threshold=4.0,
            mc_samples=mc_matrix
        )
        ```
    """
    dist_name, dist_params = dist_info
    dist_obj = getattr(stats, dist_name)
    n_mc = len(mc_samples)

    marginal_pod = np.zeros(len(X_eval_grid))

    for i, poi_val in enumerate(X_eval_grid):
        # 1. Create matrix for this specific PoI value across all MC nuisance variations
        # E.g., [[Size_1, Angle_mc1], [Size_1, Angle_mc2], ...]
        poi_col = np.full((n_mc, 1), poi_val)
        X_mc_eval = np.hstack((poi_col, mc_samples))

        # 2. Predict signals for all Monte Carlo variations
        mean_preds = mean_model.predict(X_mc_eval)

        # 3. Calculate conditional PoDs
        # Calculate true local standard deviation across the N-Dimensional MC sample space
        sigma_preds = pod.predict_local_std(X_orig, residuals, X_mc_eval, bandwidth)
        z_thresholds = (threshold - mean_preds) / sigma_preds
        conditional_pods = 1 - dist_obj.cdf(z_thresholds, *dist_params)

        # 4. Average the probabilities to mathematically marginalize the nuisance parameters
        marginal_pod[i] = np.mean(conditional_pods)

    return marginal_pod
