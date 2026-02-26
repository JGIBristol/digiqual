import numpy as np
import scipy.stats as stats
from typing import Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#### Mean Model - Robust Regression (Polynomial + Kriging) ####

def fit_robust_mean_model(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 10
) -> Any:
    """
    Fits regression models (Polynomials and Kriging) and selects the optimal one.

    This function performs k-fold Cross Validation (CV) to find the model type
    (Polynomial or Kriging) and parameters (e.g., degree) that minimize the
    Mean Squared Error (MSE), balancing bias and variance.

    Args:
        X (np.ndarray): 1D array of input variable values (e.g., flaw size).
        y (np.ndarray): 1D array of outcome values (e.g., signal response).
        max_degree (int, optional): The maximum polynomial degree to test. Defaults to 10.
        n_folds (int, optional): Number of folds for Cross Validation. Defaults to 10.

    Returns:
        Any: A fitted scikit-learn model with the following added attributes:
        - `model_type_` (str): Either 'Polynomial' or 'Kriging'.
        - `model_params_` (Any): The selected integer degree or the fitted kernel.
        - `cv_scores_` (dict): The cross-validation MSE scores for all tested models.
    """
    X_2d = X.reshape(-1, 1)
    degrees = range(1, max_degree + 1)
    cv_scores = {}

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 1. Evaluate Polynomials
    for d in degrees:
        model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
        # Use neg_mean_squared_error; we take the negative to get positive MSE
        scores = cross_val_score(model, X_2d, y, cv=cv, scoring='neg_mean_squared_error')
        cv_scores[('Polynomial', d)] = -np.mean(scores)

    # 2. Evaluate Kriging (Gaussian Process)
    # INCREASED BOUNDS: Constant value up to 1e6 and RBF length scale up to 1e5
    # This addresses the ConvergenceWarning
    kernel = C(1.0, (1e-5, 1e6)) * RBF(1.0, (1e-3, 1e5))

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10, # Increased for better global search
        alpha=np.var(y) * 0.01,
        random_state=42
    )

    gpr_scores = cross_val_score(gpr, X_2d, y, cv=cv, scoring='neg_mean_squared_error')
    cv_scores[('Kriging', None)] = -np.mean(gpr_scores)

    # 3. Find best overall model
    best_model_info = min(cv_scores, key=cv_scores.get)
    best_type, best_params = best_model_info

    # 4. Train final model
    if best_type == 'Polynomial':
        final_model = make_pipeline(PolynomialFeatures(degree=best_params), LinearRegression())
        final_model.fit(X_2d, y)
        final_model.model_params_ = best_params
    else:
        # Re-initialize with the same wide-bound kernel
        final_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=15, # More restarts for the final fit
            alpha=np.var(y) * 0.01,
            random_state=42
        )
        final_model.fit(X_2d, y)
        final_model.model_params_ = final_model.kernel_

    final_model.model_type_ = best_type
    final_model.cv_scores_ = cv_scores

    return final_model

def plot_model_selection(cv_scores: dict) -> Any:
    """
    Generates a normalized bar chart of the Bias-Variance Tradeoff from CV scores,
    alongside a sorted table of the exact MSE values in best-fit order.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Prepare the data for the bar chart
    poly_keys = [k for k in cv_scores.keys() if k[0] == 'Polynomial']
    poly_degrees = sorted([k[1] for k in poly_keys])

    labels = []
    mses = []

    for d in poly_degrees:
        labels.append(f"Poly {d}")
        mses.append(cv_scores[('Polynomial', d)])

    if ('Kriging', None) in cv_scores:
        labels.append("Kriging")
        mses.append(cv_scores[('Kriging', None)])

    # Normalize by minimum error (matching the MATLAB paper implementation)
    min_mse = min(mses)
    normalized_mses = [m / min_mse for m in mses]

    # 2. Prepare the data for the sorted table
    # Sort dictionary by MSE ascending (lowest/best first)
    sorted_scores = sorted(cv_scores.items(), key=lambda item: item[1])
    table_data = []
    for (m_type, m_param), score in sorted_scores:
        name = f"Poly {m_param}" if m_type == 'Polynomial' else "Kriging"
        # Format to 1 decimal place in scientific notation = 2 significant figures
        val_str = f"{score:.1e}"
        table_data.append([name, val_str])

    # 3. Create the Figure and subplots (Bar chart on left, Table on right)
    fig, (ax_plot, ax_table) = plt.subplots(
        1, 2, figsize=(12, 5),dpi=300, gridspec_kw={'width_ratios': [2.5, 1]}
    )

    # --- Bar Chart (Left) ---
    bars = ax_plot.bar(labels, normalized_mses, color='#1f77b4', edgecolor='black', alpha=0.8)

    # Highlight the winning model in red
    best_idx = np.argmin(mses)
    bars[best_idx].set_color('crimson')
    bars[best_idx].set_edgecolor('black')

    # Add the reference line at 1.0
    ax_plot.axhline(1.0, color='red', linestyle='-.', linewidth=1.5, label='Min Error')

    ax_plot.set_title('Model Selection: Bias-Variance Tradeoff', fontweight='bold')
    ax_plot.set_ylabel('Error / Min Error [-]')
    ax_plot.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Rotate x-axis labels to prevent overlap
    ax_plot.tick_params(axis='x', rotation=45)
    ax_plot.legend()

    # --- Table (Right) ---
    ax_table.axis('off')
    ax_table.set_title('MSE Values\n(Best Fit Order)', fontweight='bold', pad=10)

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Model", "MSE"],
        loc='center',
        cellLoc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8) # Stretch vertically for better readability

    # Bold the headers and highlight the winning row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        if row == 1:
            cell.set_facecolor('#ffcccc') # Light red background for the winner

    fig.tight_layout()

    return fig


#### Variance Model - Kernel Smoothing ####

def fit_variance_model(
    X: np.ndarray,
    y: np.ndarray,
    mean_model: Any,
    bandwidth_ratio: float = 0.1,
    n_eval_points: int = 100
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Calculates residuals and prepares the grid for variance estimation.

    This acts as the setup phase for the heteroscedasticity model. It computes
    the raw residuals from the mean model and defines the smoothing bandwidth.

    Args:
        X (np.ndarray): The original input data.
        y (np.ndarray): The original outcome data.
        mean_model (Any): The fitted sklearn pipeline from `fit_robust_mean_model`.
        bandwidth_ratio (float, optional): The kernel smoothing window size as a
            fraction of the data range (X_max - X_min). Defaults to 0.1.
        n_eval_points (int, optional): Number of points in the evaluation grid.
            Defaults to 100.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]:
            - residuals: Raw differences between `y` and the mean model prediction.
            - bandwidth: The calculated smoothing window size (absolute units).
            - X_eval: A linearly spaced grid over the X domain for plotting/evaluation.
    """
    X_2d = X.reshape(-1, 1)
    y_pred = mean_model.predict(X_2d)
    residuals = y - y_pred

    bandwidth = (X.max() - X.min()) * bandwidth_ratio
    X_eval = np.linspace(X.min(), X.max(), n_eval_points)

    return residuals, bandwidth, X_eval


def predict_local_std(
    X: np.ndarray,
    residuals: np.ndarray,
    X_eval: np.ndarray,
    bandwidth: float
) -> np.ndarray:
    """
    Estimates the local standard deviation using Gaussian Kernel Smoothing.

    This implements a Nadaraya-Watson estimator specifically for the squared
    residuals to model how noise varies across the input domain (heteroscedasticity).

    Args:
        X (np.ndarray): The source locations (original data inputs).
        residuals (np.ndarray): The residuals observed at `X`.
        X_eval (np.ndarray): The target locations to estimate variance at.
        bandwidth (float): The width of the Gaussian kernel (sigma).

    Returns:
        np.ndarray: An array of standard deviation estimates corresponding to `X_eval`.
    """
    X_source = X.reshape(1, -1)
    X_target = X_eval.reshape(-1, 1)

    sq_residuals = residuals.flatten() ** 2

    diff = X_target - X_source
    weights = stats.norm.pdf(diff, loc=0, scale=bandwidth)

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    weights = weights / row_sums

    return np.sqrt(weights @ sq_residuals)


#### Residual Distribution Fitting ####

def infer_best_distribution(
    residuals: np.ndarray,
    X: np.ndarray,
    bandwidth: float
) -> Tuple[str, Tuple]:
    """
    Selects the best statistical distribution for the standardized residuals using AIC.

    This function normalizes residuals by their local standard deviation (Z-scores)
    and tests them against a suite of candidate distributions (Normal, Gumbel,
    Logistic, Laplace, t-Student).

    Args:
        residuals (np.ndarray): Raw residuals from the mean model.
        X (np.ndarray): Input locations for the residuals.
        bandwidth (float): Bandwidth used for local standardization.

    Returns:
        Tuple[str, Tuple]:
            - best_name: The SciPy name of the best-fitting distribution (e.g., 'norm').
            - best_params: The fitted parameters for that distribution (e.g., loc, scale).
    """
    #
    local_std = predict_local_std(X, residuals, X, bandwidth)
    z_scores = residuals.flatten() / local_std.flatten()

    candidates = [
        "norm",         # Gaussian (The Standard)
        "gumbel_r",     # Right-skewed Extreme Value
        "gumbel_l",     # Left-skewed Extreme Value (Common for cracks)
        "logistic",     # Heavier tails than Normal
        "laplace",      # Sharper peak, heavy tails
        "t",            # Student's t (Very robust to outliers)
    ]

    best_aic = np.inf
    best_result = ("norm", (0, 1))

    for dist_name in candidates:
        try:
            dist_obj = getattr(stats, dist_name)
            params = dist_obj.fit(z_scores)
            log_likelihood = np.sum(np.log(dist_obj.pdf(z_scores, *params)))

            k = len(params)
            aic = 2*k - 2*log_likelihood

            if aic < best_aic:
                best_aic = aic
                best_result = (dist_name, params)
        except Exception:
            continue

    return best_result


#### PoD Generation and Bootstrap Intervals ####

def compute_pod_curve(
    X_eval: np.ndarray,
    mean_model: Any,
    X: np.ndarray,
    residuals: np.ndarray,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Probability of Detection (PoD) curve.

    Combines the Mean Model, Variance Model, and Error Distribution to compute
    the probability that the signal exceeds the threshold at every point in `X_eval`.

    Args:
        X_eval (np.ndarray): The grid points to calculate PoD for.
        mean_model (Any): The fitted sklearn mean response model.
        X (np.ndarray): Original input data (needed for variance prediction).
        residuals (np.ndarray): Original residuals (needed for variance prediction).
        bandwidth (float): Smoothing bandwidth.
        dist_info (Tuple[str, Tuple]): The (name, params) of the error distribution.
        threshold (float): The detection threshold value.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pod_curve: Array of probabilities [0, 1] for each point in X_eval.
            - mean_curve: Array of mean signal response values for X_eval.

    Examples
    --------
    ```python
    # Assuming we have fitted models (mean_model) and data (X, residuals)
    # Calculate the PoD curve for a threshold of 0.5
    pod, mean_resp = compute_pod_curve(
        X_eval=np.linspace(0, 10, 100),
        mean_model=mean_model,
        X=X,
        residuals=residuals,
        bandwidth=1.5,
        dist_info=('norm', (0, 1)),
        threshold=0.5
    )
    ```
    """
    #
    dist_name, dist_params = dist_info

    mean_curve = mean_model.predict(X_eval.reshape(-1, 1))
    sigma_curve = predict_local_std(X, residuals, X_eval, bandwidth)

    z_threshold = (threshold - mean_curve) / sigma_curve

    dist_obj = getattr(stats, dist_name)
    pod_curve = 1 - dist_obj.cdf(z_threshold, *dist_params)

    return pod_curve, mean_curve


def bootstrap_pod_ci(
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    threshold: float,
    model_type: str,
    model_params: Any,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    n_boot: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates 95% Confidence Bounds for the PoD curve via Bootstrapping.

    This function resamples the original data with replacement `n_boot` times.
    For each resample, it refits the Mean Model (dynamically rebuilding either
    a Polynomial or Kriging model), recalculates residuals, and generates a new PoD curve.
    If Kriging is selected, the optimizer is disabled during bootstrapping to remain
    computationally tractable.

    Args:
        X (np.ndarray): Original input data.
        y (np.ndarray): Original outcome data.
        X_eval (np.ndarray): Grid points for evaluation.
        threshold (float): Detection threshold.
        model_type (str): The type of mean model ('Polynomial' or 'Kriging').
        model_params (Any): Model parameters (integer degree for Poly, kernel for Kriging).
        bandwidth (float): Smoothing bandwidth (fixed from original fit).
        dist_info (Tuple[str, Tuple]): Error distribution (fixed from original fit).
        n_boot (int, optional): Number of bootstrap iterations. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - lower_ci: The 2.5th percentile PoD curve (Lower 95% Bound).
            - upper_ci: The 97.5th percentile PoD curve (Upper 95% Bound).

    Examples
    --------
    ```python
    # Generate 95% confidence bounds
    lower, upper = bootstrap_pod_ci(
        X, y, X_eval, threshold=0.5,
        model_type='Polynomial', model_params=3,
        bandwidth=1.5, dist_info=('norm', (0, 1)), n_boot=100
    )
    ```
    """
    n_samples = len(y)
    pod_matrix = np.zeros((n_boot, len(X_eval)))

    for i in range(n_boot):
        # Resample indices
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_res, y_res = X[idx], y[idx]

        # Conditionally Fit Mean Model based on winning type
        if model_type == 'Polynomial':
            mean_model = make_pipeline(PolynomialFeatures(model_params), LinearRegression())
        elif model_type == 'Kriging':
            # Use the already optimized kernel (model_params) and turn off optimizer
            mean_model = GaussianProcessRegressor(
                kernel=model_params, alpha=np.var(y_res)*0.01, optimizer=None
            )

        mean_model.fit(X_res.reshape(-1, 1), y_res)

        # Residuals
        y_pred = mean_model.predict(X_res.reshape(-1, 1))
        res_res = y_res - y_pred

        # Predict
        pod_curve, _ = compute_pod_curve(
            X_eval, mean_model, X_res, res_res, bandwidth, dist_info, threshold
        )
        pod_matrix[i, :] = pod_curve

    return np.percentile(pod_matrix, 2.5, axis=0), np.percentile(pod_matrix, 97.5, axis=0)


def calculate_reliability_point(
    X_eval: np.ndarray,
    ci_lower: np.ndarray,
    target_pod: float = 0.90
) -> float:
    """
    Calculates the defect size (a90/95) where the Lower Confidence Bound
    crosses the target reliability threshold (usually 0.90).

    Returns:
        float: The interpolated x-value, or np.nan if not reached.
    """
    # Check if the curve actually reaches the target
    if np.max(ci_lower) < target_pod:
        return np.nan

    # Interpolate to find exact crossing point
    # We swap args because we are solving for X given Y=0.90
    return float(np.interp(target_pod, ci_lower, X_eval))
