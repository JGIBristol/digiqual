import numpy as np
import scipy.stats as stats
from typing import Tuple, Any, Dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize_scalar

import os
from joblib import Parallel, delayed

import warnings
from sklearn.exceptions import ConvergenceWarning

#### Mean Model - Robust Regression (Polynomial + Kriging) ####

def fit_all_robust_mean_models(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 10
) -> Tuple[Dict[Tuple[str, Any], Any], Dict[Tuple[str, Any], float], Tuple[str, Any]]:
    """
    Fits all polynomial models (and optionally Kriging) and returns them for caching.

    Instead of fitting models, selecting the best, and throwing the rest away,
    this function evaluates all candidates via k-fold Cross Validation (CV) and
    then fits *every* model to the full dataset. This allows the application to
    instantly swap between different model structures without recalculating.

    Args:
        X (np.ndarray): 1D array or 2D matrix of input variable values.
        y (np.ndarray): 1D array of outcome values (e.g., signal response).
        max_degree (int, optional): The maximum polynomial degree to test. Defaults to 10.
        n_folds (int, optional): Number of folds for Cross Validation. Defaults to 10.

    Returns:
        Tuple[Dict, Dict, Tuple]:
            - `fitted_models`: A dictionary mapping a key like `('Polynomial', 3)` to the fully trained scikit-learn model.
            - `cv_scores`: A dictionary mapping the same keys to their Cross-Validation MSE scores.
            - `cv_winner_key`: The key of the model that achieved the lowest MSE.

    Examples:
        ```python
        import numpy as np
        X = np.linspace(0, 10, 50)
        y = 3 * X + np.random.normal(0, 1, 50)

        models, scores, best_key = fit_all_robust_mean_models(X, y)

        print(f"The best model was: {best_key}")

        # Instantly retrieve the degree-4 polynomial without refitting
        poly_4 = models[('Polynomial', 4)]
        ```
    """
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)

    fitted_models = {}
    cv_scores = {}

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 1. Evaluate & Fit ALL Polynomials
    for d in range(1, max_degree + 1):
        model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())

        # A) Run CV for the bias-variance tradeoff evaluation
        scores = cross_val_score(model, X_2d, y, cv=cv, scoring='neg_mean_squared_error')
        cv_scores[('Polynomial', d)] = -np.mean(scores)

        # B) Fit to the FULL dataset and store it
        model.fit(X_2d, y)
        model.model_type_ = 'Polynomial'
        model.model_params_ = d
        fitted_models[('Polynomial', d)] = model

    # 2. Evaluate & Fit Kriging (With safeguard for large datasets)
    n_samples = len(y)
    if n_samples <= 1000:
        kernel = C(1.0, (1e-5, 1e6)) * RBF(1.0, (1e-3, 1e5))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=np.var(y) * 0.01,
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            # A) Run CV
            gpr_scores = cross_val_score(gpr, X_2d, y, cv=cv, scoring='neg_mean_squared_error')
            cv_scores[('Kriging', None)] = -np.mean(gpr_scores)

            # B) Fit to FULL dataset using more restarts for the final fit
            gpr.n_restarts_optimizer = 15
            gpr.fit(X_2d, y)

            gpr.model_type_ = 'Kriging'
            gpr.model_params_ = gpr.kernel_
            fitted_models[('Kriging', None)] = gpr
    else:
        print(f"Skipping Kriging evaluation to prevent timeout (Dataset N={n_samples} > 1000).")

    # 3. Determine the overall CV winner
    cv_winner_key = min(cv_scores, key=cv_scores.get)

    return fitted_models, cv_scores, cv_winner_key


def generate_latex_equation(model: Any, feature_names: list, outcome_name: str = "y") -> str:
    """
    Extracts a LaTeX formatted equation from a fitted Polynomial Pipeline.
    """
    if getattr(model, 'model_type_', None) != 'Polynomial':
        return "Equation not available for Kriging models (Gaussian Process)."

    poly = model.named_steps['polynomialfeatures']
    lr = model.named_steps['linearregression']

    terms = poly.get_feature_names_out(feature_names)
    # Ensure coefs and intercept are flattened correctly
    import numpy as np
    coefs = lr.coef_[0] if lr.coef_.ndim > 1 else lr.coef_
    intercept = lr.intercept_[0] if np.ndim(lr.intercept_) > 0 else lr.intercept_

    # Format outcome name for LaTeX (escape underscores)
    latex_outcome = outcome_name.replace("_", "\\_")
    equation = f"{latex_outcome} = {intercept:.4g}"

    import re
    for coef, term in zip(coefs, terms):
        # Skip the constant term (intercept is already added) or zero coefficients
        if term == "1" or coef == 0:
            continue

        # Format the feature names for LaTeX
        formatted_term = term.replace(" ", " \\cdot ")
        formatted_term = re.sub(r'\^(\d+)', r'^{\1}', formatted_term)
        formatted_term = formatted_term.replace("_", "\\_")

        sign = "+" if coef > 0 else "-"
        equation += f" {sign} {abs(coef):.4g} \\cdot {formatted_term}"

    return equation

def plot_model_selection(
    cv_scores: dict,
    used_key: tuple | None = None,
    cv_winner_key: tuple | None = None
) -> Any:
    """
    Generates a normalized bar chart of the Bias-Variance Tradeoff from CV scores,
    alongside a sorted table of the exact MSE values in best-fit order.

    Bars are colour-coded to distinguish the CV winner, the user-forced model
    (when different from the CV winner), and all other candidates.

    Args:
        cv_scores (dict): Dictionary mapping ``(model_type, params)`` tuples to
            their Cross-Validation MSE scores.
        used_key (tuple | None): The ``(type, params)`` key of the model that was
            actually used for the PoD calculation. If ``None``, the bar with the
            lowest MSE is treated as the used model.
        cv_winner_key (tuple | None): The ``(type, params)`` key of the CV winner.
            If ``None``, falls back to the bar with the lowest MSE.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # --- 1. Build ordered labels and MSE values ---
    poly_keys = [k for k in cv_scores.keys() if k[0] == 'Polynomial']
    poly_degrees = sorted([k[1] for k in poly_keys])

    labels = []
    mses = []
    keys = []

    for d in poly_degrees:
        labels.append(f"Poly {d}")
        mses.append(cv_scores[('Polynomial', d)])
        keys.append(('Polynomial', d))

    if ('Kriging', None) in cv_scores:
        labels.append("Kriging")
        mses.append(cv_scores[('Kriging', None)])
        keys.append(('Kriging', None))

    # Normalise by minimum error
    min_mse = min(mses)
    normalized_mses = [m / min_mse for m in mses]

    # Resolve which bar is the CV winner and which is the used model
    if cv_winner_key is None:
        cv_winner_key = keys[int(np.argmin(mses))]
    if used_key is None:
        used_key = cv_winner_key

    forced = (used_key != cv_winner_key)

    # --- 2. Assign bar colours ---
    colours = []
    for k in keys:
        if k == cv_winner_key:
            colours.append('crimson')
        elif k == used_key:
            colours.append('#ff7f0e')
        else:
            colours.append('#1f77b4')

    # --- 3. Build the sorted MSE table ---
    sorted_scores = sorted(cv_scores.items(), key=lambda item: item[1])
    table_data = []
    for (m_type, m_param), score in sorted_scores:
        name = f"Poly {m_param}" if m_type == 'Polynomial' else "Kriging"
        table_data.append([name, f"{score:.1e}"])

    # --- 4. Create figure ---
    fig, (ax_plot, ax_table) = plt.subplots(
        1, 2, figsize=(10, 5.5), gridspec_kw={'width_ratios': [2.2, 1]}
    )

    # --- Bar Chart ---
    ax_plot.bar(labels, normalized_mses, color=colours, edgecolor='black', alpha=0.85)

    y_limit = 6
    ax_plot.set_ylim(0, y_limit)

    # Add arrows for cut-off bars
    for i, val in enumerate(normalized_mses):
        if val > y_limit:
            ax_plot.annotate(
                '',
                xy=(i, y_limit),
                xytext=(i, y_limit - 0.4),
                arrowprops=dict(
                    facecolor='black',
                    shrink=0.05,
                    width=2,
                    headwidth=8
                )
            )

    ax_plot.axhline(1.0, color='red', linestyle='-.', linewidth=1.5, label='Min Error (CV)')
    ax_plot.set_title('Model Selection: Bias-Variance Tradeoff', fontweight='bold', pad=25)
    ax_plot.set_ylabel('Error / Min Error [-]')
    ax_plot.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax_plot.tick_params(axis='x', rotation=45)

    # Legend placed cleanly above the plot
    legend_handles = [mpatches.Patch(color='crimson', label='CV Winner')]
    if forced:
        legend_handles.append(mpatches.Patch(color='#ff7f0e', label='Used (Override)'))
    legend_handles.append(mpatches.Patch(color='#1f77b4', label='Other Candidates'))

    ax_plot.legend(
        handles=legend_handles,
        fontsize=9,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02), # Anchors legend just above the title
        ncol=3,
        frameon=False
    )

    # --- Table ---
    ax_table.axis('off')
    ax_table.set_title('MSE Values\n(Best Fit Order)', fontweight='bold')

    # Bounding box [x0, y0, width, height] prevents table from expanding into the title
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Model", "MSE"],
        bbox=[0, 0, 1, 0.85],
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style table rows: bold headers, highlight CV winner and used model
    cv_winner_name = f"Poly {cv_winner_key[1]}" if cv_winner_key[0] == 'Polynomial' else "Kriging"
    used_name = f"Poly {used_key[1]}" if used_key[0] == 'Polynomial' else "Kriging"

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            continue
        cell_label = table_data[row - 1][0]
        if cell_label == cv_winner_name:
            cell.set_facecolor('#ffcccc')
        if forced and cell_label == used_name:
            cell.set_facecolor('#ffe0b2')

    fig.tight_layout(rect=[0, 0, 1, 0.90]) # Leaves a 10% margin at the top for the title and legend
    return fig


#### Variance Model - Kernel Smoothing ####

def optimise_bandwidth(
    X: np.ndarray,
    residuals: np.ndarray,
    min_ratio: float = 0.01,
    max_ratio: float = 0.5
) -> float:
    """
    Finds the optimal kernel smoothing bandwidth using Leave-One-Out Cross-Validation (LOO-CV).

    This function automatically determines the best smoothing window (sigma) for the
    variance model. It evaluates different bandwidths by predicting the squared residual
    of each point using a Gaussian weighted average of all *other* points, selecting
    the bandwidth that minimizes the Mean Squared Error (MSE) of these predictions.

    Args:
        X (np.ndarray): A 1D array of the original input locations (e.g., flaw sizes).
        residuals (np.ndarray): The raw residuals calculated from the mean model
            (differences between observed y and predicted mean y).
        min_ratio (float, optional): The lower bound for the optimizer's search space,
            defined as a fraction of the total range of X (X.max() - X.min()). Defaults to 0.01.
        max_ratio (float, optional): The upper bound for the optimizer's search space,
            defined as a fraction of the total range of X. Defaults to 0.5.

    Returns:
        float: The optimal smoothing bandwidth in the absolute units of X.

    Examples:
        ```python
        import numpy as np

        # 1. Generate dummy input data and simulated residuals
        X = np.linspace(0, 10, 50)
        # Simulate heteroscedastic noise (variance increases with X)
        residuals = np.random.normal(0, X * 0.5, size=50)

        # 2. Find the optimal bandwidth
        optimal_bw = optimize_bandwidth(X, residuals)
        print(f"Optimal Bandwidth: {optimal_bw:.4f}")
        ```
    """
    from scipy.spatial.distance import cdist
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)
    sq_residuals = residuals.flatten() ** 2
    data_range = np.max(X_2d.max(axis=0) - X_2d.min(axis=0))

    def loo_cv_objective(bw: float) -> float:
        # Calculate euclidean distance matrix
        dists = cdist(X_2d, X_2d, metric='euclidean')

        # Calculate Gaussian weights
        weights = stats.norm.pdf(dists, loc=0, scale=bw)

        # Leave-One-Out: Set diagonal to zero so a point doesn't predict itself
        np.fill_diagonal(weights, 0)

        # Normalize weights so each row sums to 1
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0] = 1e-10  # Prevent division by zero
        weights = weights / row_sums.reshape(-1, 1)

        # Predict the squared residuals
        preds = weights @ sq_residuals

        # Return the Mean Squared Error of the variance prediction
        return float(np.mean((sq_residuals - preds) ** 2))

    # Define the search bounds for the optimizer
    bounds = (data_range * min_ratio, data_range * max_ratio)

    # Run a bounded scalar optimization to find the minimum LOO-CV error
    res = minimize_scalar(loo_cv_objective, bounds=bounds, method='bounded')

    return float(res.x)


def fit_variance_model(
    X: np.ndarray,
    y: np.ndarray,
    mean_model: Any,
    auto_bandwidth: bool = True,
    bandwidth_ratio: float = 0.1
) -> Tuple[np.ndarray, float]:
    """
    Calculates residuals and defines the smoothing bandwidth for variance estimation.

    This function acts as the setup phase for modeling heteroscedasticity. It computes
    the raw residuals from the provided mean model and establishes the smoothing
    bandwidth either via automated Cross-Validation or a fixed user-defined ratio.
    It also generates a linearly spaced evaluation grid over the X domain.

    Args:
        X (np.ndarray): The 1D array of original input data (e.g., parameter of interest).
        y (np.ndarray): The 1D array of original outcome data (e.g., signal response).
        mean_model (Any): A fitted scikit-learn estimator (e.g., Pipeline or
            GaussianProcessRegressor) that implements a `.predict()` method.
        auto_bandwidth (bool, optional): If True, dynamically calculates the optimal
            bandwidth using Leave-One-Out Cross-Validation. If False, falls back to
            the fixed `bandwidth_ratio`. Defaults to True.
        bandwidth_ratio (float, optional): The kernel smoothing window size as a
            fraction of the data range (X.max() - X.min()). Only used if
            `auto_bandwidth` is False. Defaults to 0.1.

    Returns:
        Tuple[np.ndarray, float]:
            - residuals: Raw differences between `y` and the mean model predictions.
            - bandwidth: The selected smoothing window size (in absolute units of X).

    Examples:
        ```python
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # 1. Setup dummy data and a basic mean model
        X = np.linspace(0, 10, 50)
        y = 2.5 * X + np.random.normal(0, 1, 50)

        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)

        # 2. Extract residuals and optimized bandwidth
        residuals, bandwidth = fit_variance_model(
            X, y,
            mean_model=model,
            auto_bandwidth=True
        )

        print(f"Calculated Bandwidth: {bandwidth:.4f}")
        print(f"Evaluation Grid Size: {len(X_eval)}")
        ```
    """
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)
    y_pred = mean_model.predict(X_2d)
    residuals = y - y_pred

    if auto_bandwidth:
        print("   -> Optimizing bandwidth via LOO-CV...")
        bandwidth = optimise_bandwidth(X_2d, residuals)
    else:
        data_range = np.max(X_2d.max(axis=0) - X_2d.min(axis=0))
        bandwidth = data_range * bandwidth_ratio

    return residuals, bandwidth


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

    Examples:
        ```python
        local_std = predict_local_std(X, residuals, X_eval, bandwidth)
        ```
    """
    from scipy.spatial.distance import cdist
    X_source = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)
    X_target = np.atleast_2d(X_eval).T if np.asarray(X_eval).ndim == 1 else np.asarray(X_eval)

    sq_residuals = residuals.flatten() ** 2

    dists = cdist(X_target, X_source, metric='euclidean')
    weights = stats.norm.pdf(dists, loc=0, scale=bandwidth)

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

    Examples:
        ```python
        dist_name, dist_params = infer_best_distribution(residuals, X, bandwidth)
        print(f"Best distribution: {dist_name}")
        ```
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

    Examples:
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

    X_eval_2d = np.atleast_2d(X_eval).T if np.asarray(X_eval).ndim == 1 else np.asarray(X_eval)
    mean_curve = mean_model.predict(X_eval_2d)
    sigma_curve = predict_local_std(X, residuals, X_eval_2d, bandwidth)

    z_threshold = (threshold - mean_curve) / sigma_curve

    dist_obj = getattr(stats, dist_name)
    pod_curve = 1 - dist_obj.cdf(z_threshold, *dist_params)

    return pod_curve, mean_curve



def _single_bootstrap_step(
    X_2d, y, X_eval, threshold, model_type, model_params,
    bandwidth, dist_info, nuisance_ranges, n_samples,
    feature_names=None, poi_names=None
):
    """Internal helper to process a single bootstrap iteration."""
    # Resample indices
    idx = np.random.choice(n_samples, n_samples, replace=True)
    X_res_2d = X_2d[idx]
    y_res = y[idx]

    # Fit Mean Model
    if model_type == 'Polynomial':
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        mean_model = make_pipeline(PolynomialFeatures(model_params), LinearRegression())
    elif model_type == 'Kriging':
        from sklearn.gaussian_process import GaussianProcessRegressor
        mean_model = GaussianProcessRegressor(
            kernel=model_params, alpha=np.var(y_res)*0.01, optimizer=None
        )

    mean_model.fit(X_res_2d, y_res)

    # Calculate Residuals
    y_pred = mean_model.predict(X_res_2d)
    res_res = y_res - y_pred

    # Compute PoD for this iteration
    from .integration import compute_multi_dim_pod
    pod_curve, _ = compute_multi_dim_pod(
        X_eval, nuisance_ranges or {}, mean_model, X_res_2d, res_res,
        bandwidth, dist_info, threshold, n_mc_samples=1000,
        feature_names=feature_names, poi_names=poi_names
    )
    return pod_curve




def bootstrap_pod_ci(
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    threshold: float,
    model_type: str,
    model_params: Any,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    n_boot: int = 1000,
    nuisance_ranges: dict = None,
    n_jobs: int | None = None,
    feature_names: list = None,
    poi_names: list = None
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
        n_jobs (int | None, optional): Number of CPU cores to use.
            ``None`` or ``1`` means single-core execution (no parallelisation).
            ``-1`` means use all available cores minus one. Defaults to ``None``.
        feature_names (list, optional): Names of all feature columns in ``X``, in the exact
            same order as the columns appear in ``X``. For one-dimensional inputs this
            can be omitted or contain a single name, but for multi-dimensional
            bootstrapping it is used to identify which variables are parameters of
            interest versus nuisance variables.
        poi_names (list, optional): Names of the parameters of interest (PoIs). Each entry must
            correspond to a name in ``feature_names``. During multi-dimensional
            bootstrapping, PoD curves are evaluated and resampled with respect to these
            variables, while any remaining features in ``X`` are treated as nuisance
            variables. This should therefore be provided whenever ``X`` has multiple
            columns and PoIs need to be distinguished from nuisance inputs.

    Returns:

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - lower_ci: The 2.5th percentile PoD curve (Lower 95% Bound).
            - upper_ci: The 97.5th percentile PoD curve (Upper 95% Bound).

    Examples:
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
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)

    # Standardize joblib convention: None or 1 = single core, -1 = auto max
    if n_jobs is None or n_jobs == 1:
        n_jobs_actual = 1
    elif n_jobs == -1:
        total_cores = os.cpu_count() or 1
        n_jobs_actual = max(total_cores - 1, 1)
    else:
        n_jobs_actual = n_jobs

    # Parallel execution via Joblib
    results = Parallel(n_jobs=n_jobs_actual,backend="multiprocessing", verbose=2)(
        delayed(_single_bootstrap_step)(
            X_2d, y, X_eval, threshold, model_type, model_params,
            bandwidth, dist_info, nuisance_ranges, n_samples,
            feature_names, poi_names # <-- ADD THIS
        ) for _ in range(n_boot)
    )

    pod_matrix = np.array(results)
    return np.percentile(pod_matrix, 2.5, axis=0), np.percentile(pod_matrix, 97.5, axis=0)


def calculate_reliability_point(
    X_eval: np.ndarray,
    ci_lower: np.ndarray,
    target_pod: float = 0.90
) -> float:
    """
    Calculates the defect size (a90/95) where the Lower Confidence Bound
    crosses the target reliability threshold (usually 0.90).

    Args:
        X_eval (np.ndarray): The evaluation grid points.
        ci_lower (np.ndarray): The lower confidence bound curve (y values).
        target_pod (float, optional): Target reliability level. Defaults to 0.90.

    Returns:
        float: The interpolated x-value, or np.nan if not reached.

    Examples:
        ```python
        a90_95 = calculate_reliability_point(X_eval, lower_ci, target_pod=0.90)
        print(f"a90/95 point: {a90_95:.2f}")
        ```
    """
    # Check if the curve actually reaches the target
    if np.max(ci_lower) < target_pod:
        return np.nan

    # Interpolate to find exact crossing point
    # We swap args because we are solving for X given Y=0.90
    monotonic_ci = np.maximum.accumulate(ci_lower)
    return float(np.interp(target_pod, monotonic_ci, X_eval))


def calculate_sobol_indices(mean_model: Any, feature_names: list, data_df, n_samples: int = 1024) -> dict | None:
    """
    Calculates the Total-Order Sobol sensitivity index for the fitted mean model.
    Optimized for speed by disabling second-order interaction matrices.
    """
    try:
        from SALib.sample import sobol as salib_sample
        from SALib.analyze import sobol as salib_analyze
    except ImportError:
        print("Warning: SALib not found. Skipping Sobol index calculation.")
        return None

    # 1. Define the bounds for each feature
    bounds = []
    for col in feature_names:
        bounds.append([float(data_df[col].min()), float(data_df[col].max())])

    problem = {
        'num_vars': len(feature_names),
        'names': feature_names,
        'bounds': bounds
    }

    # 2. FAST SAMPLING: explicitly disable second-order calculations
    X_sample = salib_sample.sample(problem, n_samples, calc_second_order=False)

    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(-1, 1)

    y_sample = mean_model.predict(X_sample)

    # 3. Analyze the results (also disabling second-order here)
    import warnings
    with warnings.catch_warnings():
        # Ignore divide-by-zero warnings if the predicted surface is perfectly flat
        warnings.simplefilter("ignore", RuntimeWarning)
        Si = salib_analyze.analyze(problem, y_sample.flatten(), print_to_console=False, calc_second_order=False)

    # 4. Extract ONLY the Total-Order effect (ST)
    results = {}
    for i, name in enumerate(feature_names):
        results[name] = float(Si['ST'][i])

    return results
