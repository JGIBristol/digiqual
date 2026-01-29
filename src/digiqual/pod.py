import numpy as np
import scipy.stats as stats
from typing import Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold

# ==============================================================================
# 1. MEAN MODEL (Robust Polynomial Regression)
# ==============================================================================

def fit_robust_mean_model(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 10,
    plot_cv: bool = False
) -> Any:
    """
    Automatically selects the best polynomial degree using k-fold Cross Validation.
    """
    X_2d = X.reshape(-1, 1)
    degrees = range(1, max_degree + 1)
    cv_scores = []

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for d in degrees:
        model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
        scores = cross_val_score(model, X_2d, y, cv=cv, scoring='neg_mean_squared_error')
        cv_scores.append(-np.mean(scores))

    best_degree = degrees[np.argmin(cv_scores)]

    if plot_cv:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(degrees, cv_scores, marker='o')
        plt.axvline(best_degree, color='r', linestyle='--', label=f'Best: {best_degree}')
        plt.title('Model Selection: Bias-Variance Tradeoff')
        plt.show()

    final_model = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression())
    final_model.fit(X_2d, y)
    final_model.best_degree_ = best_degree

    return final_model

# ==============================================================================
# 2. VARIANCE MODEL (Kernel Smoothing)
# ==============================================================================

def fit_variance_model(
    X: np.ndarray,
    y: np.ndarray,
    mean_model: Any,
    bandwidth_ratio: float = 0.1,
    n_eval_points: int = 100
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Prepares the Variance Model and Evaluation Grid.
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
    Calculates Local Standard Deviation using Gaussian Kernel Smoothing.
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

# ==============================================================================
# 3. DISTRIBUTION INFERENCE
# ==============================================================================

def infer_best_distribution(
    residuals: np.ndarray,
    X: np.ndarray,
    bandwidth: float
) -> Tuple[str, Tuple]:
    """
    Tests candidate distributions against standardized residuals.
    Returns (best_name, best_params).
    """
    local_std = predict_local_std(X, residuals, X, bandwidth)
    z_scores = residuals.flatten() / local_std.flatten()

    candidates = ["norm", "gumbel_r", "gumbel_l", "logistic", "laplace", "t"]

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

# ==============================================================================
# 4. PREDICTION & BOOTSTRAP
# ==============================================================================

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
    Calculates PoD using the dynamic best-fit distribution.
    """
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
    degree: int,
    bandwidth: float,
    dist_info: Tuple[str, Tuple],
    n_boot: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstraps the PoD curve using the specific distribution parameters found.
    """
    n_samples = len(y)
    pod_matrix = np.zeros((n_boot, len(X_eval)))

    for i in range(n_boot):
        # Resample indices
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_res, y_res = X[idx], y[idx]

        # Fit Mean
        mean_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
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
