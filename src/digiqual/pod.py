import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any, List, Optional
from scipy.stats import norm, gumbel_r, gumbel_l
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold

# --- 1. THE MEAN MODEL (Robust Selection) ---

def fit_robust_mean_model(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 10,
    plot_cv: bool = False
) -> Any:
    """
    Automatically selects the best polynomial degree using k-fold Cross Validation,
    then fits the winning model to the full dataset.

    Args:
        X (np.ndarray): Predictor variable (e.g., crack size 'a').
        y (np.ndarray): Response variable (e.g., signal 'a-hat').
        max_degree (int): The highest polynomial order to test (default 10).
        n_folds (int): Number of CV folds (default 10).
        plot_cv (bool): If True, plots the CV error curve to visualize the "sweet spot".

    Returns:
        model: A fitted sklearn Pipeline (PolynomialFeatures -> LinearRegression).
    """
    # Ensure X is 2D for sklearn
    X_2d = X.reshape(-1, 1)

    # 1. Setup the "Tournament"
    degrees = range(1, max_degree + 1)
    cv_scores = []

    # Shuffle is crucial for simulation data which often comes ordered
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=123)

    # 2. Run Cross-Validation for each degree
    for d in degrees:
        # Create a fresh pipeline for this degree
        candidate_model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())

        # Calculate MSE for each fold (scoring='neg_mean_squared_error' returns negative values)
        scores = cross_val_score(candidate_model, X_2d, y, cv=cv, scoring='neg_mean_squared_error')

        # Convert to positive MSE and store average
        mean_mse = -np.mean(scores)
        cv_scores.append(mean_mse)

    # 3. Pick the Winner
    best_idx = np.argmin(cv_scores)
    best_degree = degrees[best_idx]

    # 4. Optional: Diagnostic Plot (The "Proof" of robustness)
    if plot_cv:
        plt.figure(figsize=(8, 4))
        plt.plot(degrees, cv_scores, marker='o', linestyle='-', color='b', label='CV Error')
        plt.axvline(best_degree, color='r', linestyle='--', label=f'Best Fit (Degree {best_degree})')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Mean Squared Error (Lower is Better)')
        plt.title('Model Selection: Bias-Variance Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 5. Fit the Final Model on ALL data
    # Now that we know the best degree, we train it on the full dataset
    final_model = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression())
    final_model.fit(X_2d, y)

    # Attach the chosen degree as metadata (useful for logging)
    final_model.best_degree_ = best_degree

    return final_model
