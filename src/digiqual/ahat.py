import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional
import os
from joblib import Parallel, delayed


def fit_linear_a_hat_model(
    X: np.ndarray,
    y: np.ndarray,
    xlog: bool = False,
    ylog: bool = False
) -> Tuple[LinearRegression, float]:
    """
    Fits a simple linear regression model according to the standard a-hat vs a method.

    Enforces a linear fit and a constant variance (homoskedasticity) assumption across
    all inputs. It allows for optional logarithmic transformations to help linearize
    the data.

    Args:
        X (np.ndarray): 1D array of input parameter values (e.g., flaw size, a).
        y (np.ndarray): 1D array of observed responses (e.g., signal amplitude, a-hat).
        xlog (bool): If True, applies a natural logarithm transformation to X.
        ylog (bool): If True, applies a natural logarithm transformation to y.

    Returns:
        Tuple[LinearRegression, float]:
            - model: The fitted sklearn LinearRegression object.
            - tau: The constant standard deviation of the residuals.

    Examples:
        ```python
        import numpy as np
        X = np.linspace(1, 10, 50)
        y = 2.5 * np.log(X) + np.random.normal(0, 0.5, 50)

        # Fit the standard model, taking the log of X to achieve linearity
        model, tau = fit_linear_a_hat_model(X, y, xlog=True, ylog=False)
        print(f"Constant Standard Deviation: {tau:.4f}")
        ```
    """
    # 1. Apply optional transformations
    X_proc = np.log(X) if xlog else np.copy(X)
    y_proc = np.log(y) if ylog else np.copy(y)

    X_2d = np.atleast_2d(X_proc).T if np.asarray(X_proc).ndim == 1 else np.asarray(X_proc)

    # 2. Fit the linear expectation model
    model = LinearRegression()
    model.fit(X_2d, y_proc)

    # 3. Calculate residuals and enforce constant variance assumption
    y_pred = model.predict(X_2d)
    residuals = y_proc - y_pred

    N = len(y_proc)
    # The variance (tau^2) is calculated as the mean squared error divided by (N-2)
    tau = np.sqrt(np.sum(residuals**2) / (N - 2))

    return model, tau


def compute_linear_pod_curve(
    X_eval: np.ndarray,
    model: LinearRegression,
    tau: float,
    threshold: float,
    xlog: bool = False,
    ylog: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the PoD curve using the standard a-hat vs a analytical assumptions.

    Assumes that the residual errors are perfectly normally distributed with a
    constant standard deviation (tau).

    Args:
        X_eval (np.ndarray): The 1D grid of points to evaluate the PoD curve.
        model (LinearRegression): The fitted linear expectation model.
        tau (float): The constant standard deviation of the residuals.
        threshold (float): The detection threshold in original (untransformed) units.
        xlog (bool): Indicates if the model was trained with log-transformed X.
        ylog (bool): Indicates if the model was trained with log-transformed y.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pod_curve: Array of probabilities [0, 1] for each point in X_eval.
            - mean_curve: Array of expected mean signal responses in original units.

    Examples:
        ```python
        # Assuming we have the model and tau from the fitting step
        X_eval = np.linspace(1, 10, 100)
        pod, mean_response = compute_linear_pod_curve(
            X_eval, model, tau, threshold=3.5, xlog=True, ylog=False
        )
        ```
    """
    # 1. Transform inputs for evaluation
    X_eval_proc = np.log(X_eval) if xlog else np.copy(X_eval)
    X_eval_2d = np.atleast_2d(X_eval_proc).T if np.asarray(X_eval_proc).ndim == 1 else np.asarray(X_eval_proc)

    # 2. Get predictions from the linear model
    mean_pred_proc = model.predict(X_eval_2d)

    # 3. Transform the threshold if needed
    threshold_proc = np.log(threshold) if ylog else threshold

    # 4. Calculate probability of exceedance (Assuming Gaussian Distribution)
    z_scores = (threshold_proc - mean_pred_proc) / tau
    pod_curve = 1.0 - stats.norm.cdf(z_scores)

    # 5. Reverse transformation for the mean response curve
    mean_curve = np.exp(mean_pred_proc) if ylog else mean_pred_proc

    return pod_curve, mean_curve


def _single_linear_bootstrap_step(X, y, X_eval, threshold, xlog, ylog):
    """Internal helper to process a single linear bootstrap iteration."""
    # 1. Resample the data with replacement
    n_samples = len(X)
    idx = np.random.choice(n_samples, n_samples, replace=True)
    X_res = X[idx]
    y_res = y[idx]

    # 2. Fit the strict linear model & calculate constant variance (tau)
    model, tau = fit_linear_a_hat_model(X_res, y_res, xlog=xlog, ylog=ylog)

    # 3. Compute the classical PoD curve for this specific resample
    pod_curve, _ = compute_linear_pod_curve(
        X_eval, model, tau, threshold, xlog=xlog, ylog=ylog
    )
    return pod_curve


def bootstrap_linear_pod_ci(
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    threshold: float,
    xlog: bool = False,
    ylog: bool = False,
    n_boot: int = 1000,
    n_jobs: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates 95% Confidence Bounds for the classical linear PoD curve via Bootstrapping.
    Maintains the strict assumptions of constant variance and normally distributed errors.
    """
    if n_jobs is None or n_jobs == 1:
        n_jobs_actual = 1
    elif n_jobs == -1:
        n_jobs_actual = max((os.cpu_count() or 1) - 1, 1)
    else:
        n_jobs_actual = n_jobs

    # Run the bootstrap steps in parallel
    results = Parallel(n_jobs=n_jobs_actual, verbose=0)(
        delayed(_single_linear_bootstrap_step)(
            X, y, X_eval, threshold, xlog, ylog
        ) for _ in range(n_boot)
    )

    pod_matrix = np.array(results)

    # Return the 2.5% (Lower Bound) and 97.5% (Upper Bound) curves
    return np.percentile(pod_matrix, 2.5, axis=0), np.percentile(pod_matrix, 97.5, axis=0)



def plot_linear_signal_model(
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    model: LinearRegression,
    threshold: float,
    tau: float,
    xlog: bool = False,
    ylog: bool = False,
    poi_name: str = "Parameter of Interest",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Diagnostic Plot: Visualizes the standard linear a-hat vs a model.

    Plots the raw data, the linear expectation model, the constant 95%
    prediction interval (calculated using tau), and the target threshold.
    It automatically scales the plot axes based on the chosen log transformations.

    Args:
        X (np.ndarray): Original simulation inputs.
        y (np.ndarray): Original simulation outcomes.
        X_eval (np.ndarray): Grid used for curve evaluation.
        model (LinearRegression): The fitted linear expectation model.
        threshold (float): The detection threshold limit.
        tau (float): The constant standard deviation of the residuals.
        xlog (bool): Set to True if the model used log-transformed X.
        ylog (bool): Set to True if the model used log-transformed y.
        poi_name (str): Label for the x-axis. Defaults to "Parameter of Interest".
        ax (Optional[plt.Axes]): Existing Matplotlib axes. Created if None.

    Returns:
        plt.Axes: The configured Matplotlib axis containing the plot.

    Examples:
        ```python
        import matplotlib.pyplot as plt

        ax = plot_linear_signal_model(
            X, y, X_eval, model, threshold=3.5, tau=tau,
            xlog=True, ylog=False, poi_name="Crack Length (mm)"
        )
        plt.show()
        ```
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate transformations and bounds
    X_eval_proc = np.log(X_eval) if xlog else np.copy(X_eval)
    X_eval_2d = np.atleast_2d(X_eval_proc).T if np.asarray(X_eval_proc).ndim == 1 else np.asarray(X_eval_proc)

    mean_pred_proc = model.predict(X_eval_2d)

    # Standard constant variance assumption applied at +/- 2 Standard Deviations
    upper_proc = mean_pred_proc + 2 * tau
    lower_proc = mean_pred_proc - 2 * tau

    # Back transform everything to the physical scale for plotting
    mean_curve = np.exp(mean_pred_proc) if ylog else mean_pred_proc
    upper_bound = np.exp(upper_proc) if ylog else upper_proc
    lower_bound = np.exp(lower_proc) if ylog else lower_proc

    # Scatter of raw data
    ax.scatter(X, y, alpha=0.5, c='grey', s=20, label='Simulation Data')

    # Regression line
    ax.plot(X_eval, mean_curve, color='blue', linewidth=2, label='Mean Response')

    # Threshold Plane
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')

    # Prediction Intervals
    ax.plot(X_eval, upper_bound, color='blue', linestyle=':', alpha=0.6)
    ax.plot(X_eval, lower_bound, color='blue', linestyle=':', alpha=0.6)
    ax.fill_between(
        X_eval, lower_bound, upper_bound,
        color='blue', alpha=0.1,
        label='95% Prediction Interval (Constant SD)'
    )

    # Format Axes Scaling
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    # Formatting and Styling
    ax.set_xlabel(poi_name)
    ax.set_ylabel("Signal Response")
    ax.set_title(f"Standard $\hat{{a}}$ vs $a$ Model ({poi_name})")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which="both", ls="--")

    return ax
