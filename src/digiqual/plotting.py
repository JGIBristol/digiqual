import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_signal_model(
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    mean_curve: np.ndarray,
    threshold: float,
    local_std: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Diagnostic Plot 1: Signal vs Crack Length (The Physics).

    Visualizes the raw simulation data, the fitted mean model, and the detection threshold.
    Equivalent to Figure 6/12 in the Generalized Method paper.

    Args:
        X: The raw crack sizes (or PoI).
        y: The raw signal responses.
        X_eval: The grid of points used for the curves.
        mean_curve: The predicted mean response at X_eval.
        threshold: The detection threshold (horizontal line).
        local_std: (Optional) The predicted standard deviation at X_eval. If provided, adds 95% prediction bounds to show noise structure.
        ax: (Optional) Matplotlib axes to plot on. Creates new if None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Plot Raw Data (Simulations)
    ax.scatter(X, y, alpha=0.5, c='grey', s=20, label='Simulation Data')

    # 2. Plot The Mean Model
    ax.plot(X_eval, mean_curve, color='blue', linewidth=2, label='Mean Response')

    # 3. Plot The Threshold
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold} dB)')

    # 4. (Optional) Plot Prediction Intervals (+/- 2 Sigma)
    # This visualizes the "Heteroscedasticity" (changing noise levels)
    if local_std is not None:
        upper = mean_curve + 2 * local_std
        lower = mean_curve - 2 * local_std

        # Plot the bounds
        ax.plot(X_eval, upper, color='blue', linestyle=':', alpha=0.6)
        ax.plot(X_eval, lower, color='blue', linestyle=':', alpha=0.6)

        # Shade the region
        ax.fill_between(
            X_eval, lower, upper,
            color='blue', alpha=0.1,
            label='95% Prediction Interval'
        )

    # Formatting
    ax.set_xlabel("Parameter of Interest (e.g. Crack Length)")
    ax.set_ylabel("Signal Response")
    ax.set_title("Signal Response Model (â vs a)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_pod_curve(
    X_eval: np.ndarray,
    pod_curve: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    target_pod: float = 0.90,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Result Plot 2: Probability of Detection (The Reliability).

    Visualizes the PoD curve with Bootstrap Confidence Intervals.
    Equivalent to Figure 11 in the Generalized Method paper.

    Args:
        X_eval: The grid of points used for the curves.
        pod_curve: The main PoD estimate (0.0 to 1.0).
        ci_lower: (Optional) The lower 95% confidence bound.
        ci_upper: (Optional) The upper 95% confidence bound.
        target_pod: The target reliability level (usually 0.90) to mark on the plot.
        ax: (Optional) Matplotlib axes to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Plot the Main PoD Curve
    ax.plot(X_eval, pod_curve, color='black', linewidth=2, label='PoD Estimate')

    # 2. Plot Confidence Bounds (The "Band of Uncertainty")
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            X_eval, ci_lower, ci_upper,
            color='orange', alpha=0.3,
            label='95% Confidence Bounds'
        )
        # Often in NDE we specifically care about the "Conservative" (Lower) bound
        ax.plot(X_eval, ci_lower, color='orange', linestyle='--', linewidth=1)

    # 3. Mark the a90/95 point (The "Qualifiable Size")
    # We find the crack size where the Lower Bound crosses 90% (or target)
    if ci_lower is not None:
        try:
            # Find index where lower bound exceeds target
            idx = np.where(ci_lower >= target_pod)[0][0]
            a90_95 = X_eval[idx]

            # Draw the marker lines
            ax.axvline(a90_95, color='green', linestyle='-.', label=f'a90/95 = {a90_95:.2f} mm')
            ax.axhline(target_pod, color='green', linestyle=':', alpha=0.5)
            ax.scatter([a90_95], [target_pod], color='green', zorder=5)

        except IndexError:
            # Curve never reached target
            pass

    # Formatting
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Parameter of Interest (e.g. Crack Length)")
    ax.set_ylabel("Probability of Detection")
    ax.set_title("PoD Curve w/ Confidence Bounds")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax
