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
    poi_name: str = "Parameter of Interest",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Diagnostic Plot 1: Signal vs Parameter of Interest (The Physics).

    Visualizes the raw simulation data, the fitted mean model, and the detection threshold.
    Equivalent to Figure 6/12 in the Generalized Method paper.

    Args:
        X: The raw PoI.
        y: The raw signal responses.
        X_eval: The grid of points used for the curves.
        mean_curve: The predicted mean response at X_eval.
        threshold: The detection threshold (horizontal line).
        local_std: (Optional) The predicted standard deviation at X_eval. If provided, adds 95% prediction bounds to show noise structure.
        poi_name: The label to use for the Parameter of Interest on the x-axis.
        ax: (Optional) Matplotlib axes to plot on. Creates new if None.

    Examples:
        ```python
        import matplotlib.pyplot as plt

        # Plot the physics model
        ax = plot_signal_model(
            X, y, X_eval, mean_curve,
            threshold=3.0,
            local_std=std_curve,
            poi_name="Crack Length (mm)"
        )
        plt.show()
        ```
    """
    if ax is None:
        fig, ax = plt.subplots()

    # 1. Plot Raw Data (Simulations)
    ax.scatter(X, y, alpha=0.5, c='grey', s=20, label='Simulation Data')

    # 2. Plot The Mean Model
    ax.plot(X_eval, mean_curve, color='blue', linewidth=2, label='Mean Response')

    # 3. Plot The Threshold
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold} dB)')

    # 4. (Optional) Plot Prediction Intervals (+/- 2 Sigma)
    if local_std is not None:
        upper = mean_curve + 2 * local_std
        lower = mean_curve - 2 * local_std
        ax.plot(X_eval, upper, color='blue', linestyle=':', alpha=0.6)
        ax.plot(X_eval, lower, color='blue', linestyle=':', alpha=0.6)
        ax.fill_between(
            X_eval, lower, upper,
            color='blue', alpha=0.1,
            label='95% Prediction Interval'
        )

    # Formatting
    ax.set_xlabel(poi_name)
    ax.set_ylabel("Signal Response")
    ax.set_title(f"Signal Response Model ({poi_name})")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_pod_curve(
    X_eval: np.ndarray,
    pod_curve: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    target_pod: float = 0.90,
    poi_name: str = "Parameter of Interest",
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
        poi_name: The label to use for the Parameter of Interest on the x-axis.
        ax: (Optional) Matplotlib axes to plot on.

    Examples:
        ```python
        # Plot the reliability curve with confidence bounds
        ax = plot_pod_curve(
            X_eval, pod_curve,
            ci_lower=lower_bound,
            ci_upper=upper_bound,
            target_pod=0.90,
            poi_name="Crack Length (mm)"
        )
        plt.show()
        ```
    """
    if ax is None:
        fig, ax = plt.subplots()

    # 1. Plot the Main PoD Curve
    ax.plot(X_eval, pod_curve, color='black', linewidth=2, label='PoD Estimate')

    # 2. Plot Confidence Bounds
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            X_eval, ci_lower, ci_upper,
            color='orange', alpha=0.3,
            label='95% Confidence Bounds'
        )
        ax.plot(X_eval, ci_lower, color='orange', linestyle='--', linewidth=1)

    # 3. Mark the a90/95 point (UPDATED FOR PRECISION)
    if ci_lower is not None:
        # Check if we actually reach the target reliability
        if np.max(ci_lower) >= target_pod:
            from digiqual.pod import calculate_reliability_point
            a90_95 = calculate_reliability_point(X_eval, ci_lower, target_pod)

            # Draw the marker lines
            label_text = f"a90/95 = {a90_95:.3f}"
            ax.axvline(a90_95, color='green', linestyle='-.', label=label_text)
            ax.axhline(target_pod, color='green', linestyle=':', alpha=0.5)
            ax.scatter([a90_95], [target_pod], color='green', zorder=5)

    # Formatting
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(poi_name)
    ax.set_ylabel("Probability of Detection")
    ax.set_title(f"PoD Curve ({poi_name})")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax

def plot_pod_surface(
    poi_grids: list,
    pod_curve: np.ndarray,
    poi_names: list,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a 2D heatmap / contour for multi-dimensional PoD (2 Parameters of Interest).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.get_figure()

    grid_x, grid_y = np.meshgrid(poi_grids[0], poi_grids[1], indexing='ij')
    Z = pod_curve.reshape(len(poi_grids[0]), len(poi_grids[1]))

    c = ax.contourf(grid_x, grid_y, Z, levels=np.linspace(0, 1.0, 11), cmap='viridis', alpha=0.9)
    fig.colorbar(c, ax=ax, label="Probability of Detection")

    try:
        # Avoid error if no contours generated
        ax.contour(grid_x, grid_y, Z, levels=[0.90], colors='white', linewidths=2, linestyles='--')
        ax.plot([], [], color='white', linestyle='--', linewidth=2, label='a90/95 Contour')
    except Exception:
        pass

    ax.set_xlabel(poi_names[0])
    ax.set_ylabel(poi_names[1])
    ax.set_title(f"PoD Surface ({poi_names[0]} vs {poi_names[1]})")
    ax.legend(loc='lower right')

    return ax


def plot_signal_surface(
    poi_grids: list,
    mean_curve: np.ndarray,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    threshold: float,
    poi_names: list,
    ax=None
) -> plt.Axes:
    """
    Result Plot 1 (Multi-Dimensional): Signal vs Parameters of Interest.

    Visualizes the raw simulation data in 3D, the fitted mean surface,
    and the detection threshold plane.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # 1. Prepare the grids for the surface
    grid_x, grid_y = np.meshgrid(poi_grids[0], poi_grids[1], indexing='ij')
    Z = mean_curve.reshape(len(poi_grids[0]), len(poi_grids[1]))

    # 2. Plot the fitted mean surface
    ax.plot_surface(grid_x, grid_y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

    # 3. Plot the raw simulation data
    # Assuming X_raw has shape (N, 2) since it's multi-dimensional
    ax.scatter(X_raw[:, 0], X_raw[:, 1], y_raw, color='grey', s=20, label='Simulation Data')

    # 4. Plot the threshold plane
    Z_thresh = np.full_like(Z, threshold)
    ax.plot_surface(grid_x, grid_y, Z_thresh, color='red', alpha=0.3, edgecolor='none')

    # Formatting
    ax.set_xlabel(poi_names[0])
    ax.set_ylabel(poi_names[1])
    ax.set_zlabel("Signal Response")
    ax.set_title(f"Signal Response Surface ({poi_names[0]} vs {poi_names[1]})")

    return ax
