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

    # 3. Plot The Threshold (Updated to round to 2 decimal places)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.2f} dB)')

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
    confidence_level: float = 95,
    poi_name: str = "Parameter of Interest",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Result Plot 2: Probability of Detection (The Reliability).

    Visualizes the PoD curve with Bootstrap Confidence Intervals.
    Equivalent to Figure 11 in the Generalized Method paper.
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
            label=f"{int(confidence_level)}% Confidence Bounds"
        )
        ax.plot(X_eval, ci_lower, color='orange', linestyle='--', linewidth=1)

    # 3. Mark the reliability point
    if ci_lower is not None:
        # Check if we actually reach the target reliability
        if np.max(ci_lower) >= target_pod:
            from digiqual.pod import calculate_reliability_point
            rel_pt = calculate_reliability_point(X_eval, ci_lower, target_pod)

            # Draw the marker lines
            label_text = f"a{int(target_pod*100)}/{int(confidence_level)} = {rel_pt:.3f}"
            ax.axvline(rel_pt, color='green', linestyle='-.', label=label_text)
            ax.axhline(target_pod, color='green', linestyle=':', alpha=0.5)
            ax.scatter([rel_pt], [target_pod], color='green', zorder=5)

    # Formatting
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(poi_name)
    ax.set_ylabel("Probability of Detection")
    ax.set_title(f"PoD Curve ({poi_name})")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_pod_vs_threshold(
    X_eval: np.ndarray,
    thresholds: np.ndarray,
    pod_matrix: np.ndarray,
    poi_name: str = "Parameter of Interest",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots Probability of Detection (y-axis) vs. Detection Threshold (x-axis)
    for a selection of representative defect sizes (flaw sizes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Select 5 representative defect sizes from the X_eval grid
    # E.g. at percentiles 10%, 30%, 50%, 70%, 90% of the evaluated range
    n_points = len(X_eval)
    indices = [int(n_points * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    indices = [min(max(0, idx), n_points - 1) for idx in indices]
    indices = sorted(list(set(indices)))

    for idx in indices:
        size_val = X_eval[idx]
        pod_vs_t = pod_matrix[idx, :]
        ax.plot(thresholds, pod_vs_t, label=f"{poi_name} = {size_val:.2f}", linewidth=2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Detection Threshold")
    ax.set_ylabel("Probability of Detection (Mean)")
    ax.set_title("Probability of Detection vs. Detection Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", title="Defect Size")

    return ax


def plot_pod_surface(
    poi_grids: list,
    pod_curve: np.ndarray,
    poi_names: list,
    ci_lower: Optional[np.ndarray] = None, # <-- NEW ARGUMENT
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a 2D heatmap / contour for multi-dimensional PoD (2 Parameters of Interest).
    Draws the mean a90 contour by default, or the true a90/95 contour if ci_lower is provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.get_figure()

    grid_x, grid_y = np.meshgrid(poi_grids[0], poi_grids[1], indexing='ij')
    Z = pod_curve.reshape(len(poi_grids[0]), len(poi_grids[1]))

    # Always shade the background using the Mean PoD surface
    c = ax.contourf(grid_x, grid_y, Z, levels=np.linspace(0, 1.0, 11), cmap='viridis', alpha=0.9)
    fig.colorbar(c, ax=ax, label="Probability of Detection")

    try:
        if ci_lower is not None:
            # UQ TAB: Draw the a90/95 contour based on the LOWER BOUND surface
            Z_lower = ci_lower.reshape(len(poi_grids[0]), len(poi_grids[1]))
            ax.contour(grid_x, grid_y, Z_lower, levels=[0.90], colors='#d13438', linewidths=2, linestyles='--')
            ax.plot([], [], color='#d13438', linestyle='--', linewidth=2, label='a90/95 Contour')
        else:
            # EXPLORER TAB: Draw the simple a90 contour based on the MEAN surface
            ax.contour(grid_x, grid_y, Z, levels=[0.90], colors='white', linewidths=2, linestyles='--')
            ax.plot([], [], color='white', linestyle='--', linewidth=2, label='a90 Contour (Mean)')
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
    outcome_name: str = "Signal Response",
    ax=None
) -> plt.Axes:
    """
    Result Plot 1 (Multi-Dimensional): Signal vs Parameters of Interest.

    Visualizes the fitted mean surface and the detection threshold plane.
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

    # 4. Plot the threshold plane (Lowered alpha to 0.15 for more transparency)
    Z_thresh = np.full_like(Z, threshold)
    ax.plot_surface(grid_x, grid_y, Z_thresh, color='red', alpha=0.15, edgecolor='none')

    # Formatting (Updated to use dynamic outcome_name)
    ax.set_xlabel(poi_names[0])
    ax.set_ylabel(poi_names[1])
    ax.set_zlabel(outcome_name)
    ax.set_title(f"{outcome_name} Surface ({poi_names[0]} vs {poi_names[1]})")

    return ax


def plot_collinearity_matrix(
    df,
    input_cols: list,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots a Pearson correlation matrix heatmap of the input variables to visualize collinearity.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # 1. Calculate correlation matrix
    corr_matrix = df[input_cols].corr()

    # 2. Plot heatmap (diagonal masked so it doesn't affect the color scale)
    n = len(input_cols)
    matrix_values = corr_matrix.values.copy()
    off_diag_mask = np.eye(n, dtype=bool)
    masked_values = np.ma.masked_array(matrix_values, mask=off_diag_mask)

    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad(color='green')

    im = ax.imshow(masked_values, cmap=cmap, vmin=-1.0, vmax=1.0)

    # 3. Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation Coefficient")

    # 4. Show ticks and label them with input variable names
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(input_cols, rotation=45, ha="right")
    ax.set_yticklabels(input_cols)

    # 5. Annotate each cell with the correlation value
    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            if np.isnan(val):
                text = "N/A"
                text_color = "black"
            elif i == j:
                text = f"{val:.2f}"
                text_color = "white"
            else:
                text = f"{val:.2f}"
                text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontweight="bold")

    ax.set_title("Input Collinearity Matrix (Correlation Heatmap)")
    fig.tight_layout()
    return ax


def plot_kriging_diagnostics(
    std_residuals: np.ndarray,
    outlier_scale_factor: float = 1.0,
    best_kernel_name: str = "Matérn 5/2",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Diagnostic Plot: Visualizes Standardized LOO Residuals and Outlier Calibration.

    Plots a histogram of standardized LOO residuals e_i against the Standard Normal
    distribution N(0, 1) alongside a scatter plot against the [-3, 3] outlier threshold bounds.
    Equivalent to Figure 10 in Malkiel et al. (2026).

    Args:
        std_residuals (np.ndarray): Array of standardized LOO residuals e_i.
        outlier_scale_factor (float, optional): Outlier scaling factor gamma. Defaults to 1.0.
        best_kernel_name (str, optional): Name of the selected Kriging kernel. Defaults to "Matérn 5/2".
        ax (Optional[plt.Axes], optional): Matplotlib axes to plot on. Defaults to None.

    Returns:
        plt.Axes: The configured Matplotlib axis containing the plot.

    Examples:
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        from digiqual.plotting import plot_kriging_diagnostics

        std_res = np.random.normal(0, 1, 50)
        ax = plot_kriging_diagnostics(std_res, outlier_scale_factor=1.2, best_kernel_name="Matérn 5/2")
        plt.show()
        ```
    """
    import scipy.stats as stats

    if ax is None:
        fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(10, 4.5))
    else:
        fig = ax.get_figure()
        ax_hist, ax_scatter = ax, None

    # 1. Histogram of Standardized LOO Residuals vs Standard Normal N(0,1)
    std_res = np.asarray(std_residuals).flatten()
    n, bins, patches = ax_hist.hist(std_res, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='LOO Residuals')
    x_pdf = np.linspace(min(-4.0, std_res.min() - 0.5), max(4.0, std_res.max() + 0.5), 200)
    pdf = stats.norm.pdf(x_pdf, 0, 1)
    ax_hist.plot(x_pdf, pdf, 'r-', linewidth=2, label='Standard Normal N(0,1)')
    ax_hist.set_xlabel("Standardized LOO Residuals e_i")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.set_title(f"Kriging Residual Distribution ({best_kernel_name})")
    ax_hist.legend(loc='upper right')
    ax_hist.grid(True, alpha=0.3)

    if ax_scatter is not None:
        # 2. Scatter plot vs Outlier Threshold Bounds [-3, 3]
        indices = np.arange(len(std_res))
        ax_scatter.scatter(indices, std_res, color='blue', alpha=0.7, s=25, label='LOO Residuals e_i')
        ax_scatter.axhline(3.0, color='red', linestyle='--', linewidth=1.5, label='Outlier Threshold (+/- 3)')
        ax_scatter.axhline(-3.0, color='red', linestyle='--', linewidth=1.5)
        ax_scatter.axhline(0.0, color='black', linestyle=':', alpha=0.5)

        if outlier_scale_factor > 1.0:
            calibrated_res = std_res / np.sqrt(outlier_scale_factor)
            ax_scatter.scatter(indices, calibrated_res, color='green', marker='x', alpha=0.8, s=25, label=f'Calibrated (gamma={outlier_scale_factor:.2f})')

        ax_scatter.set_xlabel("Observation Index")
        ax_scatter.set_ylabel("Standardized Residual")
        ax_scatter.set_title(f"LOO Residual Outliers (gamma={outlier_scale_factor:.2f})")
        ax_scatter.legend(loc='upper right')
        ax_scatter.grid(True, alpha=0.3)

    fig.tight_layout()
    return ax_hist
