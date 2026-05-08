import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
import os

from .diagnostics import validate_simulation, sample_sufficiency, ValidationError
from .adaptive import generate_targeted_samples, run_adaptive_search
from .plotting import plot_signal_model, plot_pod_curve
from . import pod
from .executors import Executor
from .ahat import fit_linear_a_hat_model, compute_linear_pod_curve, plot_linear_signal_model, bootstrap_linear_pod_ci



class SimulationStudy:
    """
    A workflow manager for simulation reliability assessment.

    Args:
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Name of the outcome variable.
        max_gap_ratio (float, optional): Max allowable gap between data points as a fraction of the total range. Defaults to 0.20.
        min_r2_score (float, optional): Minimum cross-validated R-squared score required for the signal fit. Defaults to 0.50.
        max_avg_cv (float, optional): Max allowable average relative width of the bootstrap predictions. Defaults to 0.15.
        max_max_cv (float, optional): Max allowable relative width at the tail ends (10th/90th percentiles). Defaults to 0.30.

    Attributes:
        inputs (List[str]): List of input variable names.
        outcome (str): Name of the outcome variable.
        data (pd.DataFrame): The raw simulation data.
        clean_data (pd.DataFrame): Data that has passed validation.
        sufficiency_results (pd.DataFrame): The latest diagnostic results.
        pod_results (Dict): Results from the latest PoD analysis.
        plots (Dict): Stores the latest generated figures.
        linear_pod_results (Dict): Results from the classical linear analysis.
        linear_plots (Dict): Stores figures from the classical linear analysis.

    Examples:
        ```python
        from digiqual.core import SimulationStudy

        # Initialize with stricter diagnostic thresholds
        study = SimulationStudy(
            input_cols=['Length', 'Angle'],
            outcome_col='Signal',
            max_gap_ratio=0.10,
            min_r2_score=0.75
        )
        ```
    """

#### Initialisation ####
    def __init__(
        self,
        input_cols: List[str],
        outcome_col: str,
        max_gap_ratio: float = 0.20,
        min_r2_score: float = 0.50,
        max_avg_cv: float = 0.15,
        max_max_cv: float = 0.30
    ):
        self.inputs = input_cols
        self.outcome = outcome_col

        # Save custom diagnostic thresholds
        self.max_gap_ratio = max_gap_ratio
        self.min_r2_score = min_r2_score
        self.max_avg_cv = max_avg_cv
        self.max_max_cv = max_max_cv

        # Internal state
        self.data: pd.DataFrame = pd.DataFrame()
        self.clean_data: pd.DataFrame = pd.DataFrame()
        self.removed_data: pd.DataFrame = pd.DataFrame()

        # Generalized method storage
        self.sufficiency_results: pd.DataFrame = pd.DataFrame()
        self.pod_results: Dict[str, Any] = {}
        self.plots: Dict[str, Any] = {}

        # Linear a-hat vs a method storage
        self.linear_pod_results: Dict[str, Any] = {}
        self.linear_plots: Dict[str, Any] = {}

#### Adding Data ####
    def add_data(self, df: pd.DataFrame) -> None:
        """
        Ingests raw simulation data, filtering for relevant columns only.

        This method automatically strips away any columns in `df` that were not
        defined in `self.inputs` or `self.outcome` during initialization.

        Args:
            df (pd.DataFrame): The DataFrame to ingest.

        Raises:
            ValueError: If the new data is missing any required input or outcome columns.

        Examples:
            ```python
            import pandas as pd
            df = pd.DataFrame({
                'Length': [1.0, 2.5, 5.0],
                'Angle': [0, 15, -10],
                'Signal': [0.5, 0.8, 1.2]
            })
            study.add_data(df)
            ```
        """
        required_cols = self.inputs + [self.outcome]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"New data is missing required columns: {missing}")

        df_subset = df[required_cols].copy()

        if self.data.empty:
            self.data = df_subset
        else:
            self.data = pd.concat([self.data, df_subset], ignore_index=True)

        print(f"Data updated. Total rows: {len(self.data)}")

        # RESET State
        self.clean_data = pd.DataFrame()
        self.sufficiency_results = pd.DataFrame()
        self.pod_results = {}
        self.plots = {}
        self.linear_pod_results = {}
        self.linear_plots = {}

#### Validating self.data ####
    def _validate(self) -> None:
        """
        Cleans and validates the raw data stored in `self.data`.

        Populates `self.clean_data` with valid rows and `self.removed_data`
        with invalid ones (e.g., NaNs, negative signals, wrong types).

        Examples:
            ```python
            study._validate()
            # Output: Running validation...
            # Output: Validation passed. 50 valid rows ready.
            ```
        """
        print("Running validation...")
        try:
            clean, removed = validate_simulation(self.data, self.inputs, self.outcome)
            self.clean_data = clean
            self.removed_data = removed
            print(f"Validation passed. {len(clean)} valid rows ready.")
            if not removed.empty:
                print(f"Warning: {len(removed)} invalid rows were dropped. See .removed_data")
        except ValidationError as e:
            print(f"Validation FAILED: {e}")
            self.clean_data = pd.DataFrame()

#### Checking Sample Sufficiency ####
    def diagnose(self) -> pd.DataFrame:
        """
        Runs statistical diagnostics to evaluate if the current sample size is sufficient.
        """
        if self.clean_data.empty:
            if self.data.empty:
                print("No data found. Please run add_data() first.")
                return pd.DataFrame()

            self._validate()

            if self.clean_data.empty:
                print("Cannot run diagnostics because validation failed.")
                return pd.DataFrame()

        print("Checking sample sufficiency...")

        # Pass the class-level thresholds into the diagnostic function
        self.sufficiency_results = sample_sufficiency(
            self.clean_data, self.inputs, self.outcome,
            skip_validation=True,
            max_gap_ratio=self.max_gap_ratio,
            min_r2_score=self.min_r2_score,
            max_avg_cv=self.max_avg_cv,
            max_max_cv=self.max_max_cv
        )
        return self.sufficiency_results

#### Adaptive Refinement ####
    def refine(self, n_points: int = 10) -> pd.DataFrame:
        """
        Identifies gaps or high-variance regions and suggests new simulation points.

        Args:
            n_points (int): Number of new samples to suggest per detected issue.

        Returns:
            pd.DataFrame: A DataFrame of recommended new input coordinates.

        Examples:
            ```python
            # If diagnostics fail, ask for 10 new points to fix it
            new_samples = study.refine(n_points=10)
            print(new_samples.head())
            ```
        """

        if self.clean_data.empty:
            print("No clean data available. Running validation...")
            self._validate()

            if self.clean_data.empty:
                return pd.DataFrame()

        new_samples = generate_targeted_samples(
            df=self.clean_data,
            input_cols=self.inputs,
            outcome_col=self.outcome,
            n_new_per_fix=n_points,
            failed_data=self.removed_data,
            max_gap_ratio=self.max_gap_ratio,
            min_r2_score=self.min_r2_score,
            max_avg_cv=self.max_avg_cv,
            max_max_cv=self.max_max_cv
        )

        return new_samples

#### Automated Optimisation ####
    def optimise(
        self,
        executor: Union[Executor, str],
        ranges: Dict[str, Tuple[float, float]],
        n_start: int = 20,
        n_step: int = 10,
        max_iter: int = 5,
        max_hours: float = None
    ) -> None:
        """
        Runs the automated Active Learning loop (Initialize -> Execute -> Diagnose -> Refine).

        Args:
            executor (Executor | str): The solver adapter to use (e.g., PythonExecutor, MatlabExecutor).
                                        Accepts a legacy command string for backward compatibility.
            ranges (Dict): Input bounds, e.g. {"Length": (0, 10)}.
            n_start (int): Initial sample size (only if data is empty).
            n_step (int): Batch size for refinement.
            max_iter (int): Max refinement loops.
            max_hours (float, optional): Physical time limit in hours to safely stop the loop.

        Examples:
            ```python
            from digiqual.core import SimulationStudy
            from digiqual.executors import PythonExecutor

            # 1. Define the variable ranges
            ranges = {"Length": (0, 10), "Angle": (-45, 45)}
            study = SimulationStudy(input_cols=["Length", "Angle"], outcome_col="Signal")

            # 2. Define a simple Python solver
            def my_solver(row):
                return row['Length'] * 2 + row['Angle']

            my_exec = PythonExecutor(solver_func=my_solver, outcome_col="Signal")

            # 3. Run the automated loop
            study.optimise(
                executor=my_exec,
                ranges=ranges,
                max_iter=3
            )

            # 4. View the results
            _ = study.pod(poi_col="Length", threshold=4.0)
            study.visualise()
            ```
        """

        # --- NEW SAFEGUARD: Validate input ranges ---
        expected_inputs = set(self.inputs)
        provided_ranges = set(ranges.keys())

        if expected_inputs != provided_ranges:
            raise ValueError(
                f"Variable Mismatch! The keys in your 'ranges' dictionary {list(provided_ranges)} "
                f"do not match the 'input_cols' {list(expected_inputs)} defined in the SimulationStudy."
            )

        # 1. Delegate to the Agnostic Engine
        final_data = run_adaptive_search(
            executor=executor,            # <-- Passes our new Executor object
            input_cols=self.inputs,
            outcome_col=self.outcome,
            ranges=ranges,
            existing_data=self.data,
            n_start=n_start,
            n_step=n_step,
            max_iter=max_iter,
            max_hours=max_hours,
            # --- The 4 Custom Diagnostic Thresholds ---
            max_gap_ratio=self.max_gap_ratio,
            min_r2_score=self.min_r2_score,
            max_avg_cv=self.max_avg_cv,
            max_max_cv=self.max_max_cv
        )

        # 2. Update Class State with the result
        self.data = pd.DataFrame() # Clear old state to avoid duplication
        self.add_data(final_data)

#### PoD Analysis ####
    def pod(
        self,
        poi_col: list | str,
        threshold: float,
        nuisance_col: list | str | None = None,
        slice_values: dict | None = None,
        bandwidth_ratio: float = 0.1,
        n_boot: int = 1000,
        model_override: str = "auto",
        force_degree: int | None = None,
        n_jobs: int | None = None
    ) -> Dict[str, Any]:
        """
        Runs the generalized Probability of Detection (PoD) analysis.

        Args:
            poi_col (list | str): The parameter(s) of interest (e.g., 'Crack Length', ['Angle', 'Depth']).
            threshold (float): The failure threshold (e.g., 4.0 dB).
            nuisance_col (list | str | None): The nuisance parameters to marginalize over via MC integration.
            slice_values (dict | None): The sliced parameters and their values.
            bandwidth_ratio (float): Smoothing bandwidth fraction (default 0.1).
            n_boot (int): Bootstrap iterations for confidence bounds.
            model_override (str): Force a model type. One of "auto",
                "polynomial", or "kriging". Defaults to "auto".
            force_degree (int | None): When model_override="polynomial",
                use this degree. Defaults to None (CV selects).
            n_jobs (int | None): Number of CPU cores for parallel bootstrap execution.
                Defaults to ``None`` (single-core). Set to ``-1`` to auto-detect and use all available cores.

        Returns:
            Dict: Dictionary containing models, curves, and fit statistics.

        Examples:
            ```python
            results = study.pod(poi_col="Length", threshold=0.5)
            print(results['dist_info'])
            ```
        """
        # 0. Safety Checks
        if self.clean_data.empty:
            self._validate()
            if self.clean_data.empty:
                raise ValueError("Cannot run PoD analysis: No valid data available.")

        if isinstance(poi_col, str):
            poi_cols = [poi_col]
        else:
            poi_cols = poi_col

        # --- ADD THIS SAFETY CHECK ---
        if len(poi_cols) > 2:
            raise ValueError("digiqual currently only supports plotting 1 or 2 Parameters of Interest. "
                                "Please move additional variables to 'nuisance_col' or 'slice_values'.")

        if nuisance_col is None:
            nuisance_cols = []
        elif isinstance(nuisance_col, str):
            nuisance_cols = [nuisance_col]
        else:
            nuisance_cols = nuisance_col

        slice_values = slice_values or {}

        # 1. The model is ALWAYS trained on all initialized inputs
        all_cols = self.inputs

        # Validate that requested PoIs and Nuisances are actually in the dataset
        for c in poi_cols + nuisance_cols:
            if c not in all_cols:
                raise ValueError(f"Variable '{c}' is not in the initialized input_cols.")

        # 2. Automatically handle sliced parameters (Leftovers)
        final_slice_values = {}
        for c in all_cols:
            if c not in poi_cols and c not in nuisance_cols:
                if c in slice_values:
                    # Use the specific value the user provided
                    final_slice_values[c] = float(slice_values[c])
                else:
                    # Default to the median of the dataset
                    final_slice_values[c] = float(self.clean_data[c].median())

        # 3. Prepare Data Vectors
        print(f"--- Starting Reliability Analysis (PoIs: {poi_cols}) ---")
        if final_slice_values:
            print(f"-> Slicing surface at: {final_slice_values}")

        X = self.clean_data[all_cols].values
        y = self.clean_data[self.outcome].values

        # 4. Fit Mean Model (Robust Regression)
        print("1. Selecting Mean Model (Cross-Validation)...")
        mean_model = pod.fit_robust_mean_model(
            X, y, model_override=model_override, force_degree=force_degree
        )
        equation = pod.generate_latex_equation(mean_model, all_cols, self.outcome)

        if mean_model.model_type_ == 'Polynomial':
            print(f"-> Selected Model: Polynomial (Degree {mean_model.model_params_})")
            print("-> Equation extracted.")
        else:
            print("-> Selected Model: Kriging (Gaussian Process)")

        # 5. Fit Variance Model & Generate Grid
        print("2. Fitting Variance Model (Kernel Smoothing)...")
        residuals, bandwidth = pod.fit_variance_model(
            X, y, mean_model, bandwidth_ratio=bandwidth_ratio
        )
        print(f"   -> Smoothing Bandwidth: {bandwidth:.4f}")

        # 6. Infer Distribution
        print("3. Inferring Error Distribution (AIC)...")
        dist_name, dist_params = pod.infer_best_distribution(residuals, X, bandwidth)
        print(f"   -> Selected Distribution: {dist_name}")

        # Build PoI grid and nuisance ranges
        poi_grids = []
        for col in poi_cols:
            num_points = 100 if len(poi_cols) == 1 else 30
            poi_grids.append(np.linspace(self.clean_data[col].min(), self.clean_data[col].max(), num_points))

        if len(poi_cols) == 1:
            X_eval = poi_grids[0].reshape(-1, 1)
        else:
            mesh = np.meshgrid(*poi_grids, indexing='ij')
            X_eval = np.vstack([m.flatten() for m in mesh]).T

        # Build nuisance ranges for true nuisances
        nuisance_ranges = {c: (float(self.clean_data[c].min()), float(self.clean_data[c].max())) for c in nuisance_cols}

        # Inject the sliced parameters into the integrator with Min == Max
        for c, val in final_slice_values.items():
            nuisance_ranges[c] = (val, val)

        # 7. Compute PoD Curve
        print("4. Computing PoD Curve...")
        from digiqual.integration import compute_multi_dim_pod
        pod_curve, mean_curve = compute_multi_dim_pod(
            X_eval, nuisance_ranges, mean_model, X, residuals, bandwidth, (dist_name, dist_params), threshold
        )

        # 8. Bootstrap Confidence Intervals (Parallelized)
        if n_boot > 0:
            if n_jobs is None or n_jobs == 1:
                actual_cores = 1
            elif n_jobs == -1:
                actual_cores = max((os.cpu_count() or 1) - 1, 1)
            else:
                actual_cores = n_jobs

            print(f"5. Running Bootstrap ({n_boot} iterations on {actual_cores} cores)...", flush=True)

            lower_ci, upper_ci = pod.bootstrap_pod_ci(
                X, y, X_eval, threshold,
                mean_model.model_type_, mean_model.model_params_, bandwidth, (dist_name, dist_params),
                n_boot=n_boot, nuisance_ranges=nuisance_ranges,
                n_jobs=n_jobs
            )
        else:
            print("5. Skipping Bootstrap (n_boot=0)...", flush=True)
            lower_ci, upper_ci = None, None

        # 9. Calculate Reliability Point
        a90_95 = np.nan
        if len(poi_cols) == 1 and lower_ci is not None:
            a90_95 = pod.calculate_reliability_point(X_eval.flatten(), lower_ci, target_pod=0.90)
            print(f"   -> a90/95 Reliability Index: {a90_95:.3f}")

        # 10. Package Results
        self.pod_results = {
            "poi_cols": poi_cols,
            "nuisance_cols": nuisance_cols,
            "slice_values": final_slice_values,
            "threshold": threshold,
            "n_boot" : n_boot,
            "a90_95": a90_95,
            "X": X,
            "y": y,
            "X_eval": X_eval,
            "poi_grids": poi_grids,
            "mean_model": mean_model,
            "equation": equation,
            "bandwidth": bandwidth,
            "residuals": residuals,
            "dist_info": (dist_name, dist_params),
            "curves": {
                "mean_response": mean_curve,
                "pod": pod_curve,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci
            }
        }

        print("--- Analysis Complete ---")
        return self.pod_results

#### Real-Time Slice Evaluation ####
    def update_slice(self, slice_values: dict) -> Dict[str, Any]:
        """
        Updates the evaluated slice for the PoD surface without re-fitting the surrogate model.
        This allows for real-time plot updates when changing constant parameters.
        """
        if not self.pod_results:
            return {}

        # 1. Retrieve the frozen n-dimensional model and state
        mean_model = self.pod_results["mean_model"]
        bandwidth = self.pod_results["bandwidth"]
        residuals = self.pod_results["residuals"]
        dist_info = self.pod_results["dist_info"]
        poi_cols = self.pod_results["poi_cols"]
        nuisance_cols = self.pod_results["nuisance_cols"]
        threshold = self.pod_results["threshold"]
        X_eval = self.pod_results["X_eval"]
        X = self.pod_results["X"]

        # 2. Build the final slice values using median fallbacks
        final_slice_values = {}
        for c in self.inputs:
            if c not in poi_cols and c not in nuisance_cols:
                if c in slice_values:
                    final_slice_values[c] = float(slice_values[c])
                else:
                    final_slice_values[c] = float(self.clean_data[c].median())

        # 3. Inject the new sliced parameters into the nuisance ranges
        nuisance_ranges = {c: (float(self.clean_data[c].min()), float(self.clean_data[c].max())) for c in nuisance_cols}
        for c, val in final_slice_values.items():
            nuisance_ranges[c] = (val, val)

        # 4. Re-calculate the PoD and Mean curves instantly
        from digiqual.integration import compute_multi_dim_pod
        pod_curve, mean_curve = compute_multi_dim_pod(
            X_eval, nuisance_ranges, mean_model, X, residuals, bandwidth, dist_info, threshold
        )

        # 5. Update the internal results dictionary
        self.pod_results["slice_values"] = final_slice_values
        self.pod_results["curves"]["mean_response"] = mean_curve
        self.pod_results["curves"]["pod"] = pod_curve

        # Invalidate CI bounds as they apply to the previous slice
        self.pod_results["curves"]["ci_lower"] = None
        self.pod_results["curves"]["ci_upper"] = None
        self.pod_results["a90_95"] = np.nan

        return self.pod_results

#### Visualise Results ####
    def visualise(self, show: bool = True, save_path: str = None) -> None:
        """
        Generates and displays diagnostic plots (Signal Model and PoD Curve).

        Args:
            show (bool): If True, calls plt.show().
            save_path (str, optional): If provided, saves figures to disk (e.g., 'results/run1'). Appends '_signal.png' and '_pod.png'.

        Examples:
            ```python
            # Display only
            study.visualise()

            # Save to disk
            study.visualise(show=False, save_path='my_plots')
            ```
        """
        if not self.pod_results:
            print("No PoD results found. Please run .pod() first.")
            return

        res = self.pod_results
        poi_cols = res.get("poi_cols", ["Parameter of Interest"])

        # --- FIXED BLOCK: Multidimensional Local SD Calculation ---
        local_std = None
        if len(poi_cols) == 1:
            # 1. Create a full-dimension evaluation grid that matches the training data (e.g., 7 columns)
            n_samples_eval = len(res["X_eval"])
            n_total_vars = len(self.inputs)
            X_eval_full = np.zeros((n_samples_eval, n_total_vars))

            # 2. Gather the fixed values for this specific plot slice
            # We use the current slice values, and fallback to medians for nuisances
            current_view_values = res["slice_values"].copy()
            for c in res["nuisance_cols"]:
                current_view_values[c] = float(self.clean_data[c].median())

            # 3. Populate the high-dimensional grid for the noise estimator
            for i, col_name in enumerate(self.inputs):
                if col_name == poi_cols[0]:
                    # This column gets the sliding PoI values (x-axis)
                    X_eval_full[:, i] = res["X_eval"].flatten()
                else:
                    # These columns get the constant slice/nuisance values
                    X_eval_full[:, i] = current_view_values[col_name]

            # 4. Now the dimensions match (7 columns vs 7 columns)!
            local_std = pod.predict_local_std(
                res["X"], res["residuals"], X_eval_full, res["bandwidth"]
            )

        if hasattr(res["mean_model"], "cv_scores_"):
            mean_model = res["mean_model"]
            model_type = getattr(mean_model, 'model_type_', None)
            model_params = getattr(mean_model, 'model_params_', None)
            cv_winner_key = getattr(mean_model, 'cv_winner_', None)

            if model_type == 'Polynomial' and model_params is not None:
                used_key = ('Polynomial', model_params)
            elif model_type == 'Kriging':
                used_key = ('Kriging', None)
            else:
                used_key = None  # Let plot_model_selection fall back to argmin

            self.plots["model_selection"] = pod.plot_model_selection(
                mean_model.cv_scores_,
                used_key=used_key,
                cv_winner_key=cv_winner_key
            )

        # 1. Signal Model Plot
        if len(poi_cols) == 1:
            poi_idx = self.inputs.index(poi_cols[0])
            self.plots["signal_model"] = plot_signal_model(
                X=res["X"][:, poi_idx],  # Safely grabbed the correct column!
                y=res["y"],
                X_eval=res["X_eval"].flatten(),
                mean_curve=res["curves"]["mean_response"],
                threshold=res["threshold"],
                local_std=local_std,
                poi_name=poi_cols[0]
            )
        else:
            from digiqual.plotting import plot_signal_surface
            poi_idx_0 = self.inputs.index(poi_cols[0])
            poi_idx_1 = self.inputs.index(poi_cols[1])

            self.plots["signal_model"] = plot_signal_surface(
                poi_grids=res["poi_grids"],
                mean_curve=res["curves"]["mean_response"],
                X_raw=res["X"][:, [poi_idx_0, poi_idx_1]], # Safely grabbed the 2 columns!
                y_raw=res["y"],
                threshold=res["threshold"],
                poi_names=poi_cols,
                outcome_name=self.outcome
            )

        # 2. PoD Curve/Surface Plot
        if res["curves"]["ci_lower"] is not None:
            if len(poi_cols) == 1:
                self.plots["pod_curve"] = plot_pod_curve(
                    X_eval=res["X_eval"].flatten(),
                    pod_curve=res["curves"]["pod"],
                    ci_lower=res["curves"]["ci_lower"],
                    ci_upper=res["curves"]["ci_upper"],
                    target_pod=0.90,
                    poi_name=poi_cols[0]
                )
            else:
                from digiqual.plotting import plot_pod_surface
                self.plots["pod_curve"] = plot_pod_surface(
                    poi_grids=res["poi_grids"],
                    pod_curve=res["curves"]["pod"],
                    poi_names=poi_cols
                )

        # Handle Saving
        if save_path:
            if "model_selection" in self.plots:
                self.plots["model_selection"].savefig(f"{save_path}_model_selection.png")
            self.plots["signal_model"].get_figure().savefig(f"{save_path}_signal.png")
            self.plots["pod_curve"].get_figure().savefig(f"{save_path}_pod.png")
            print(f"Plots saved to {save_path}_*.png")

        # Handle Display
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except ImportError:
                pass

#### Linear a-hat vs a Analysis ####
    def linear_pod(
        self,
        poi_col: str,
        threshold: float,
        xlog: bool = False,
        ylog: bool = False,
        n_boot: int = 1000,
        n_jobs: int | None = None
    ) -> Dict[str, Any]:
        """
        Runs the classical linear a-hat vs a PoD analysis with bootstrapped bounds.
        """
        if self.clean_data.empty:
            self._validate()
            if self.clean_data.empty:
                raise ValueError("Cannot run linear PoD analysis: No valid data available.")

        if poi_col not in self.clean_data.columns:
            raise ValueError(f"Variable '{poi_col}' not found in data columns.")

        print(f"--- Starting Linear a-hat vs a Analysis (PoI: {poi_col}) ---")

        X = self.clean_data[poi_col].values
        y = self.clean_data[self.outcome].values

        # 1. Fit Base Model
        model, tau = fit_linear_a_hat_model(X, y, xlog=xlog, ylog=ylog)
        X_eval = np.linspace(X.min(), X.max(), 100)
        pod_curve, mean_curve = compute_linear_pod_curve(
            X_eval=X_eval, model=model, tau=tau, threshold=threshold, xlog=xlog, ylog=ylog
        )

        # 2. Bootstrap Confidence Intervals
        print(f"Running Bootstrap ({n_boot} iterations) to establish classical confidence bounds...")
        lower_ci, upper_ci = bootstrap_linear_pod_ci(
            X, y, X_eval, threshold, xlog, ylog, n_boot, n_jobs
        )

        # 3. Calculate classical a90/95 point
        a90_95 = pod.calculate_reliability_point(X_eval, lower_ci, target_pod=0.90)
        if not np.isnan(a90_95):
            print(f"   -> Classical a90/95 Reliability Index: {a90_95:.3f} mm")

        # 4. Store Results
        self.linear_pod_results = {
            "poi_col": poi_col,
            "threshold": threshold,
            "xlog": xlog,
            "ylog": ylog,
            "X": X,
            "y": y,
            "X_eval": X_eval,
            "model": model,
            "tau": tau,
            "curves": {
                "mean_response": mean_curve,
                "pod": pod_curve,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci
            }
        }

        print("--- Linear Analysis Complete ---")
        return self.linear_pod_results

#### Visualise Linear Results ####
    def visualise_linear(self, show: bool = True, save_path: str = None) -> None:
        if not self.linear_pod_results:
            print("No linear PoD results found. Please run .linear_pod() first.")
            return

        res = self.linear_pod_results

        self.linear_plots["signal_model"] = plot_linear_signal_model(
            X=res["X"], y=res["y"], X_eval=res["X_eval"], model=res["model"],
            threshold=res["threshold"], tau=res["tau"], xlog=res["xlog"],
            ylog=res["ylog"], poi_name=res["poi_col"]
        )

        # Now pass the calculated bounds into the plot!
        self.linear_plots["pod_curve"] = plot_pod_curve(
            X_eval=res["X_eval"].flatten(),
            pod_curve=res["curves"]["pod"],
            ci_lower=res["curves"]["ci_lower"],
            ci_upper=res["curves"]["ci_upper"],
            target_pod=0.90,
            poi_name=res["poi_col"]
        )

        if save_path:
            self.linear_plots["signal_model"].get_figure().savefig(f"{save_path}_linear_signal.png")
            self.linear_plots["pod_curve"].get_figure().savefig(f"{save_path}_linear_pod.png")
            print(f"Plots saved to {save_path}_linear_*.png")

        if show:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except ImportError:
                pass
