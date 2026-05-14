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
        models_cache (Dict): Cached mean models to prevent redundant fitting.
        variance_cache (Dict): Cached variance models (residuals/bandwidth).
        pod_curves_cache (Dict): Cached integrated PoD curves.
        threshold_spectrum_cache (Dict): Cached threshold spectrum.

    Examples:
        ```python
        from digiqual.core import SimulationStudy

        # 1. Clean Initialization
        study = SimulationStudy()

        # 2. Add data (automatically infers input columns)
        study.add_data(df, outcome_col='Signal')

        # 3. Run diagnostics with custom strict thresholds
        results = study.diagnose(max_gap_ratio=0.10, min_r2_score=0.75)
        ```
    """
#### Initialisation ####
    def __init__(self):
        # Internal Data State
        self.inputs: List[str] = []
        self.outcome: str = ""
        self.data: pd.DataFrame = pd.DataFrame()
        self.clean_data: pd.DataFrame = pd.DataFrame()
        self.removed_data: pd.DataFrame = pd.DataFrame()

        # Initialize and clear all caches
        self._clear_caches()

    def _clear_caches(self) -> None:
        """Internal method to wipe all cached mathematical results and state."""
        # Generalized method storage
        self.sufficiency_results: pd.DataFrame = pd.DataFrame()
        self.pod_results: Dict[str, Any] = {}
        self.plots: Dict[str, Any] = {}

        # Linear a-hat vs a method storage
        self.linear_pod_results: Dict[str, Any] = {}
        self.linear_plots: Dict[str, Any] = {}

        # Layered Caches for performance optimization
        self.models_cache: Dict[str, Any] = {}       # Stores fitted mean models
        self.variance_cache: Dict[str, Any] = {}     # Stores residuals and bandwidths
        self.pod_curves_cache: Dict[str, Any] = {}   # Stores integrated PoD curves
        self.threshold_spectrum_cache: Dict[Tuple, Dict] = {} # Stores threshold spectrum

#### Adding Data & Cache Management ####
    def add_data(
        self,
        df: pd.DataFrame,
        outcome_col: str = None,
        input_cols: List[str] = None,
        overwrite: bool = False
    ) -> None:
        """
        Ingests raw simulation data, configures columns, and manages the cache.

        This method sets the outcome and input variables, subsets the data
        accordingly, and automatically clears the mathematical caches whenever
        new data is ingested to prevent state mismatches. If `input_cols` is not
        provided, it smartly infers that all columns other than the outcome are inputs.

        Args:
            df (pd.DataFrame): The DataFrame to ingest.
            outcome_col (str, optional): The name of the target/outcome variable.
                Required on first ingestion, optional when appending.
            input_cols (List[str], optional): Explicit list of input variables.
                Defaults to None (infers all non-outcome columns).
            overwrite (bool, optional): If True, replaces existing data.
                If False, appends to existing data. Defaults to False.
        """
        # 1. Configure Columns (Only if initializing or overwriting)
        if self.data.empty or overwrite:
            if outcome_col is None:
                raise ValueError("You must provide 'outcome_col' when initializing or overwriting data.")
            self.outcome = outcome_col

            if input_cols is None:
                # Smart Inference: Assume all other columns in the DataFrame are inputs
                self.inputs = [col for col in df.columns if col != outcome_col]
            else:
                self.inputs = input_cols
        else:
            # If appending, ignore new column definitions and strictly use the established ones
            if outcome_col is not None and outcome_col != self.outcome:
                print(f"Note: Using established outcome '{self.outcome}' (ignoring '{outcome_col}').")

        required_cols = self.inputs + [self.outcome]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"New data is missing required columns: {missing}")

        # This cleanly strips away any extra helper metadata (like 'Refinement_Reason')
        df_subset = df[required_cols].copy()

        # 2. Append or Overwrite Data
        if self.data.empty or overwrite:
            self.data = df_subset
            print(f"Data initialized/overwritten. Total rows: {len(self.data)}")
        else:
            self.data = pd.concat([self.data, df_subset], ignore_index=True)
            print(f"Data appended. Total rows: {len(self.data)}")

        # 3. Reset validation state
        self.clean_data = pd.DataFrame()
        self.removed_data = pd.DataFrame()

        # 4. CRITICAL: Data changed, math is invalid. Wipe the caches.
        self._clear_caches()


#### UI Helper Methods ####
    def get_unassigned_parameters(self, poi_cols: List[str], nuisance_cols: List[str] = None) -> List[str]:
        """
        Calculates which input variables are not currently assigned as a Parameter
        of Interest or a Nuisance parameter.

        Args:
            poi_cols (List[str]): Currently selected Parameters of Interest.
            nuisance_cols (List[str], optional): Currently selected Nuisance parameters. Defaults to None.

        Returns:
            List[str]: A list of unassigned column names.
        """
        nuisance_cols = nuisance_cols or []
        return [c for c in self.inputs if c not in poi_cols and c not in nuisance_cols]

    def get_data_summary(self, col_name: str) -> Dict[str, float]:
        """
        Calculates the minimum, maximum, and median values for a given column.
        Useful for instantly populating UI slider bounds or default thresholds.

        Args:
            col_name (str): The name of the column to summarize.

        Returns:
            Dict[str, float]: A dictionary containing 'min', 'median', and 'max'.
            Returns None values if the data is empty or the column does not exist.
        """
        if self.data.empty or col_name not in self.data.columns:
            return {"min": None, "median": None, "max": None}

        # Prefer cleaned data if validation has run
        df_target = self.clean_data if not self.clean_data.empty else self.data
        vals = pd.to_numeric(df_target[col_name], errors="coerce").dropna()

        if vals.empty:
            return {"min": None, "median": None, "max": None}

        return {
            "min": float(vals.min()),
            "median": float(vals.median()),
            "max": float(vals.max())
        }

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
    def diagnose(
        self,
        max_gap_ratio: float = 0.20,
        min_r2_score: float = 0.50,
        max_avg_cv: float = 0.15,
        max_max_cv: float = 0.30
    ) -> pd.DataFrame:
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

        self.sufficiency_results = sample_sufficiency(
            self.clean_data, self.inputs, self.outcome,
            skip_validation=True,
            max_gap_ratio=max_gap_ratio,
            min_r2_score=min_r2_score,
            max_avg_cv=max_avg_cv,
            max_max_cv=max_max_cv
        )
        return self.sufficiency_results

#### Adaptive Refinement ####
    def refine(self, n_points: int = 10,
        max_gap_ratio: float = 0.20,
        min_r2_score: float = 0.50,
        max_avg_cv: float = 0.15,
        max_max_cv: float = 0.30) -> pd.DataFrame:
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
            max_gap_ratio=max_gap_ratio,
            min_r2_score=min_r2_score,
            max_avg_cv=max_avg_cv,
            max_max_cv=max_max_cv
        )

        return new_samples

#### Automated Optimisation ####
    def optimise(
        self,
        executor: Union[Executor, str],
        ranges: Dict[str, Tuple[float, float]],
        outcome_col: str = None,
        n_start: int = 20,
        n_step: int = 10,
        max_iter: int = 5,
        max_hours: float = None,
        max_gap_ratio: float = 0.20,
        min_r2_score: float = 0.50,
        max_avg_cv: float = 0.15,
        max_max_cv: float = 0.30
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

        # --- NEW: Cold Start Configuration ---
        if not self.outcome:  # If the study hasn't been configured yet
            if outcome_col is None:
                raise ValueError("When running optimise() from a cold start, you must provide an 'outcome_col'.")
            self.outcome = outcome_col
            self.inputs = list(ranges.keys())
        elif outcome_col is not None and outcome_col != self.outcome:
            print(f"Note: Using established outcome '{self.outcome}' (ignoring '{outcome_col}').")

        # --- SAFEGUARD: Validate input ranges ---
        expected_inputs = set(self.inputs)
        provided_ranges = set(ranges.keys())

        if expected_inputs != provided_ranges:
            raise ValueError(
                f"Variable Mismatch! The keys in your 'ranges' dictionary {list(provided_ranges)} "
                f"do not match the 'input_cols' {list(expected_inputs)} defined in the SimulationStudy."
            )

        # 1. Delegate to the Agnostic Engine
        final_data = run_adaptive_search(
            executor=executor,
            input_cols=self.inputs,
            outcome_col=self.outcome,
            ranges=ranges,
            existing_data=self.data,
            n_start=n_start,
            n_step=n_step,
            max_iter=max_iter,
            max_hours=max_hours,
            max_gap_ratio=max_gap_ratio,
            min_r2_score=min_r2_score,
            max_avg_cv=max_avg_cv,
            max_max_cv=max_max_cv
        )

        # 2. Update Class State with the result
        self.data = pd.DataFrame() # Clear old state to avoid duplication
        self.add_data(final_data, outcome_col=self.outcome, input_cols=self.inputs)


    def compute_pod_spectrum(
        self,
        poi_col: list | str,
        nuisance_col: list | str | None = None,
        slice_values: dict | None = None,
        n_threshold_points: int = 100,
        bandwidth_ratio: float = 0.1,
        model_override: str = "auto",
        force_degree: int | None = None
    ) -> Dict[str, Any]:
        """Pre-calculates a spectrum of PoD curves across a range of signal thresholds."""
        # 1. Standardize column configurations
        if isinstance(poi_col, str):
            poi_cols = [poi_col]
        else:
            poi_cols = poi_col

        nuisance_cols = [nuisance_col] if isinstance(nuisance_col, str) else (nuisance_col or [])
        slice_values = slice_values or {}

        # 2. Establish baseline model and variance
        print("--- Initiating Threshold Spectrum Generation ---")
        median_thresh = float(self.clean_data[self.outcome].median())
        temp_results = self.pod(
            poi_col=poi_cols,
            threshold=median_thresh,
            nuisance_col=nuisance_cols,
            slice_values=slice_values,
            bandwidth_ratio=bandwidth_ratio,
            n_boot=0,
            model_override=model_override,
            force_degree=force_degree
        )

        # Extract the key for cache indexing
        mean_model = temp_results['mean_model']
        selected_key = ('Polynomial', mean_model.model_params_) if mean_model.model_type_ == 'Polynomial' else ('Kriging', None)

        # Create the Spectrum Key and Layer 3 Key to access our matrices
        spectrum_key = (selected_key, tuple(poi_cols), tuple(nuisance_cols), frozenset(temp_results['slice_values'].items()))
        l3_key = (selected_key, median_thresh, tuple(poi_cols), tuple(nuisance_cols), frozenset(temp_results['slice_values'].items()))

        # 3. Check if this exact spectrum configuration is already cached
        if spectrum_key in self.threshold_spectrum_cache:
            print("4. Threshold Spectrum already cached (Layer 4 Hit).")
            return self.threshold_spectrum_cache[spectrum_key]

        print(f"4. Computing PoD Spectrum for {n_threshold_points} thresholds (Layer 4 Cache Miss)...")

        # 4. Generate Threshold Vector across the observed signal range
        y_min, y_max = float(self.clean_data[self.outcome].min()), float(self.clean_data[self.outcome].max())
        thresh_vec = np.linspace(y_min, y_max, n_threshold_points)

        # --- THE FIX: Pull matrices directly from Layer 2 and Layer 3 Caches ---
        l3_cache = self.pod_curves_cache[l3_key]

        from .integration import compute_multi_dim_pod
        pod_matrix, mean_curve = compute_multi_dim_pod(
            poi_grid=l3_cache['X_eval'],
            nuisance_ranges=l3_cache['nuisance_ranges'],
            model=mean_model,
            X_train=temp_results['X'],
            residuals=temp_results['residuals'],
            bandwidth=temp_results['bandwidth'],
            dist_info=temp_results['dist_info'],
            thresholds=thresh_vec,
            feature_names=self.inputs, # <-- ADD THIS
            poi_names=poi_cols         # <-- ADD THIS
        )

        # 6. Package and store in Layer 4
        spectrum_data = {
            "thresholds": thresh_vec,
            "pod_matrix": pod_matrix,
            "mean_curve": mean_curve
        }

        self.threshold_spectrum_cache[spectrum_key] = spectrum_data
        print("--- Spectrum Generation Complete ---")

        return spectrum_data



#### Time Heuristic ####
    def estimate_compute_time(self, model_type: str, n_boot: int, n_nuisances: int, n_jobs: int) -> float:
        """
        Physics-aware heuristic to estimate PoD computation time in seconds.
        Accounts for initial cache-building (CV), Kriging complexity, and MC Integration.
        """
        n_samples = len(self.clean_data) if not self.clean_data.empty else len(self.data)
        if n_samples == 0:
            return 0.0

        # 1. Base Fit Time (Per Iteration)
        t_poly = 0.001 * (n_samples / 100.0)
        t_kriging = (n_samples / 500.0) ** 3 * 1.5  # Kriging matrix inversion is heavy!

        # 2. Evaluate Caching State
        if n_boot == 0 and not self.models_cache:
            # INITIAL CACHE BUILD: 10 polys * 10 folds = 100 fits. Kriging = ~12 heavy fits.
            # Plus Leave-One-Out CV for the variance smoothing bandwidth (~2 seconds)
            if n_samples > 1000:
                t_fit = (100 * t_poly) + 2.0 # Kriging is skipped automatically for N>1000
            else:
                t_fit = (100 * t_poly) + (12 * t_kriging) + 2.0
        elif model_type.lower() == "kriging":
            t_fit = t_kriging
        else:
            t_fit = t_poly

        # 3. Integration Time (Per Iteration)
        if n_nuisances > 0:
            # SLOW PATH: 100 grid points * MC Samples
            t_int = 0.6 if model_type.lower() == "kriging" else 0.08
        else:
            # FAST PATH: Single Vectorized Matrix Operation
            t_int = 0.02 if model_type.lower() == "kriging" else 0.002

        t_single_iteration = t_fit + t_int

        # 4. Core scaling for Bootstrap
        if n_jobs == -1:
            import os
            cores = max((os.cpu_count() or 1) - 1, 1)
        else:
            cores = max(n_jobs or 1, 1)

        # 5. Total Time
        if n_boot == 0:
            total_time = t_fit + t_int + 0.5  # Initial fit or slider slice update
        else:
            total_time = ((t_single_iteration * n_boot) / cores) + 0.5  # Heavy UQ bootstrap

        return total_time


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
        print(f"--- Starting Reliability Analysis (PoIs: {poi_cols} - Nuisance: {nuisance_cols}) ---")
        if final_slice_values:
            print(f"-> Slicing surface at: {final_slice_values}")

        X = self.clean_data[all_cols].values
        y = self.clean_data[self.outcome].values

        # ---------------------------------------------------------
        # 4. LAYER 1 CACHE: Mean Models
        # ---------------------------------------------------------
        if not self.models_cache:
            print("1. Training all surrogate models (Cache Miss)...")
            # Fit everything and save to caches
            models, scores, winner = pod.fit_all_robust_mean_models(X, y)
            self.models_cache = models
            self.cv_scores_cache = scores
            self.cv_winner_key = winner
        else:
            print("1. Loading surrogate models from cache (Cache Hit)...")

        # Select the specific model requested by the user
        if model_override == "polynomial" and force_degree is not None:
            selected_key = ('Polynomial', force_degree)
            if selected_key not in self.models_cache:
                raise ValueError(f"Polynomial degree {force_degree} was requested, but it is not available in the cache.")

        elif model_override == "polynomial":
            # User wants Poly but didn't force a degree -> Pick the best Poly
            poly_scores = {k: v for k, v in self.cv_scores_cache.items() if k[0] == 'Polynomial'}
            selected_key = min(poly_scores, key=poly_scores.get)

        elif model_override == "kriging":
            selected_key = ('Kriging', None)
            if selected_key not in self.models_cache:
                raise ValueError(
                    "Kriging model was requested but is not available in the cache. "
                    "This usually occurs if your dataset exceeds 1,000 samples, "
                    "as Kriging evaluation is automatically skipped to prevent timeouts."
                )
        else:
            selected_key = self.cv_winner_key

        # Retrieve the selected model!
        mean_model = self.models_cache[selected_key]

        # Attach the CV data to the model so plotting functions can still read it
        mean_model.cv_scores_ = self.cv_scores_cache
        mean_model.cv_winner_ = self.cv_winner_key
        mean_model.forced_model_ = (selected_key != self.cv_winner_key)

        equation = pod.generate_latex_equation(mean_model, all_cols, self.outcome)

        if mean_model.model_type_ == 'Polynomial':
            print(f"-> Selected Model: Polynomial (Degree {mean_model.model_params_})")
        else:
            print("-> Selected Model: Kriging (Gaussian Process)")

        # ---------------------------------------------------------
        # 5. LAYER 2 CACHE: Variance Model, Distribution & Sobol
        # ---------------------------------------------------------
        if selected_key not in self.variance_cache:
            print("2. Fitting Variance Model & Inferring Distribution (Cache Miss)...")

            residuals, bandwidth = pod.fit_variance_model(
                X, y, mean_model, bandwidth_ratio=bandwidth_ratio
            )
            dist_name, dist_params = pod.infer_best_distribution(residuals, X, bandwidth)

            # Calculate Sobol Sensitivity Indices (Cache Miss)
            print("-> Calculating Total-Order Sobol Indices...")
            from digiqual.pod import calculate_sobol_indices
            sobol_indices = calculate_sobol_indices(
                mean_model=mean_model,
                feature_names=all_cols,
                data_df=self.clean_data
            )

            # Save the heavy lifting to the cache
            self.variance_cache[selected_key] = {
                "residuals": residuals,
                "bandwidth": bandwidth,
                "dist_info": (dist_name, dist_params),
                "sobol_indices": sobol_indices  # <--- Now safely cached!
            }
        else:
            print("2. Loading Variance Model, Distribution & Sobol from cache (Cache Hit)...")
            cached_var = self.variance_cache[selected_key]
            residuals = cached_var["residuals"]
            bandwidth = cached_var["bandwidth"]
            dist_name, dist_params = cached_var["dist_info"]
            sobol_indices = cached_var.get("sobol_indices", None)  # <--- Instantly retrieved!

        print(f"   -> Smoothing Bandwidth: {bandwidth:.4f}")
        print(f"   -> Selected Distribution: {dist_name}")

        # ---------------------------------------------------------
        # 6. LAYER 3 & 4 CACHE: Integration & Spectrum Interpolation
        # ---------------------------------------------------------
        # Define keys for the different caching layers
        spectrum_key = (selected_key, tuple(poi_cols), tuple(nuisance_cols), frozenset(final_slice_values.items()))
        l3_key = (selected_key, threshold, tuple(poi_cols), tuple(nuisance_cols), frozenset(final_slice_values.items()))

        # A) Setup Evaluation Grid & Nuisance Parameters (Lightweight Setup)
        poi_grids = []
        for col in poi_cols:
            num_points = 100 if len(poi_cols) == 1 else 30
            poi_grids.append(np.linspace(self.clean_data[col].min(), self.clean_data[col].max(), num_points))

        if len(poi_cols) == 1:
            X_eval = poi_grids[0].reshape(-1, 1)
        else:
            mesh = np.meshgrid(*poi_grids, indexing='ij')
            X_eval = np.vstack([m.flatten() for m in mesh]).T

        # Build nuisance ranges (including sliced parameters locked as single-value ranges)
        nuisance_ranges = {c: (float(self.clean_data[c].min()), float(self.clean_data[c].max())) for c in nuisance_cols}
        for c, val in final_slice_values.items():
            nuisance_ranges[c] = (val, val)

        # B) Resolve Analysis (Layer 4 -> Layer 3 -> Miss)
        if spectrum_key in self.threshold_spectrum_cache and n_boot == 0:
            print("3. Interpolating PoD Curve from Threshold Spectrum (Layer 4 Hit)...")
            spec = self.threshold_spectrum_cache[spectrum_key]

            # Linear Interpolation across the pre-calculated threshold spectrum
            # pod_matrix shape: (n_grid_points, n_thresholds)
            pod_curve = np.array([
                np.interp(threshold, spec["thresholds"], spec["pod_matrix"][i, :])
                for i in range(len(X_eval))
            ])
            mean_curve = spec["mean_curve"]

            # Calculate Preliminary Reliability Point (1D only)
            a90_95 = np.nan
            if len(poi_cols) == 1:
                a90_95 = pod.calculate_reliability_point(X_eval.flatten(), pod_curve, target_pod=0.90)

        elif l3_key in self.pod_curves_cache:
            print("3. Loading PoD Curve from Individual Cache (Layer 3 Hit)...")
            cached_l3 = self.pod_curves_cache[l3_key]
            pod_curve = cached_l3["pod_curve"]
            mean_curve = cached_l3["mean_curve"]
            a90_95 = cached_l3["a90_95"]

        else:
            print("3. Integrating PoD Curve from Scratch (Cache Miss)...")
            from .integration import compute_multi_dim_pod
            pod_curve, mean_curve = compute_multi_dim_pod(
                X_eval, nuisance_ranges, mean_model, X, residuals, bandwidth, (dist_name, dist_params), threshold,
            feature_names=all_cols, poi_names=poi_cols
            )

            # Calculate Preliminary Reliability Point (1D only)
            a90_95 = np.nan
            if len(poi_cols) == 1:
                a90_95 = pod.calculate_reliability_point(X_eval.flatten(), pod_curve, target_pod=0.90)

            # Store in Layer 3 Cache for fast slicing/parameter swapping
            self.pod_curves_cache[l3_key] = {
                "X_eval": X_eval,
                "poi_grids": poi_grids,
                "nuisance_ranges": nuisance_ranges,
                "pod_curve": pod_curve,
                "mean_curve": mean_curve,
                "a90_95": a90_95
            }


        # ---------------------------------------------------------
        # 7. Bootstrap Confidence Intervals (Parallelized)
        # ---------------------------------------------------------
        # We don't cache the bootstrap because n_boot can change, and it's heavily intentional when run.
        if n_boot > 0:
            if n_jobs is None or n_jobs == 1:
                actual_cores = 1
            elif n_jobs == -1:
                actual_cores = max((os.cpu_count() or 1) - 1, 1)
            else:
                actual_cores = n_jobs

            print(f"4. Running Bootstrap ({n_boot} iterations on {actual_cores} cores)...", flush=True)

            lower_ci, upper_ci = pod.bootstrap_pod_ci(
                X, y, X_eval, threshold,
                mean_model.model_type_, mean_model.model_params_, bandwidth, (dist_name, dist_params),
                n_boot=n_boot, nuisance_ranges=nuisance_ranges,
                n_jobs=n_jobs, feature_names=all_cols, poi_names=poi_cols
            )

            # Recalculate true a90/95 based on the LOWER confidence bound
            if len(poi_cols) == 1 and lower_ci is not None:
                a90_95 = pod.calculate_reliability_point(X_eval.flatten(), lower_ci, target_pod=0.90)
                print(f"   -> a90/95 Reliability Index: {a90_95:.3f}")

        else:
            print("4. Skipping Bootstrap (n_boot=0)...", flush=True)
            lower_ci, upper_ci = None, None

        # 8. Package Results
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
            "sobol_indices": sobol_indices,
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
        Updates the evaluated slice for the PoD surface using Layer 3 Caching.
        This allows for real-time plot updates when changing constant parameters in the UI.
        """
        if not self.pod_results:
            return {}

        # Just re-call .pod() with n_boot=0!
        # Because we built the Layer 1, 2, and 3 caches, this call will resolve in
        # microseconds if it's cached, or only run the lightweight Layer 3 if it's a new slice.

        return self.pod(
            poi_col=self.pod_results["poi_cols"],
            threshold=self.pod_results["threshold"],
            nuisance_col=self.pod_results["nuisance_cols"],
            slice_values=slice_values,
            n_boot=0, # Never run bootstrap during a slider drag!

            # Force it to use the exact same model we are currently looking at
            model_override="polynomial" if self.pod_results["mean_model"].model_type_ == "Polynomial" else "kriging",
            force_degree=self.pod_results["mean_model"].model_params_ if self.pod_results["mean_model"].model_type_ == "Polynomial" else None
        )

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

        import matplotlib.pyplot as plt

        # --- THE FIX: Safely close ONLY DigiQual's previous figures to free memory ---
        for plot_name, item in self.plots.items():
            try:
                # Check if the item is an Axes (needs .get_figure()) or a Figure itself
                fig = item.get_figure() if hasattr(item, 'get_figure') else item
                plt.close(fig)
            except Exception:
                pass
        self.plots.clear()

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
            from .plotting import plot_signal_surface
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
        if len(poi_cols) == 1:
            self.plots["pod_curve"] = plot_pod_curve(
                X_eval=res["X_eval"].flatten(),
                pod_curve=res["curves"]["pod"],
                ci_lower=res["curves"]["ci_lower"], # plotting.py handles None gracefully!
                ci_upper=res["curves"]["ci_upper"],
                target_pod=0.90,
                poi_name=poi_cols[0]
            )
        else:
            from .plotting import plot_pod_surface
            self.plots["pod_curve"] = plot_pod_surface(
                poi_grids=res["poi_grids"],
                pod_curve=res["curves"]["pod"],
                poi_names=poi_cols,
                ci_lower=res["curves"]["ci_lower"] # <-- ADD THIS
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
