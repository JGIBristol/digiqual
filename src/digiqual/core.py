import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .diagnostics import validate_simulation, sample_sufficiency, ValidationError
from .adaptive import generate_targeted_samples, run_adaptive_search
from .plotting import plot_signal_model, plot_pod_curve
from . import pod
from . import integration

class SimulationStudy:
    """
    A workflow manager for simulation-based reliability assessment.

    This class serves as the main API for the digiqual package. It handles data
    ingestion, validation, active learning (adaptive refinement), and coordinates
    both 1D and Multi-Dimensional Probability of Detection (PoD) analyses.

    Attributes:
        inputs (List[str]): List of input variable names defined by the user.
        outcome (str): Name of the outcome variable (signal response).
        data (pd.DataFrame): The raw, unvalidated simulation data.
        clean_data (pd.DataFrame): Data that has passed all numeric validation checks.
        removed_data (pd.DataFrame): Rows removed during validation (e.g., NaNs).
        sufficiency_results (pd.DataFrame): Diagnostics on sample size and stability.
        pod_results (Dict): Results and models from the latest 1D PoD analysis.
        multi_results (Dict): Results from the latest Multi-Dimensional PoD analysis.
        plots (Dict): Stores the latest generated matplotlib figures.

    Examples:
        ```python
        from digiqual.core import SimulationStudy

        # Initialize the study with your parameter names
        study = SimulationStudy(
            input_cols=['Crack_Length', 'Defect_Angle'],
            outcome_col='Ultrasonic_Signal'
        )
        ```
    """

    #### Initialisation ####
    def __init__(
        self,
        input_cols: List[str],
        outcome_col: str
    ):
        self.inputs = input_cols
        self.outcome = outcome_col

        # Internal state
        self.data: pd.DataFrame = pd.DataFrame()
        self.clean_data: pd.DataFrame = pd.DataFrame()
        self.removed_data: pd.DataFrame = pd.DataFrame()
        self.sufficiency_results: pd.DataFrame = pd.DataFrame()
        self.pod_results: Dict[str, Any] = {}
        self.multi_results: Dict[str, Any] = {}
        self.plots: Dict[str, Any] = {}

    #### Adding Data ####
    def add_data(self, df: pd.DataFrame) -> None:
        """
        Ingests raw simulation data, filtering for relevant columns only.

        This method strips away any columns in the provided dataframe that were not
        defined in `self.inputs` or `self.outcome` during initialization. It also
        resets the internal analysis state so old results aren't accidentally plotted.

        Args:
            df (pd.DataFrame): The DataFrame to ingest.

        Raises:
            ValueError: If the new data is missing any required input or outcome columns.

        Examples:
            ```python
            import pandas as pd
            df = pd.read_csv("simulation_results.csv")
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
        self.multi_results = {}
        self.plots = {}

    #### Validating self.data ####
    def _validate(self) -> None:
        """
        Cleans and validates the raw data stored in `self.data`.

        Populates `self.clean_data` with valid numeric rows and `self.removed_data`
        with invalid ones (e.g., NaNs, text values). Automatically called before analysis.
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

        Checks for Input Coverage (Gaps), Model Fit Stability, and Bootstrap Convergence.

        Returns:
            pd.DataFrame: A summary table of diagnostic metrics and pass/fail status.

        Examples:
            ```python
            report = study.diagnose()
            print(report[['Test', 'Pass']])
            ```
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
            self.clean_data, self.inputs, self.outcome
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
            n_new_per_fix=n_points
        )

        return new_samples

    #### Automated Optimisation ####
    def optimise(
        self,
        command: str,
        ranges: Dict[str, Tuple[float, float]],
        n_start: int = 20,
        n_step: int = 10,
        max_iter: int = 5,
        max_hours: float = None,
        input_file: str = "sim_input.csv",
        output_file: str = "sim_output.csv"
    ) -> None:
        """
        Runs the automated Active Learning loop (Initialize -> Execute -> Diagnose -> Refine).

        Args:
            command (str): Solver command (e.g. "python solver.py {input} {output}").
            ranges (Dict): Input bounds, e.g. {"Length": (0, 10)}.
            n_start (int): Initial sample size (only if data is empty).
            n_step (int): Batch size for refinement.
            max_iter (int): Max refinement loops.
            max_hours (float, optional): Physical time limit in hours to safely stop the loop.
            input_file (str): Temp input filename.
            output_file (str): Temp output filename.
        """
        final_data = run_adaptive_search(
            command=command,
            input_cols=self.inputs,
            outcome_col=self.outcome,
            ranges=ranges,
            existing_data=self.data,
            n_start=n_start,
            n_step=n_step,
            max_iter=max_iter,
            max_hours=max_hours,
            input_file=input_file,
            output_file=output_file
        )

        self.data = pd.DataFrame()
        self.add_data(final_data)

    #### 1D PoD Analysis ####
    def pod(
        self,
        poi_col: str,
        threshold: float,
        bandwidth_ratio: float = 0.1,
        n_boot: int = 1000
    ) -> Dict[str, Any]:
        """
        Runs the baseline 1D Probability of Detection (PoD) analysis.

        This isolates a single Parameter of Interest and models its direct relationship
        with the signal. It balances robust mean regression (Kriging/Polynomials) with
        heteroscedastic variance modeling to produce standard a-vs-a reliability curves.

        Args:
            poi_col (str): The parameter of interest (e.g., 'Crack Length').
            threshold (float): The failure threshold for detection.
            bandwidth_ratio (float, optional): Smoothing bandwidth fraction. Defaults to 0.1.
            n_boot (int, optional): Bootstrap iterations for confidence bounds. Defaults to 1000.

        Returns:
            Dict: Dictionary containing models, generated curves, and fit statistics.
            This is also stored internally in `self.pod_results`.

        Examples:
            ```python
            results = study.pod(poi_col="Length", threshold=4.0)
            study.visualise()
            ```
        """
        if self.clean_data.empty:
            self._validate()
            if self.clean_data.empty:
                raise ValueError("Cannot run PoD analysis: No valid data available.")

        if poi_col not in self.clean_data.columns:
            raise ValueError(f"Parameter of Interest '{poi_col}' not found in data columns.")

        print(f"--- Starting 1D Reliability Analysis (PoI: {poi_col}) ---")
        X = self.clean_data[poi_col].values
        y = self.clean_data[self.outcome].values

        print("1. Selecting Mean Model (Cross-Validation)...")
        mean_model = pod.fit_robust_mean_model(X, y)

        print("2. Fitting Variance Model (Kernel Smoothing)...")
        residuals, bandwidth, X_eval = pod.fit_variance_model(
            X, y, mean_model, bandwidth_ratio=bandwidth_ratio
        )

        print("3. Inferring Error Distribution (AIC)...")
        dist_name, dist_params = pod.infer_best_distribution(residuals, X, bandwidth)

        print("4. Computing PoD Curve...")
        pod_curve, mean_curve = pod.compute_pod_curve(
            X_eval, mean_model, X, residuals, bandwidth, (dist_name, dist_params), threshold
        )

        print(f"5. Running Bootstrap ({n_boot} iterations)...")
        lower_ci, upper_ci = pod.bootstrap_pod_ci(
            X, y, X_eval, threshold,
            mean_model.model_type_, mean_model.model_params_, bandwidth, (dist_name, dist_params),
            n_boot=n_boot
        )

        a90_95 = pod.calculate_reliability_point(X_eval, lower_ci, target_pod=0.90)

        # Clear multi_results if running a fresh 1D pod
        self.multi_results = {}
        self.pod_results = {
            "type": "1D",
            "poi_col": poi_col,
            "threshold": threshold,
            "a90_95": a90_95,
            "X": X,
            "y": y,
            "X_eval": X_eval,
            "mean_model": mean_model,
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

    #### Multi-Dimensional PoD Analysis (NEW) ####
    def multi(
        self,
        poi_cols: List[str],
        nuisance_cols: List[str],
        threshold: float,
        nuisance_dists: Optional[Dict[str, Any]] = None,
        bandwidth_ratio: float = 0.1,
        n_mc_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Runs flexible, multi-dimensional PoD analysis with Monte Carlo integration.

        This method trains an N-dimensional Kriging model to learn the complex signal
        landscape across all parameters. It then marginalizes the `nuisance_cols` by
        simulating thousands of virtual real-world scenarios (defined by `nuisance_dists`)
        and integrating them out to produce a highly calibrated 1D PoD curve for the `poi_col`.

        Args:
            poi_cols (List[str]): The parameters of interest to plot against (e.g., ['Size', 'Angle']). Supports length 1 or 2.
            nuisance_cols (List[str]): Additional variables affecting the signal (e.g., ['Angle']).
            threshold (float): The failure threshold for detection.
            nuisance_dists (Dict[str, Any], optional): Mapping of nuisance columns to
                `scipy.stats` distributions. If omitted, defaults to Uniform bounds.
            bandwidth_ratio (float, optional): Smoothing bandwidth fraction. Defaults to 0.1.
            n_mc_samples (int, optional): Size of the Monte Carlo integration matrix.
                Higher values yield smoother curves. Defaults to 5000.

        Returns:
            Dict: Dictionary containing the n-dimensional model and the marginalized curve.
            This is also stored internally in `self.multi_results`.

        Examples:
            ```python
            import scipy.stats as stats

            # Define how nuisance parameters behave in reality
            dists = {'Defect_Angle': stats.norm(0, 5)}

            # Run the multi-dimensional assessment
            results = study.multi(
                poi_col='Crack_Length',
                nuisance_cols=['Defect_Angle'],
                threshold=4.0,
                nuisance_dists=dists,
                n_mc_samples=10000
            )
            study.visualise()
            ```
        """
        if self.clean_data.empty:
            self._validate()
            if self.clean_data.empty:
                raise ValueError("Cannot run analysis: No valid data available.")

        all_cols = poi_cols + nuisance_cols
        missing = [c for c in all_cols if c not in self.clean_data.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        print("--- Starting Multi-Dimensional Reliability Analysis ---")
        print(f"PoI: {poi_cols} | Nuisance Parameters: {nuisance_cols}")

        # 1. Prepare N-Dimensional Data Matrix
        X_nd = self.clean_data[all_cols].values
        y = self.clean_data[self.outcome].values

        # 2. Fit Models
        print("1. Fitting Multi-Dimensional Mean Model (Cross-Validation)...")
        mean_model = pod.fit_robust_mean_model(X_nd, y)
        if mean_model.model_type_ == 'Polynomial':
            print(f"   -> Selected: Multi-D Polynomial (Degree {mean_model.model_params_})")
        else:
            print("   -> Selected: N-Dimensional Kriging (Gaussian Process)")

        print("2. Fitting Variance Model & Distribution...")
        residuals, bandwidth, _ = pod.fit_variance_model(
            X_nd, y, mean_model, bandwidth_ratio=bandwidth_ratio
        )
        dist_name, dist_params = pod.infer_best_distribution(residuals, X_nd, bandwidth)

        # 3. Monte Carlo Integration
        print(f"3. Running Monte Carlo Integration ({n_mc_samples} samples)...")

        # Grid just for the final 1D/2D marginalized plot
        if len(poi_cols) == 1:
            X_eval_poi = np.linspace(
                self.clean_data[poi_cols[0]].min(),
                self.clean_data[poi_cols[0]].max(),
                100
            )
            X_eval_poi = X_eval_poi.reshape(-1, 1)
        elif len(poi_cols) == 2:
            x1 = np.linspace(self.clean_data[poi_cols[0]].min(), self.clean_data[poi_cols[0]].max(), 30)
            x2 = np.linspace(self.clean_data[poi_cols[1]].min(), self.clean_data[poi_cols[1]].max(), 30)
            X1, X2 = np.meshgrid(x1, x2)
            X_eval_poi = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        else:
            raise ValueError("Only 1 or 2 Parameters of Interest are supported.")

        # Build the virtual reality space
        mc_samples = integration.build_integration_space(
            nuisance_cols=nuisance_cols,
            reference_data=self.clean_data,
            nuisance_dists=nuisance_dists,
            n_mc_samples=n_mc_samples
        )

        # Integrate out the noise
        marginal_pod_curve = integration.compute_marginal_pod(
            X_eval_grid=X_eval_poi,
            mean_model=mean_model,
            bandwidth=bandwidth,
            dist_info=(dist_name, dist_params),
            threshold=threshold,
            mc_samples=mc_samples,
            X_orig=X_nd,
            residuals=residuals
        )

        # Clear pod_results if running a fresh multi pod
        self.pod_results = {}
        self.multi_results = {
            "type": "Multi-D",
            "poi_cols": poi_cols,
            "threshold": threshold,
            "X_eval_poi": X_eval_poi,
            "marginal_pod": marginal_pod_curve,
            "mean_model": mean_model,
            # We save X and y so we can plot the raw scattered data against the curve later
            "X_raw_poi": self.clean_data[poi_cols].values,
            "y_raw": y
        }

        print("--- Multi-Dimensional Analysis Complete ---")
        return self.multi_results


    #### Visualise Results ####
    def visualise(self, show: bool = True, save_path: str = None) -> None:
        """
        Generates and displays diagnostic plots.

        Automatically routes to either 1D or Multi-Dimensional plotting logic
        based on the most recently executed analysis method (`.pod()` or `.multi()`).

        Args:
            show (bool, optional): If True, calls plt.show() to display in the IDE/notebook. Defaults to True.
            save_path (str, optional): Base file path to save figures (e.g., 'results/run1'). Defaults to None.

        Examples:
            ```python
            # Run analysis and immediately show plots
            study.pod("Length", threshold=4.0)
            study.visualise()

            # Run a new multi-dimensional analysis and save plots to disk
            study.multi("Length", ["Angle"], threshold=4.0)
            study.visualise(show=False, save_path='./outputs/multi_study')
            ```
        """
        # Determine which results to use based on which method was run last
        if self.multi_results:
            res = self.multi_results
            is_multi = True
        elif self.pod_results:
            res = self.pod_results
            is_multi = False
        else:
            print("No results found. Please run .pod() or .multi() first.")
            return

        poi_label = res.get("poi_col", "Parameter of Interest")

        # --- MULTI-DIMENSIONAL PLOTTING ---
        if is_multi:
            # 0. Model Selection Plot
            if hasattr(res["mean_model"], "cv_scores_"):
                self.plots["model_selection"] = pod.plot_model_selection(res["mean_model"].cv_scores_)

            if len(res["poi_cols"]) == 1:
                poi_label = res["poi_cols"][0]
                self.plots["signal_model"] = plot_signal_model(
                    X=res["X_raw_poi"][:, 0],
                    y=res["y_raw"],
                    X_eval=res["X_eval_poi"][:, 0],
                    mean_curve=np.full_like(res["X_eval_poi"][:, 0], np.nan),
                    threshold=res["threshold"],
                    local_std=None,
                    poi_name=poi_label
                )
                self.plots["signal_model"].set_title("Raw Signal Cloud (Nuisance Variation)", fontweight='bold')

                self.plots["pod_curve"] = plot_pod_curve(
                    X_eval=res["X_eval_poi"][:, 0],
                    pod_curve=res["marginal_pod"],
                    ci_lower=None,
                    ci_upper=None,
                    target_pod=0.90,
                    poi_name=poi_label
                )
                self.plots["pod_curve"].set_title("Marginalized PoD Curve", fontweight='bold')
            else:
                poi_labels = res["poi_cols"]
                from .plotting import plot_pod_surface
                self.plots["pod_curve"] = plot_pod_surface(
                    X_eval=res["X_eval_poi"],
                    pod_curve=res["marginal_pod"],
                    poi_names=poi_labels
                )

        # --- 1D PLOTTING ---
        else:
            local_std = pod.predict_local_std(
                res["X"], res["residuals"], res["X_eval"], res["bandwidth"]
            )

            if hasattr(res["mean_model"], "cv_scores_"):
                self.plots["model_selection"] = pod.plot_model_selection(res["mean_model"].cv_scores_)

            self.plots["signal_model"] = plot_signal_model(
                X=res["X"],
                y=res["y"],
                X_eval=res["X_eval"],
                mean_curve=res["curves"]["mean_response"],
                threshold=res["threshold"],
                local_std=local_std,
                poi_name=poi_label
            )

            self.plots["pod_curve"] = plot_pod_curve(
                X_eval=res["X_eval"],
                pod_curve=res["curves"]["pod"],
                ci_lower=res["curves"]["ci_lower"],
                ci_upper=res["curves"]["ci_upper"],
                target_pod=0.90,
                poi_name=poi_label
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
