import pandas as pd
from typing import List, Dict, Any, Tuple

from .diagnostics import validate_simulation, sample_sufficiency, ValidationError
from .adaptive import generate_targeted_samples, run_adaptive_search
from .plotting import plot_signal_model, plot_pod_curve
from . import pod

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

    Examples
    --------
    ```python
    from digiqual.core import SimulationStudy
    study = SimulationStudy(
        input_cols=['Length', 'Angle'],
        outcome_col='Signal'
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
        self.plots: Dict[str, Any] = {}

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

        Examples
        --------
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

    #### Validating self.data ####
    def validate(self) -> None:
        """
        Cleans and validates the raw data stored in `self.data`.

        Populates `self.clean_data` with valid rows and `self.removed_data`
        with invalid ones (e.g., NaNs, negative signals, wrong types).

        Examples
        --------
        ```python
        study.validate()
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

        Checks for Input Coverage (Gaps), Model Fit Stability, and Bootstrap Convergence.

        Returns:
            pd.DataFrame: A summary of diagnostic metrics.

        Examples
        --------
        ```python
        report = study.diagnose()
        print(report[['Test', 'Pass']])
        ```

        """
        if self.clean_data.empty:
            if self.data.empty:
                print("No data found. Please run add_data() first.")
                return pd.DataFrame()

            self.validate()

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

        Examples
        --------
        ```python
        # If diagnostics fail, ask for 10 new points to fix it
        new_samples = study.refine(n_points=10)
        print(new_samples.head())
        ```
        """

        if self.clean_data.empty:
            print("No clean data available. Running validation...")
            self.validate()

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
            input_file (str): Temp input filename.
            output_file (str): Temp output filename.

        Examples
        --------
        ```python
        # 1. Define the variable ranges
        ranges = {"Length": (0, 10), "Angle": (-45, 45)}

        study = SimulationStudy(input_cols=["Length", "Angle"], outcome_col="Signal")

        # 2. Define a "solver" command.
        # We use 'python -c' to simulate an external tool (like Ansys/Abaqus)
        # that reads {input}, does math, and saves to {output}.
        cmd = (
        "python -c "
        "'import pandas as pd; "
        'df=pd.read_csv("{input}"); '
        'df["Signal"] = df["Length"]*2; '
        'df.to_csv("{output}", index=False)'
        "'"
        )

        # 3. Run the automated loop
        study.optimise(
        command=cmd,
        ranges=ranges,
        max_iter=3
        )

        # 4. View the results
        _ = study.pod()
        study.visualise()
        ```
        """
        # 1. Delegate to the Agnostic Engine
        final_data = run_adaptive_search(
            command=command,
            input_cols=self.inputs,       # Pass List[str]
            outcome_col=self.outcome,     # Pass str
            ranges=ranges,
            existing_data=self.data,      # Pass DataFrame (State)
            n_start=n_start,
            n_step=n_step,
            max_iter=max_iter,
            input_file=input_file,
            output_file=output_file
        )

        # 2. Update Class State with the result
        self.data = pd.DataFrame() # Clear old state to avoid duplication
        self.add_data(final_data)

#### PoD Analysis ####
    def pod(
        self,
        poi_col: str,
        threshold: float,
        bandwidth_ratio: float = 0.1,
        n_boot: int = 1000
    ) -> Dict[str, Any]:
        """
        Runs the generalized Probability of Detection (PoD) analysis.

        Args:
            poi_col (str): The parameter of interest (e.g., 'Crack Length').
            threshold (float): The failure threshold (e.g., 4.0 dB).
            bandwidth_ratio (float): Smoothing bandwidth fraction (default 0.1).
            n_boot (int): Bootstrap iterations for confidence bounds.

        Returns:
            Dict: Dictionary containing models, curves, and fit statistics.

        Examples
        --------
        ```python
        results = study.pod(poi_col="Length", threshold=0.5)
        print(results['dist_info'])
        ```
        """
        # 0. Safety Checks
        if self.clean_data.empty:
            self.validate()
            if self.clean_data.empty:
                raise ValueError("Cannot run PoD analysis: No valid data available.")

        if poi_col not in self.clean_data.columns:
            raise ValueError(f"Parameter of Interest '{poi_col}' not found in data columns.")

        # 1. Prepare Data Vectors
        print(f"--- Starting Reliability Analysis (PoI: {poi_col}) ---")
        X = self.clean_data[poi_col].values
        y = self.clean_data[self.outcome].values

        # 2. Fit Mean Model (Robust Regression)
        print("1. Selecting Mean Model (Cross-Validation)...")
        mean_model = pod.fit_robust_mean_model(X, y)
        if mean_model.model_type_ == 'Polynomial':
            print(f"-> Selected Model: Polynomial (Degree {mean_model.model_params_})")
        else:
            print("-> Selected Model: Kriging (Gaussian Process)")

        # 3. Fit Variance Model & Generate Grid
        print("2. Fitting Variance Model (Kernel Smoothing)...")
        residuals, bandwidth, X_eval = pod.fit_variance_model(
            X, y, mean_model, bandwidth_ratio=bandwidth_ratio
        )
        print(f"   -> Smoothing Bandwidth: {bandwidth:.4f}")

        # 4. Infer Distribution
        print("3. Inferring Error Distribution (AIC)...")
        dist_name, dist_params = pod.infer_best_distribution(residuals, X, bandwidth)
        print(f"   -> Selected Distribution: {dist_name}")

        # 5. Compute PoD Curve
        print("4. Computing PoD Curve...")
        pod_curve, mean_curve = pod.compute_pod_curve(
            X_eval, mean_model, X, residuals, bandwidth, (dist_name, dist_params), threshold
        )

        # 6. Bootstrap Confidence Intervals
        print(f"5. Running Bootstrap ({n_boot} iterations)...")
        lower_ci, upper_ci = pod.bootstrap_pod_ci(
            X, y, X_eval, threshold,
            mean_model.model_type_, mean_model.model_params_, bandwidth, (dist_name, dist_params),
            n_boot=n_boot
        )

        # 7.
        a90_95 = pod.calculate_reliability_point(X_eval, lower_ci, target_pod=0.90)
        print(f"   -> a90/95 Reliability Index: {a90_95:.3f}")

        # 8. Package Results
        self.pod_results = {
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

    #### Visualise Results ####
    def visualise(self, show: bool = True, save_path: str = None) -> None:
        """
        Generates and displays diagnostic plots (Signal Model and PoD Curve).

        Args:
            show (bool): If True, calls plt.show().
            save_path (str, optional): If provided, saves figures to disk (e.g., 'results/run1'). Appends '_signal.png' and '_pod.png'.

        Examples
        --------
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
        poi_label = res.get("poi_col", "Parameter of Interest")

        local_std = pod.predict_local_std(
            res["X"], res["residuals"], res["X_eval"], res["bandwidth"]
        )

        # 0. Model Selection Plot (NEW)
        if hasattr(res["mean_model"], "cv_scores_"):
            self.plots["model_selection"] = pod.plot_model_selection(res["mean_model"].cv_scores_)

        # 1. Signal Model Plot
        self.plots["signal_model"] = plot_signal_model(
            X=res["X"],
            y=res["y"],
            X_eval=res["X_eval"],
            mean_curve=res["curves"]["mean_response"],
            threshold=res["threshold"],
            local_std=local_std,
            poi_name=poi_label
        )

        # 2. PoD Curve Plot
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
