import pandas as pd
from typing import List, Dict, Any

from .diagnostics import validate_simulation, sample_sufficiency, ValidationError
from .adaptive import generate_targeted_samples
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

    #### Adding Data ####
    def add_data(self, df: pd.DataFrame) -> None:
        """
        Ingests raw simulation data and updates the internal data state.

        This method appends the provided DataFrame to `self.data`. Because this
        changes the underlying dataset, all downstream results (clean_data,
        sufficiency_results, and pod_results) are reset to empty states.

        Args:
            df (pd.DataFrame): The DataFrame to ingest. It should contain
                the columns specified in `self.inputs` and `self.outcome`.

        Returns:
            None: Updates the internal `self.data` attribute in place.
        """
        if self.data.empty:
            self.data = df.copy()
        else:
            self.data = pd.concat([self.data, df], ignore_index=True)
        print(f"Data updated. Total rows: {len(self.data)}")

        # RESET State: Data changed, so previous valid/diagnostic data is stale
        self.clean_data = pd.DataFrame()
        self.sufficiency_results = pd.DataFrame()
        self.pod_results = {}

    #### Validating self.data ####
    def validate(self) -> None:
        """
        Cleans and validates the raw data stored in `self.data`.

        This method filters the raw data based on project-specific rules. It
        populates `self.clean_data` with valid rows and `self.removed_data`
        with invalid ones.

        Side Effects:
            Updates `self.clean_data` and `self.removed_data`.
            Resets `self.clean_data` to empty if validation fails critically.
        """
        print("Running validation...")
        try:
            clean, removed = validate_simulation(self.data, self.inputs, self.outcome)
            self.clean_data = clean
            self.removed_data = removed
            print(f"Validation passed. {len(clean)} valid rows ready.")
            if not removed.empty:
                print(f"Warning: {len(removed)} invalid rows were dropped.")
        except ValidationError as e:
            print(f"Validation FAILED: {e}")
            self.clean_data = pd.DataFrame()

    #### Checking Sample Sufficiency of self.clean_data ####
    def diagnose(self) -> pd.DataFrame:
        """
        Runs statistical diagnostics to evaluate if the current sample size is sufficient.

        If `self.clean_data` is empty, this method will automatically attempt
        to run `self.validate()` before proceeding.

        Returns:
            pd.DataFrame: A summary of diagnostic metrics. Also updates
                the internal `self.sufficiency_results` attribute.
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

    #### Find new samples based on results in self.sufficiency_results and self.clean_data ####
    def refine(self, n_points: int = 10) -> pd.DataFrame:
        """
        Identifies gaps in the design space and suggests new simulation points.

        Uses an Active Learning approach based on `self.clean_data`. If no
        clean data exists, it triggers `self.validate()`.

        Args:
            n_points (int): Number of new samples to suggest per failed region.

        Returns:
            pd.DataFrame: A table of suggested input coordinates for the next
                iteration of simulations. Does not modify internal data.
        """
        if self.clean_data.empty:
            print("No clean data available. Running validation...")
            self.validate()

            if self.clean_data.empty:
                print("Cannot generate samples: No valid data found.")
                return pd.DataFrame()

        new_samples = generate_targeted_samples(
            df=self.clean_data,
            input_cols=self.inputs,
            outcome_col=self.outcome,
            n_new_per_fix=n_points
        )

        return new_samples

    #### Perform Generalized PoD Analysis ####
    def pod(
        self,
        poi_col: str,
        threshold: float,
        bandwidth_ratio: float = 0.1,
        n_boot: int = 1000
    ) -> Dict[str, Any]:
        """
        Runs the full 'Generalized â vs a' pipeline.

        1. Fits Robust Mean Model (Polynomial Selection).
        2. Fits Variance Model (Kernel Smoothing).
        3. Infers Statistical Distribution (AIC Selection).
        4. Calculates PoD Curve.
        5. Calculates 95% Confidence Bounds (Bootstrap).

        Args:
            poi_col (str): The 'Parameter of Interest' column name (e.g., 'Crack Length'). Must be one of the input columns.
            threshold (float): The detection threshold (e.g., 4.0 dB).
            bandwidth_ratio (float): Smoothing window size as a fraction of data range (Default 0.1).
            n_boot (int): Number of bootstrap iterations for confidence bounds (Default 1000).

        Returns:
            Dict: A dictionary containing all curves and models needed for plotting/reporting.
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

        # 2. Fit Mean Model (Robust Polynomial)
        print("1. Selecting Mean Model (Cross-Validation)...")
        mean_model = pod.fit_robust_mean_model(X, y)
        print(f"   -> Selected Polynomial Degree: {mean_model.best_degree_}")

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
            mean_model.best_degree_, bandwidth, (dist_name, dist_params),
            n_boot=n_boot
        )

        # 7. Package Results
        self.pod_results = {
            "poi_col": poi_col,
            "threshold": threshold,
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
