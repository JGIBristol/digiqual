import pandas as pd
from typing import List

from .diagnostics import validate_simulation, sample_sufficiency, ValidationError
from .adaptive import generate_targeted_samples

class SimulationStudy:
    """
    A workflow manager for simulation reliability assessment.

    Attributes:
        inputs (List[str]): List of input variable names.
        outcome (str): Name of the outcome variable.
        data (pd.DataFrame): The raw simulation data.
        clean_data (pd.DataFrame): Data that has passed validation.
        sufficiency_results (pd.DataFrame): The latest diagnostic results.
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

    #### Adding Data ####
    def add_data(self, df: pd.DataFrame) -> None:
        """
        Ingests raw simulation data. Resets downstream results (clean_data, results).
        """
        if self.data.empty:
            self.data = df.copy()
        else:
            self.data = pd.concat([self.data, df], ignore_index=True)
        print(f"Data updated. Total rows: {len(self.data)}")

        # RESET State: Data changed, so previous valid/diagnostic data is stale
        self.clean_data = pd.DataFrame()
        self.sufficiency_results = pd.DataFrame()

    #### Validating self.data ####
    def validate(self) -> None:
        """
        Explicitly runs data cleaning. Useful for debugging dropped rows.
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
        Runs statistical diagnostics on the data.

        Automatically runs validate() first if needed.
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
        Generates new targeted samples to fix diagnostic failures (Active Learning).

        This method inspects the latest diagnostics. If issues are found (e.g. gaps
        or instability), it generates specific points to resolve them.

        Args:
            n_points (int): Number of new samples to generate per detected issue. Defaults to 10.

        Returns:
            pd.DataFrame: A dataframe of recommended simulation parameters. Returns empty if the study is already sufficient.
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
