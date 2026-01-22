import pandas as pd
from typing import List, Dict, Union

def validate_simulation(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str
) -> Dict[str, Union[bool, str, pd.DataFrame, int, None]]:
    """
    Ensures data is numeric and outcome is positive.

    Args:
        df (pd.DataFrame): Raw data.
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Signal column name.

    Returns:
        dict: A dictionary containing:
            - 'valid' (bool): True if data passed validation.
            - 'data' (pd.DataFrame or None): The cleaned dataframe.
            - 'n_dropped' (int): Number of rows removed.
            - 'message' (str): Status message.
    """
    # --- 1. Basic Structure Checks ---
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "valid": False,
            "message": "Input is not a valid data frame.",
            "data": None,
            "n_dropped": 0
        }

    # Check if all columns exist
    # We combine lists to check everything at once
    required_cols = input_cols + [outcome_col]
    if not set(required_cols).issubset(df.columns):
        return {
            "valid": False,
            "message": "Missing columns.",
            "data": None,
            "n_dropped": 0
        }

    # --- NEW: Overlap Check ---
    if outcome_col in input_cols:
        return {
            "valid": False,
            "message": "The Outcome variable cannot also be an Input variable.",
            "data": None,
            "n_dropped": 0
        }

    # Select only relevant columns to create the clean copy
    df_clean = df[required_cols].copy()

    # Count rows before cleaning to calculate drops later
    initial_rows = len(df_clean)

    # --- 2. Numeric Coercion ---
    # 'apply(pd.to_numeric)' works like R's lapply(as.numeric).
    # 'errors="coerce"' turns strings/errors into NaN (comparable to R's NAs with warning)
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

    # Drop NAs created by coercion or existing before (comparable to na.omit)
    df_clean.dropna(inplace=True)

    # --- 3. Outcome Validation (Signal) ---
    # Signals usually MUST be positive (> 0)
    df_clean = df_clean[df_clean[outcome_col] > 0]

    # Calculate how many we dropped
    final_rows = len(df_clean)
    n_dropped = initial_rows - final_rows

    # Check if we have enough data left
    if final_rows < 10:
        return {
            "valid": False,
            "message": "Too few valid rows (<10) after cleaning.",
            "data": None,
            "n_dropped": n_dropped
        }

    # Construct the success message
    msg = f"Data validated. {n_dropped} rows dropped."

    return {
        "valid": True,
        "data": df_clean,
        "n_dropped": n_dropped,
        "message": msg
    }

def check_sample_sufficiency(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    complexity: int
) -> Dict[str, Union[bool, pd.DataFrame, None]]:
    """
    Ensures data is numeric and outcome is positive.

    Args:
        df (pd.DataFrame): Validated dataframe (from validate_data()).
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Signal column name.

    Returns:
        dict: A dictionary containing:
            - 'valid' (bool): True if all variables converge.
            - 'new_samples' (pd.DataFrame or None): The new sampling framework if more samples are required for convergence.
            - 'message' (str): Status message.
    """
