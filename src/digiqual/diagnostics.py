import pandas as pd
from typing import List, Tuple

# Define a custom exception for clarity
class ValidationError(Exception):
    """Raised when simulation data fails validation checks."""
    pass


def validate_simulation(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates simulation data, coercing to numeric and removing invalid rows.

    Args:
        df (pd.DataFrame): Raw data.
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Signal column name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_clean: The validated, numeric dataframe ready for analysis.
            - df_removed: A dataframe containing the rows that were dropped (with original values).

    Raises:
        ValidationError: If columns are missing, types are wrong, or too few valid rows remain.
    """
    # --- 1. Basic Structure Checks (Fail Fast) ---
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValidationError("Input is not a valid pandas DataFrame or is empty.")

    required_cols = input_cols + [outcome_col]

    # Check for missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    # Check for variable overlap
    if outcome_col in input_cols:
        raise ValidationError(f"Outcome variable '{outcome_col}' cannot also be an Input variable.")

    # --- 2. Data Cleaning ---
    # We work on a subset to avoid touching unrelated columns
    subset = df[required_cols].copy()

    # Coerce to numeric (turn "Error", "N/A" into NaN)
    subset_numeric = subset.apply(pd.to_numeric, errors='coerce')

    # Identify rows that are valid (No NaNs AND Outcome > 0)
    # 1. Must be numeric (not NaN)
    mask_numeric = subset_numeric.notna().all(axis=1)
    # 2. Outcome must be positive (Signal > 0)
    # (We use fillna(False) to ensure NaNs in outcome don't crash this check)
    mask_positive = subset_numeric[outcome_col] > 0
    mask_positive = mask_positive.fillna(False)

    # Combine masks
    mask_valid = mask_numeric & mask_positive

    # --- 3. Split Data ---
    # Clean data: Use the numeric version
    df_clean = subset_numeric.loc[mask_valid].copy()

    # Removed data: Use the ORIGINAL version so user can see why it failed
    df_removed = df.loc[~mask_valid].copy()

    # --- 4. Final Sufficiency Check ---
    if len(df_clean) < 10:
        raise ValidationError(
            f"Too few valid rows remaining ({len(df_clean)}) after cleaning. "
            "Analysis requires at least 10 valid data points."
        )

    return df_clean, df_removed
