import pandas as pd
from scipy.stats import qmc
from typing import Optional


def generate_lhs(
    n: int, vars_df: pd.DataFrame, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates a Latin Hypercube Sample and scales it to the provided variable bounds.

    Args:
        n (int): The total number of samples to generate.
        vars_df (pd.DataFrame): A dataframe containing the variable definitions.
            Must contain columns: 'Name', 'Min', 'Max'.
        seed (int, optional): Sets the random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe containing the scaled simulation parameters, where column names correspond to vars_df['Name']. Returns an empty DataFrame if input vars_df is empty.

    Raises:
        ValueError: If required columns are missing, types are incorrect, or Min >= Max.
    """
    if vars_df.empty:
        return pd.DataFrame()

    # --- VALIDATION CHECKS ---

    # 0. Check required columns exist
    required_cols = {"Name", "Min", "Max"}
    if not required_cols.issubset(vars_df.columns):
        missing = required_cols - set(vars_df.columns)
        raise ValueError(f"Input dataframe is missing columns: {missing}")

    # 1. Check 'Name' column contains strings
    if not vars_df["Name"].apply(lambda x: isinstance(x, str)).all():
        raise ValueError("The 'Name' column must contain only character strings.")

    # 2. Check 'Min' and 'Max' are numeric
    # We try to convert them; if it fails (e.g., "Five"), it raises an error
    try:
        l_bounds_series = pd.to_numeric(vars_df["Min"])
        u_bounds_series = pd.to_numeric(vars_df["Max"])
    except ValueError:
        raise ValueError("The 'Min' and 'Max' columns must be strictly numeric.")

    # 3. Check Min < Max
    if (l_bounds_series >= u_bounds_series).any():
        # Identify which variables are failing for a helpful error message
        bad_vars = vars_df.loc[l_bounds_series >= u_bounds_series, "Name"].tolist()
        raise ValueError(
            f"Bounds error: 'Min' must be strictly lower than 'Max'. Check variables: {bad_vars}"
        )

    # --- GENERATION LOGIC ---

    # 1. Setup dimensions and bounds
    l_bounds = l_bounds_series.values
    u_bounds = u_bounds_series.values
    d = len(vars_df)

    # 2. Initialize the Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=d, seed=seed)

    # 3. Generate 0-1 LHS
    sample_01 = sampler.random(n=n)

    # 4. Scale from [0,1] to [Min, Max]
    sample_scaled = qmc.scale(sample_01, l_bounds, u_bounds)

    # 5. Format output
    df = pd.DataFrame(sample_scaled, columns=vars_df["Name"])

    return df
