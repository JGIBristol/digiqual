import pandas as pd
from scipy.stats import qmc
from typing import Optional, Union, Dict, Tuple

def generate_lhs(
    n: int,
    ranges: Union[pd.DataFrame, Dict[str, Tuple[float, float]]],
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates a Latin Hypercube Sample and scales it to the provided variable bounds.

    Args:
        n (int): The total number of samples to generate.
        ranges (Union[pd.DataFrame, Dict]): Definition of input variables.
            - Dict Format: {'Name': (Min, Max), ...} e.g. {'Length': (0, 10)}
            - DataFrame Format: Columns ['Name', 'Min', 'Max']
        seed (int, optional): Sets the random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe containing the scaled simulation parameters,
            where column names correspond to the keys in `ranges`.
            Returns an empty DataFrame if `ranges` is empty.

    Raises:
        ValueError: If required columns are missing, types are incorrect, or Min >= Max.
        TypeError: If `ranges` is not a Dictionary or DataFrame.
    """
    # --- 1. NORMALIZE INPUT TO DATAFRAME ---
    if isinstance(ranges, dict):
        # Convert cleaner Dict format to the internal DataFrame format
        if not ranges:
            return pd.DataFrame()

        vars_df = pd.DataFrame([
            {"Name": k, "Min": v[0], "Max": v[1]}
            for k, v in ranges.items()
        ])
    elif isinstance(ranges, pd.DataFrame):
        vars_df = ranges.copy()
    else:
        raise TypeError("Input 'ranges' must be a Dictionary or a DataFrame.")

    if vars_df.empty:
        return pd.DataFrame()

    # --- 2. VALIDATION CHECKS ---
    required_cols = {"Name", "Min", "Max"}
    if not required_cols.issubset(vars_df.columns):
        missing = required_cols - set(vars_df.columns)
        raise ValueError(f"Input dataframe is missing columns: {missing}")

    if not vars_df["Name"].apply(lambda x: isinstance(x, str)).all():
        raise ValueError("The 'Name' column must contain only character strings.")

    try:
        l_bounds_series = pd.to_numeric(vars_df["Min"])
        u_bounds_series = pd.to_numeric(vars_df["Max"])
    except ValueError:
        raise ValueError("The 'Min' and 'Max' columns must be strictly numeric.")

    if (l_bounds_series >= u_bounds_series).any():
        bad_vars = vars_df.loc[l_bounds_series >= u_bounds_series, "Name"].tolist()
        raise ValueError(
            f"Bounds error: 'Min' must be strictly lower than 'Max'. Check variables: {bad_vars}"
        )

    # --- 3. GENERATION LOGIC ---
    l_bounds = l_bounds_series.values
    u_bounds = u_bounds_series.values
    d = len(vars_df)

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample_01 = sampler.random(n=n)
    sample_scaled = qmc.scale(sample_01, l_bounds, u_bounds)

    return pd.DataFrame(sample_scaled, columns=vars_df["Name"])
