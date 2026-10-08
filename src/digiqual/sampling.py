import logging
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from scipy.stats import qmc

logger = logging.getLogger(__name__)

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

    Examples:
        Using a dictionary (Recommended):
        ```python
        ranges = {'Length': (0, 10), 'Angle': (0, 90)}
        df = generate_lhs(n=3, ranges=ranges, seed=42)
        print(df.round(2))
        #    Length  Angle
        # 0    3.75  85.54
        # 1    9.51  13.56
        # 2    7.32  54.17
        ```

        Using a DataFrame (Legacy/Advanced):
        ```python
        import pandas as pd
        vars_df = pd.DataFrame([
            {'Name': 'Length', 'Min': 0, 'Max': 10},
            {'Name': 'Angle', 'Min': 0, 'Max': 90}
        ])
        df = generate_lhs(n=3, ranges=vars_df, seed=42)
        ```
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
    except ValueError as err:
        raise ValueError("The 'Min' and 'Max' columns must be strictly numeric.") from err

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

    df = pd.DataFrame(sample_scaled, columns=vars_df["Name"])
    return reorder_max_min(df)


def reorder_max_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders a DataFrame of coordinates using a greedy max-min distance algorithm.
    This ensures that any prefix of size k (where k < len(df)) is as space-filling
    as possible, maximizing the spread and minimizing clustering.

    Args:
        df (pd.DataFrame): Coordinates to reorder.

    Returns:
        pd.DataFrame: The reordered DataFrame.
    """
    import numpy as np

    if len(df) <= 2:
        return df.copy()

    # Scale coordinates to [0, 1] range to ensure fair distance calculation across variables
    X = df.values
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # Prevent division by zero

    X_scaled = (X - min_vals) / range_vals

    n_points = len(df)
    ordered_indices = [0]  # Start with the first point

    # Distance from every point to its nearest already-selected point. Updating it
    # with the newest point only keeps each step O(n) (O(n^2) overall).
    min_dists = np.linalg.norm(X_scaled - X_scaled[0], axis=1)
    min_dists[0] = -np.inf  # already selected

    for _ in range(n_points - 1):
        # The point furthest from everything selected so far (ties: lowest index)
        best = int(np.argmax(min_dists))
        ordered_indices.append(best)
        min_dists = np.minimum(min_dists, np.linalg.norm(X_scaled - X_scaled[best], axis=1))
        min_dists[ordered_indices] = -np.inf

    return df.iloc[ordered_indices].copy()
