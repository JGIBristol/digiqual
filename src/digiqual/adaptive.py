import pandas as pd
import numpy as np
from typing import List
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

from .diagnostics import sample_sufficiency


#### Helper Functions for generate_targeted_samples()

def _fill_gaps(
    df: pd.DataFrame,
    target_col: str,
    all_inputs: List[str],
    n: int
    ) -> pd.DataFrame:
    """Identifies the largest empty interval and generates samples inside it."""
    # 1. Find the coordinates of the largest gap
    sorted_vals = np.sort(df[target_col].values)
    gaps = np.diff(sorted_vals)
    max_gap_idx = np.argmax(gaps)
    gap_start = sorted_vals[max_gap_idx]
    gap_end = sorted_vals[max_gap_idx + 1]

    # 2. Generate random candidates
    sampler = qmc.LatinHypercube(d=len(all_inputs))
    sample_01 = sampler.random(n=n)

    # 3. Scale candidates: Target col fits in gap; others span full range
    scaled_data = {}
    for i, col in enumerate(all_inputs):
        col_min, col_max = df[col].min(), df[col].max()

        if col == target_col:
            # Constrain to the gap
            scaled_data[col] = sample_01[:, i] * (gap_end - gap_start) + gap_start
        else:
            # Constrain to full domain
            scaled_data[col] = sample_01[:, i] * (col_max - col_min) + col_min

    return pd.DataFrame(scaled_data)

def _sample_uncertainty(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    n: int
    ) -> pd.DataFrame:
    """Uses Bootstrap Query-by-Committee to find regions of high variance."""
    # 1. Create a large candidate pool (e.g. 1000 points)
    n_candidates = 1000
    sampler = qmc.LatinHypercube(d=len(input_cols))
    sample_01 = sampler.random(n=n_candidates)

    candidates = pd.DataFrame(index=range(n_candidates))
    for i, col in enumerate(input_cols):
        candidates[col] = sample_01[:, i] * (df[col].max() - df[col].min()) + df[col].min()

    # 2. Train Committee (10 models on resampled data)
    X = df[input_cols].values
    y = df[outcome_col].values
    preds = np.zeros((n_candidates, 10))

    for i in range(10):
        X_res, y_res = resample(X, y, random_state=i) # seeded for reproducibility
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X_res, y_res)
        preds[:, i] = model.predict(candidates[input_cols].values)

    # 3. Calculate Uncertainty (Standard Deviation of predictions)
    uncertainty = np.std(preds, axis=1)

    # 4. Pick top N points with highest uncertainty
    top_indices = np.argsort(uncertainty)[-n:]
    return candidates.iloc[top_indices].copy()



#### Main Function: generate_targeted_samples() ####

def generate_targeted_samples(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    n_new_per_fix: int = 10
) -> pd.DataFrame:
    """
    Active Learning Engine: Generates new samples based on diagnostic failures.

    It consumes the results table from `sample_sufficiency`.
    - If `Input Coverage` fails -> Triggers `_fill_gaps` (Exploration).
    - If `Model Fit` or `Bootstrap` fails -> Triggers `_sample_uncertainty` (Exploitation).

    Args:
        df (pd.DataFrame): Current simulation data.
        input_cols (List[str]): Input variable names.
        outcome_col (str): Outcome variable name.
        n_new_per_fix (int): Number of samples to generate per detected issue.

    Returns:
        pd.DataFrame: A dataframe of recommended new simulation parameters.
    """
    # Run the diagnostics to get the status report
    report = sample_sufficiency(df, input_cols, outcome_col)

    # Exit if all Pass
    if report.empty or report['Pass'].all():
        print("All diagnostic checks passed. No new samples needed.")
        return pd.DataFrame()

    print("Diagnostics flagged issues. Initiating Active Learning...")
    new_samples_list = []

    # 2. DECIDE: Iterate through failures and dispatch handlers
    failures = report[~report['Pass']]

    # We use a set to track handled variables so we don't over-sample
    handled_vars = set()

    for _, row in failures.iterrows():
        test_name = row['Test']
        var_name = row['Variable']

        # --- Handler A: Input Coverage (Exploration) ---
        if test_name == "Input Coverage":
            print(f" -> Strategy: Exploration (Filling gaps in {var_name})")
            # Call the specific solver for gaps
            samples = _fill_gaps(df, var_name, input_cols, n_new_per_fix)
            new_samples_list.append(samples)
            handled_vars.add(var_name)

        # --- Handler B: Model Stability (Exploitation) ---
        # Only run this once per batch, even if multiple metrics fail
        elif test_name in ["Model Fit (CV)", "Bootstrap Convergence"]:
            if "Global_Model" not in handled_vars:
                print(" -> Strategy: Exploitation (Targeting high uncertainty regions)")
                # Call the specific solver for uncertainty
                samples = _sample_uncertainty(df, input_cols, outcome_col, n_new_per_fix)
                new_samples_list.append(samples)
                handled_vars.add("Global_Model")

    # 3. ACT: Combine all recommendations
    if not new_samples_list:
        return pd.DataFrame()

    return pd.concat(new_samples_list, ignore_index=True)
