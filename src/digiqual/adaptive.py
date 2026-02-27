import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

from .diagnostics import sample_sufficiency, validate_simulation
from .sampling import generate_lhs

import subprocess
import os


#### Helper Functions for generate_targeted_samples()

def _filter_by_graveyard(
    candidates: pd.DataFrame,
    graveyard: pd.DataFrame,
    input_cols: List[str],
    threshold: float = 0.05
) -> pd.DataFrame:
    """Filters out candidates that fall within a normalized distance threshold of known failures."""
    if graveyard is None or graveyard.empty or candidates.empty:
        return candidates

    # Combine to find the global min and max for fair 0-1 normalization
    combined = pd.concat([candidates[input_cols], graveyard[input_cols]], ignore_index=True)

    # Force float types immediately to prevent "object" dtype errors
    min_vals = combined.min().astype(float)
    max_vals = combined.max().astype(float)
    range_vals = (max_vals - min_vals).replace(0.0, 1.0)

    # Extract raw, pure NumPy arrays for safe and fast mathematical operations
    cand_arr = ((candidates[input_cols] - min_vals) / range_vals).values.astype(float)
    grave_arr = ((graveyard[input_cols] - min_vals) / range_vals).values.astype(float)

    keep_indices = []

    # Check each candidate against all points in the graveyard using pure arrays
    for i in range(len(cand_arr)):
        cand_row = cand_arr[i]

        # Calculate Euclidean distance to all failed points
        distances = np.sqrt(((grave_arr - cand_row)**2).sum(axis=1))

        # If the closest failed point is further than our threshold, keep it!
        if distances.min() > threshold:
            keep_indices.append(candidates.index[i]) # Save the original DataFrame index

    return candidates.loc[keep_indices].copy()


def _fill_gaps(
    df: pd.DataFrame,
    target_col: str,
    all_inputs: List[str],
    n: int,
    graveyard: Optional[pd.DataFrame] = None,
    threshold: float = 0.05
) -> pd.DataFrame:
    """Identifies the largest empty interval and generates samples inside it."""
    # 1. Find the coordinates of the largest gap
    sorted_vals = np.sort(df[target_col].values)
    gaps = np.diff(sorted_vals)
    max_gap_idx = np.argmax(gaps)
    gap_start = sorted_vals[max_gap_idx]
    gap_end = sorted_vals[max_gap_idx + 1]

    # 2. Generate random candidates (Oversample to ensure we have enough after filtering)
    n_pool = max(n * 10, 100)
    sampler = qmc.LatinHypercube(d=len(all_inputs))
    sample_01 = sampler.random(n=n_pool)

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

    candidates = pd.DataFrame(scaled_data)

    # 4. Filter against the graveyard BEFORE selecting the final n
    if graveyard is not None and not graveyard.empty:
        candidates = _filter_by_graveyard(candidates, graveyard, all_inputs, threshold)

    # Return exactly the number requested
    return candidates.head(n).copy()


def _sample_uncertainty(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    n: int,
    graveyard: Optional[pd.DataFrame] = None,
    threshold: float = 0.05
) -> pd.DataFrame:
    """Uses Bootstrap Query-by-Committee to find regions of high variance."""

    if n <= 0:
        return pd.DataFrame(columns=input_cols)

    # --- STEP 1: GENERATE CANDIDATE POOL ---
    # We create 1,000 "what-if" scenarios across your input ranges.
    n_candidates = 1000
    sampler = qmc.LatinHypercube(d=len(input_cols))
    sample_01 = sampler.random(n=n_candidates)

    candidates = pd.DataFrame(index=range(n_candidates))
    for i, col in enumerate(input_cols):
        # This math scales the 0-1 random samples to your data's actual Min and Max
        candidates[col] = sample_01[:, i] * (df[col].max() - df[col].min()) + df[col].min()

    # --- NEW: Filter the candidate pool BEFORE running expensive predictions ---
    if graveyard is not None and not graveyard.empty:
        candidates = _filter_by_graveyard(candidates, graveyard, input_cols, threshold)

        # If the filter removed everything (highly unlikely), generate a quick fallback
        if candidates.empty:
            return pd.DataFrame(columns=input_cols)

    # Reset index so predictions map cleanly
    candidates = candidates.reset_index(drop=True)

    # --- STEP 2: TRAIN THE COMMITTEE ---
    # We prepare to store predictions from 10 different versions of our model.
    X = df[input_cols].values
    y = df[outcome_col].values
    preds = np.zeros((len(candidates), 10)) # A matrix to hold rows x 10 model guesses

    for i in range(10):
        # A) Resample: Create a 'bootstrap' dataset (same size as original, but shuffled with duplicates)
        X_res, y_res = resample(X, y, random_state=i)

        # B) Define Model: Polynomial (curves) + Linear Regression (solver)
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())

        # C) Train: The model learns the relationship based on this specific bootstrap sample
        model.fit(X_res, y_res)

        # D) Predict: We ask this specific model to guess the outcomes for all valid candidates
        preds[:, i] = model.predict(candidates[input_cols].values)

    # --- STEP 3: MEASURE DISAGREEMENT ---
    # For each candidate point, we look at the 10 guesses.
    # If the Standard Deviation is high, the models are "confused" or disagreeing.
    uncertainty = np.std(preds, axis=1)

    # --- STEP 4: SELECTION ---
    # We sort the candidates by uncertainty and take the 'n' highest ones.
    top_indices = np.argsort(uncertainty)[-n:]

    # These are the coordinates where you should run your next experiments/simulations!
    return candidates.iloc[top_indices].copy()



#### Main Function: generate_targeted_samples() ####

def generate_targeted_samples(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    n_new_per_fix: int = 10,
    failed_data: Optional[pd.DataFrame] = None,
    distance_threshold: float = 0.05
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
        failed_data (Optional[pd.DataFrame]): Graveyard of inputs that crashed the solver.
        distance_threshold (float): Minimum normalized distance to maintain from failed points.

    Returns:
        pd.DataFrame: New recommended samples.

    Examples
    --------
    ```python
    import pandas as pd
    # 1. Setup data with a massive gap in 'Length' (0-1, then 9-10)
    df = pd.DataFrame({'Length': [0.1, 0.9, 9.1, 9.9], 'Signal': [1, 1, 1, 1]})

    # 2. Ask for new samples to fix the gap
    new_pts = generate_targeted_samples(
        df=df,
        input_cols=['Length'],
        outcome_col='Signal',
        n_new_per_fix=2
    )
    print(new_pts)
    #    Length Refinement_Reason
    # 0     5.4      Gap in Length
    # 1     3.2      Gap in Length
    ```
    """
    # 1. SENSE: Run the diagnostics to get the status report
    report = sample_sufficiency(df, input_cols, outcome_col)

    # Quick exit if everything is green
    if report.empty or report['Pass'].all():
        print("All diagnostic checks passed. No new samples needed.")
        return pd.DataFrame()

    print("Diagnostics flagged issues. Initiating Active Learning...")

    # Check if we have anything to report from the graveyard diagnostic checks
    if failed_data is not None and not failed_data.empty:
        print(f" -> Active Graveyard Tracker: Protecting against {len(failed_data)} known bad regions.")

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
            # Check if we already handled this variable
            if var_name in handled_vars:
                continue

            print(f" -> Strategy: Exploration (Filling gaps in {var_name})")
            # Call the specific solver for gaps, passing graveyard limits
            samples = _fill_gaps(
                df, var_name, input_cols, n_new_per_fix,
                graveyard=failed_data, threshold=distance_threshold
            )

            # --- NEW: Tag the reason for these samples ---
            samples['Refinement_Reason'] = f"Gap in {var_name}"

            new_samples_list.append(samples)
            handled_vars.add(var_name)

        # --- Handler B: Model Stability (Exploitation) ---
        # Only run this once per batch, even if multiple metrics fail
        elif test_name in ["Model Fit (CV)", "Bootstrap Convergence"]:
            if "Global_Model" not in handled_vars:
                print(" -> Strategy: Exploitation (Targeting high uncertainty regions)")
                # Call the specific solver for uncertainty, passing graveyard limits
                samples = _sample_uncertainty(
                    df, input_cols, outcome_col, n_new_per_fix,
                    graveyard=failed_data, threshold=distance_threshold
                )

                # --- NEW: Tag the reason for these samples ---
                samples['Refinement_Reason'] = "High Model Uncertainty"

                new_samples_list.append(samples)
                handled_vars.add("Global_Model")

    # 3. ACT: Combine all recommendations
    if not new_samples_list:
        return pd.DataFrame()

    return pd.concat(new_samples_list, ignore_index=True)


#### Batch Functionality ####
# 1. INTERNAL HELPER: Execution Logistics
def _execute_simulation(
    samples: pd.DataFrame,
    command_template: str,
    input_cols: List[str],
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Internal helper: writes CSV, runs command, reads CSV.
    Same implementation as before.
    """
    samples[input_cols].to_csv(input_path, index=False)
    cmd = command_template.format(input=input_path, output=output_path)

    print(f"   -> Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"   -> Simulation FAILED (Exit Code {e.returncode}).")
        return pd.DataFrame()

    if not os.path.exists(output_path):
        return pd.DataFrame()

    try:
        return pd.read_csv(output_path)
    except Exception:
        return pd.DataFrame()


# 2. MAIN FUNCTION: The Agnostic Adaptive Loop
def run_adaptive_search(
    command: str,
    input_cols: List[str],
    outcome_col: str,
    ranges: Dict[str, Tuple[float, float]],
    existing_data: Optional[pd.DataFrame] = None,
    n_start: int = 20,
    n_step: int = 10,
    max_iter: int = 5,
    input_file: str = "sim_input.csv",
    output_file: str = "sim_output.csv"
) -> pd.DataFrame:
    """
    Orchestrates the Active Learning loop on raw DataFrames.

    Args:
        command (str): Solver command template.
        input_cols (List[str]): Input names.
        outcome_col (str): Outcome name.
        ranges (Dict): Input bounds.
        existing_data (pd.DataFrame, optional): Start data.
        n_start (int): Init batch size.
        max_iter (int): Max loops.

    Returns:
        pd.DataFrame: Final dataset.

    Examples
    --------
    ```python
    # 1. Define bounds
    ranges = {'Length': (0, 10), 'Angle': (-45, 45)}

    # 2. Define a command that reads {input} and writes {output}
    # (Here we use python -c to simulate a solver)
    cmd = (
    "python -c "
    "'import pandas as pd; "
    'df=pd.read_csv("{input}"); '
    'df["Signal"] = df["Length"]*2; '
    'df.to_csv("{output}", index=False)'
    "'"
    )

    # 3. Run the loop (Init -> Run -> Check -> Refine)
    final_df = run_adaptive_search(
        command=cmd,
        input_cols=['Length', 'Angle'],
        outcome_col='Signal',
        ranges=ranges,
        max_iter=2,
        n_start=5
    )
    print(len(final_df))
    ```
    """
    print("\n=== STARTING ADAPTIVE OPTIMIZATION ===")

    current_data = existing_data.copy() if existing_data is not None else pd.DataFrame()
    total_attempted = 0

    # NEW: Initialize the Failure Graveyard
    failed_data = pd.DataFrame(columns=input_cols)

    # --- STEP 1: INITIALIZATION ---
    if current_data.empty:
        print(f"--- Iteration 0: Generating Initial Design ({n_start} points) ---")

        initial_samples = generate_lhs(n_start, ranges)
        total_attempted += len(initial_samples)

        results = _execute_simulation(
            initial_samples, command, input_cols, input_file, output_file
        )

        # CLEANUP: Separate successes and failures safely
        if results.empty:
            failed_data = pd.concat([failed_data, initial_samples[input_cols]], ignore_index=True)
        else:
            successful_mask = results[outcome_col].notna()
            current_data = results[successful_mask].reset_index(drop=True)
            failed_data = pd.concat([failed_data, results[~successful_mask][input_cols]], ignore_index=True)

    else:
        print(f"--- Iteration 0: Resuming with {len(current_data)} existing points ---")

    # --- STEP 2: REFINEMENT LOOP ---
    for i in range(max_iter):
        print(f"\n--- Iteration {i+1}: Diagnostics Check ---")

        clean_df, _ = validate_simulation(current_data, input_cols, outcome_col)

        if clean_df.empty:
            diag = pd.DataFrame()
        else:
            diag = sample_sufficiency(clean_df, input_cols, outcome_col)

        # B. Success?
        if not diag.empty and diag['Pass'].all():
            print("\n>>> CONVERGENCE REACHED! <<<")
            print(f"Final Report: {len(current_data)} successful runs (out of {total_attempted} attempts). Graveyard contains {len(failed_data)} points.")
            return current_data

        # C. Refine
        print(">> Model invalid. Refining design...")
        new_samples = generate_targeted_samples(
            clean_df, input_cols, outcome_col, n_new_per_fix=n_step,
            failed_data=failed_data, distance_threshold=0.05
        )

        if new_samples.empty:
            print("\n>> Refinement algorithm converged (no new valid samples needed).")
            print(f"Final Report: {len(current_data)} successful runs (out of {total_attempted} attempts). Graveyard contains {len(failed_data)} points.")
            return current_data

        # D. Execute
        print(f"--- Running Batch {i+1} ({len(new_samples)} points) ---")
        total_attempted += len(new_samples)

        new_results = _execute_simulation(
            new_samples, command, input_cols, input_file, output_file
        )

        # E. Accumulate and Cleanup cohesively
        if new_results.empty:
            failed_data = pd.concat([failed_data, new_samples[input_cols]], ignore_index=True)
        else:
            successful_mask = new_results[outcome_col].notna()
            new_successful = new_results[successful_mask]
            current_data = pd.concat([current_data, new_successful], ignore_index=True)

            new_failed = new_results[~successful_mask][input_cols]
            failed_data = pd.concat([failed_data, new_failed], ignore_index=True)

    print(f"\n>>> WARNING: Max iterations ({max_iter}) reached. <<<")
    print(f"Final Report: {len(current_data)} successful runs (out of {total_attempted} attempts). Graveyard contains {len(failed_data)} points.")
    return current_data
