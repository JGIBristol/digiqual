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

from typing import Union # Add this to your typing imports
from .executors import Executor, CLIExecutor # Import our new architecture

import time
from sklearn.model_selection import cross_val_score


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
        max_attempts = 10
        candidates_list = []
        for _ in range(max_attempts):
            filtered = _filter_by_graveyard(candidates, graveyard, all_inputs, threshold)
            candidates_list.append(filtered)
            if sum(len(c) for c in candidates_list) >= n:
                break
            # Generate more if needed
            sample_01 = sampler.random(n=n_pool)
            for i, col in enumerate(all_inputs):
                if col == target_col:
                    scaled_data[col] = sample_01[:, i] * (gap_end - gap_start) + gap_start
                else:
                    scaled_data[col] = sample_01[:, i] * (df[col].max() - df[col].min()) + df[col].min()
            candidates = pd.DataFrame(scaled_data)
        candidates = pd.concat(candidates_list, ignore_index=True)

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

    # --- NEW: Dynamically find best degree ---
    best_degree = 1
    best_mse = float('inf')
    for d in [1, 2, 3]:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        try:
            scores = cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
            mse = -np.mean(scores)
            if mse < best_mse:
                best_mse = mse
                best_degree = d
        except ValueError:
            pass # fallback to 1 if not enough data


    for i in range(10):
        # A) Resample: Create a 'bootstrap' dataset (same size as original, but shuffled with duplicates)
        X_res, y_res = resample(X, y, random_state=i)

        # B) Define Model: Polynomial (curves) + Linear Regression (solver)
        model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())

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



# MAIN FUNCTION: generate_targeted_samples()

def generate_targeted_samples(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    n_new_per_fix: int = 10,
    failed_data: Optional[pd.DataFrame] = None,
    distance_threshold: float = 0.05,
    max_gap_ratio: float = 0.20,
    min_r2_score: float = 0.50,
    max_avg_width: float = 0.15,
    max_tail_width: float = 0.30
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

    Examples:
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
    report = sample_sufficiency(
        df, input_cols, outcome_col,
        max_gap_ratio=max_gap_ratio,
        min_r2_score=min_r2_score,
        max_avg_width=max_avg_width,
        max_tail_width=max_tail_width
    )

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

#### Helper Function for Adaptive Search

def _validate_executor_output(results: pd.DataFrame, expected_samples: pd.DataFrame, input_cols: List[str], outcome_col: str) -> None:
    """Validates that the executor returned a properly formatted table."""
    if results.empty:
        return # Empty is allowed (means 100% failure, which the loop handles gracefully)

    if outcome_col not in results.columns:
        raise ValueError(
            f"Executor Output Error: The solver failed to return the outcome column '{outcome_col}'. "
            f"Returned columns were: {list(results.columns)}"
        )

    missing_inputs = [col for col in input_cols if col not in results.columns]
    if missing_inputs:
        raise ValueError(
            f"Executor Output Error: The solver dropped required input columns: {missing_inputs}. "
            f"The Executor must return both inputs and outcomes."
        )

    if len(results) != len(expected_samples):
        raise ValueError(
            f"Executor Output Error: Row mismatch! We sent {len(expected_samples)} samples, "
            f"but the solver returned {len(results)} rows."
        )


### MAIN FUNCTION: Adaptive Search
def run_adaptive_search(
    executor: Executor | str,
    input_cols: List[str],
    outcome_col: str,
    ranges: Union[pd.DataFrame, Dict[str, Tuple[float, float]]],
    existing_data: Optional[pd.DataFrame] = None,
    n_start: int = 10,
    n_step: int = 5,
    max_iter: int = 5,
    max_hours: Optional[float] = None,
    # --- The 4 Custom Diagnostic Thresholds ---
    max_gap_ratio: float = 0.20,
    min_r2_score: float = 0.50,
    max_avg_width: float = 0.15,
    max_tail_width: float = 0.30
) -> pd.DataFrame:
    """
    Orchestrates the Active Learning loop on raw DataFrames using the Executor architecture.

    Args:
        executor (Executor | str): An instance of an Executor (Python, CLI, Matlab).
                                    Accepts a legacy command string for backward compatibility.
        input_cols (List[str]): Input names.
        outcome_col (str): Outcome name.
        ranges (Dict): Input bounds.
        existing_data (pd.DataFrame, optional): Start data.
        n_start (int): Init batch size.
        n_step (int): Points added per refinement step.
        max_iter (int): Max loops.
        max_hours (float, optional): Physical time limit in hours.
        max_gap_ratio (float): Decimal Percentage threshold for maximum gap diagnostics,
        min_r2_score (float): Decimal Percentage threshold for minimum r2 diagnostics,
        max_avg_width (float): Decimal Percentage threshold for average CI diagnostics,
        max_tail_width (float): Decimal Percentage threshold for maximum CI diagnostics

    Returns:
        pd.DataFrame: Final dataset containing all successful runs.
    """
    print("\n" + "="*40)
    print("      STARTING ADAPTIVE OPTIMIZATION")
    print("="*40)

    start_time = time.time()
    max_seconds = max_hours * 3600 if max_hours is not None else None

    # --- BACKWARD COMPATIBILITY ---
    if isinstance(executor, str):
        print("   -> Legacy command string detected. Wrapping in CLIExecutor.")
        executor = CLIExecutor(command_template=executor)

    current_data = existing_data.copy() if existing_data is not None else pd.DataFrame()
    total_attempted = 0
    failed_data = pd.DataFrame(columns=input_cols)

    # --- STEP 1: INITIALIZATION ---
    if current_data.empty:
        print(f"--- Iteration 0: Generating Initial Design ({n_start} points) ---")
        initial_samples = generate_lhs(n_start, ranges)
        total_attempted += len(initial_samples)

        results = executor.run(initial_samples)

        _validate_executor_output(results, initial_samples, input_cols, outcome_col)

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
        if max_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\n[!] TIME LIMIT REACHED ({max_hours} hrs). Stopping gracefully.")
                break

        print(f"\n--- Iteration {i+1}: Diagnostics Check ---")

        clean_df, _ = validate_simulation(current_data, input_cols, outcome_col)
        current_data = clean_df.copy()

        if clean_df.empty:
            diag = pd.DataFrame()
        else:
            # --- UPDATED: Pass the custom thresholds into sample_sufficiency ---
            diag = sample_sufficiency(
                clean_df, input_cols, outcome_col,
                skip_validation=True,
                max_gap_ratio=max_gap_ratio,
                min_r2_score=min_r2_score,
                max_avg_width=max_avg_width,
                max_tail_width=max_tail_width
            )

        # Convergence Check
        if not diag.empty and diag['Pass'].all():
            print("\n>>> CONVERGENCE REACHED! <<<")
            break

        print(">> Model invalid. Refining design...")
        new_samples = generate_targeted_samples(
            clean_df, input_cols, outcome_col, n_new_per_fix=n_step,
            failed_data=failed_data, distance_threshold=0.05,
            max_gap_ratio=max_gap_ratio,
            min_r2_score=min_r2_score,
            max_avg_width=max_avg_width,
            max_tail_width=max_tail_width
        )

        if new_samples.empty:
            print("\n>> Refinement algorithm converged (no new valid samples needed).")
            break

        print(f"--- Running Batch {i+1} ({len(new_samples)} points) ---")
        total_attempted += len(new_samples)

        new_results = executor.run(new_samples)

        _validate_executor_output(new_results, new_samples, input_cols, outcome_col)

        if new_results.empty:
            failed_data = pd.concat([failed_data, new_samples[input_cols]], ignore_index=True)
        else:
            successful_mask = new_results[outcome_col].notna()
            new_successful = new_results[successful_mask]
            current_data = pd.concat([current_data, new_successful], ignore_index=True)
            new_failed = new_results[~successful_mask][input_cols]
            failed_data = pd.concat([failed_data, new_failed], ignore_index=True)

    # --- STEP 3: FINAL REPORTING ---
    end_time = time.time()
    total_duration_mins = (end_time - start_time) / 60

    print("\n" + "-"*40)
    print(">>> SEARCH COMPLETE <<<")
    print(f"Total Time:      {total_duration_mins:.2f} minutes")
    print(f"Successful Runs: {len(current_data)}")
    print(f"Failed Runs:     {len(failed_data)} (in graveyard)")
    print(f"Total Attempted: {total_attempted}")
    print("-"*40 + "\n")

    return current_data
