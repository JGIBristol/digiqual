import pandas as pd
import numpy as np
from typing import List,Dict, Tuple, Optional
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

    # --- STEP 2: TRAIN THE COMMITTEE ---
    # We prepare to store predictions from 10 different versions of our model.
    X = df[input_cols].values
    y = df[outcome_col].values
    preds = np.zeros((n_candidates, 10)) # A matrix to hold 1,000 rows x 10 model guesses

    for i in range(10):
        # A) Resample: Create a 'bootstrap' dataset (same size as original, but shuffled with duplicates)
        X_res, y_res = resample(X, y, random_state=i)

        # B) Define Model: Polynomial (curves) + Linear Regression (solver)
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())

        # C) Train: The model learns the relationship based on this specific bootstrap sample
        model.fit(X_res, y_res)

        # D) Predict: We ask this specific model to guess the outcomes for all 1,000 candidates
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
            # Call the specific solver for gaps
            samples = _fill_gaps(df, var_name, input_cols, n_new_per_fix)

            # --- NEW: Tag the reason for these samples ---
            samples['Refinement_Reason'] = f"Gap in {var_name}"

            new_samples_list.append(samples)
            handled_vars.add(var_name)

        # --- Handler B: Model Stability (Exploitation) ---
        # Only run this once per batch, even if multiple metrics fail
        elif test_name in ["Model Fit (CV)", "Bootstrap Convergence"]:
            if "Global_Model" not in handled_vars:
                print(" -> Strategy: Exploitation (Targeting high uncertainty regions)")
                # Call the specific solver for uncertainty
                samples = _sample_uncertainty(df, input_cols, outcome_col, n_new_per_fix)

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

    # Initialize working dataset
    current_data = existing_data.copy() if existing_data is not None else pd.DataFrame()

    # --- STEP 1: INITIALIZATION ---
    if current_data.empty:
        print(f"--- Iteration 0: Generating Initial Design ({n_start} points) ---")

        # 1. Generate Samples (Dict input)
        initial_samples = generate_lhs(n_start, ranges)

        # 2. Run Simulation
        results = _execute_simulation(
            initial_samples, command, input_cols, input_file, output_file
        )
        current_data = results
    else:
        print(f"--- Iteration 0: Resuming with {len(current_data)} existing points ---")

    # --- STEP 2: REFINEMENT LOOP ---
    for i in range(max_iter):
        print(f"\n--- Iteration {i+1}: Diagnostics Check ---")

        # A. Diagnose (Using pure functions from diagnostics.py)
        # We must validate first to ensure clean data for metrics
        clean_df, _ = validate_simulation(current_data, input_cols, outcome_col)

        if clean_df.empty:
            # Fallback if validation kills everything (rare)
            diag = pd.DataFrame()
        else:
            diag = sample_sufficiency(clean_df, input_cols, outcome_col)

        # B. Success?
        if not diag.empty and diag['Pass'].all():
            print(f"\n>>> CONVERGENCE REACHED at {len(current_data)} points! <<<")
            return current_data

        # C. Refine (Using pure function from adaptive.py)
        print(">> Model invalid. Refining design...")
        # Note: generate_targeted_samples is likely in this same file or imported
        new_samples = generate_targeted_samples(
            clean_df, input_cols, outcome_col, n_new_per_fix=n_step
        )

        if new_samples.empty:
            print(">> Refinement algorithm converged.")
            return current_data

        # D. Execute
        print(f"--- Running Batch {i+1} ({len(new_samples)} points) ---")
        new_results = _execute_simulation(
            new_samples, command, input_cols, input_file, output_file
        )

        # E. Accumulate
        current_data = pd.concat([current_data, new_results], ignore_index=True)

    print(f"\n>>> WARNING: Max iterations ({max_iter}) reached. <<<")
    return current_data
