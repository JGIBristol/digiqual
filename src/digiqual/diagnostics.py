import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

#### Error Function ####
class ValidationError(Exception):
    """Raised when simulation data fails validation checks."""
    pass

#### Simulation Validation ####
def validate_simulation(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates simulation data, coercing to numeric and removing invalid rows.

    Args:
        df (pd.DataFrame): The raw dataframe containing input columns and the outcome column.
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Name of the outcome variable.

    Returns:
        (Tuple[pd.DataFrame, pd.DataFrame]):
            * `df_clean`: The validated, numeric dataframe ready for analysis.
            * `df_removed`: A dataframe containing the rows that were dropped.

    Raises:
        ValidationError: If columns are missing, types are wrong, or too few valid rows remain.

    Examples
    --------
    ```python
    import pandas as pd
    # Create dirty data (includes text and negative values)
    df = pd.DataFrame({
        'Length': [1.0, 'BadValue', 5.0],
        'Signal': [0.5, 0.8, -1.2]
    })

    # Validate
    clean, removed = validate_simulation(df, ['Length'], 'Signal')
    print(f"Clean rows: {len(clean)}")
    print(f"Removed rows: {len(removed)}")
    ```

    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValidationError("Input is not a valid pandas DataFrame or is empty.")

    required_cols = input_cols + [outcome_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    if outcome_col in input_cols:
        raise ValidationError(f"Outcome variable '{outcome_col}' cannot also be an Input variable.")

    # Data Cleaning
    subset = df[required_cols].copy()
    subset_numeric = subset.apply(pd.to_numeric, errors='coerce')

    mask_numeric = subset_numeric.notna().all(axis=1)
    # Ensure outcome is positive (common requirement for PoD, adjust if needed)
    mask_positive = subset_numeric[outcome_col] > 0
    mask_positive = mask_positive.fillna(False)
    mask_valid = mask_numeric & mask_positive

    df_clean = subset_numeric.loc[mask_valid].copy()
    df_removed = df.loc[~mask_valid].copy()

    if len(df_clean) < 10:
        raise ValidationError(
            f"Too few valid rows remaining ({len(df_clean)}) after cleaning. "
            "Analysis requires at least 10 valid data points."
        )

    return df_clean, df_removed

#### Helper Functions for sample_sufficiency() ####

def _check_input_coverage(df: pd.DataFrame, input_cols: List[str]) -> Dict:
    """
    Checks for 'Input Space Coverage' (Uniformity).
    Ensures there are no large gaps (>20%) in the sampling of predictor variables.
    """
    results = {}
    for col in input_cols:
        sorted_vals = np.sort(df[col].values)
        gaps = np.diff(sorted_vals)
        data_range = sorted_vals[-1] - sorted_vals[0]

        if data_range == 0:
            max_gap_ratio = 1.0
        else:
            max_gap_ratio = np.max(gaps) / data_range

        results[col] = {
            "min": float(sorted_vals[0]),
            "max": float(sorted_vals[-1]),
            "max_gap_ratio": round(max_gap_ratio, 4),
            "sufficient_coverage": max_gap_ratio < 0.2
        }
    return results

def _check_model_fit(df: pd.DataFrame, input_cols: List[str], outcome_col: str) -> Dict:
    """
    Checks 'Model Fit Quality' using k-fold Cross-Validation on a 3rd order polynomial.
    """
    X = df[input_cols]
    y = df[outcome_col]

    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

    k = 10 if len(df) > 50 else 5
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    return {
        "model_type": "Polynomial (deg=3)",
        "cv_folds": k,
        "mean_r2_score": round(np.mean(scores), 4),
        "stable_fit": np.mean(scores) > 0.5
    }

def _check_bootstrap_convergence(df: pd.DataFrame, input_cols: List[str], outcome_col: str, n_bootstraps: int = 100) -> Dict:
    X = df[input_cols].values
    y = df[outcome_col].values
    n_samples = len(df)

    # 1. Probe Points
    probe_points = np.percentile(X, [10, 50, 90], axis=0)
    all_predictions = []

    for _ in range(n_bootstraps):
        X_res, y_res = resample(X, y, n_samples=n_samples)
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X_res, y_res)
        preds = model.predict(probe_points)
        all_predictions.append(preds)

    all_predictions = np.array(all_predictions)

    # 2. Calculate Stats
    stds = np.std(all_predictions, axis=0)
    means = np.abs(np.mean(all_predictions, axis=0))

    epsilon = 1e-6
    relative_widths = stds / (means + epsilon)

    # 3. Determine overall convergence
    # We use a threshold on the AVERAGE stability across probes
    # to allow for some heteroskedasticity at the tails.
    avg_rel_width = np.mean(relative_widths)
    max_rel_width = np.max(relative_widths)

    # Logic: Pass if the average stability is good (e.g., < 15%)
    # and the worst tail isn't totally wild (e.g., < 30%)
    is_converged = avg_rel_width < 0.15 and max_rel_width < 0.30

    return {
        "bootstrap_iterations": n_bootstraps,
        "avg_relative_width": round(avg_rel_width, 4),
        "max_relative_width": round(max_rel_width, 4),
        "converged": bool(is_converged)
    }

#### Main Function: sample_sufficiency() ####


def sample_sufficiency(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str
) -> pd.DataFrame:
    """
    Performs statistical tests on sampling sufficiency.

    Runs 3 checks:
        1. **Input Space Coverage** (Gaps)
        2. **Model Fit Stability** (CV Score)
        3. **Bootstrap Convergence** (Coefficient of Variation)

    Args:
        df (pd.DataFrame): The simulation data. Will be validated via `validate_simulation` internally.
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Name of the outcome variable.

    Returns:
        pd.DataFrame: A table containing pass/fail metrics for each test,
                      including the threshold values evaluated against.
    """
    # 1. Validate simulation data
    df_clean, df_removed = validate_simulation(df, input_cols, outcome_col)

    if not df_removed.empty:
        print(f"Note: {len(df_removed)} invalid rows were dropped automatically.")

    if len(df_clean) < 10:
        raise ValidationError(
            f"Insufficient valid data ({len(df_clean)} rows). "
            "Diagnostics require at least 10 valid data points."
        )

    # 2. Run All Checks
    coverage_res = _check_input_coverage(df_clean, input_cols)
    fit_res = _check_model_fit(df_clean, input_cols, outcome_col)
    boot_res = _check_bootstrap_convergence(df_clean, input_cols, outcome_col)

    # 3. Format Results Table
    flat_results = []

    # Input Coverage Results
    for col, res in coverage_res.items():
        flat_results.append({
            "Test": "Input Coverage",
            "Variable": col,
            "Metric": "Max Gap Ratio",
            "Value": res['max_gap_ratio'],
            "Threshold": "< 0.20",
            "Pass": res['sufficient_coverage']
        })

    # Model Fit Results
    flat_results.append({
        "Test": "Model Fit (CV)",
        "Variable": outcome_col,
        "Metric": "Mean R2 Score",
        "Value": fit_res['mean_r2_score'],
        "Threshold": "> 0.50",
        "Pass": fit_res['stable_fit']
    })

    # Bootstrap Results
    flat_results.append({
        "Test": "Bootstrap Convergence",
        "Variable": outcome_col,
        "Metric": "Avg CV (Rel Std Dev)",
        "Value": boot_res['avg_relative_width'],
        "Threshold": "< 0.15",
        "Pass": boot_res['converged']
    })

    flat_results.append({
        "Test": "Bootstrap Convergence",
        "Variable": outcome_col,
        "Metric": "Max CV (Rel Std Dev)",
        "Value": boot_res['max_relative_width'],
        "Threshold": "< 0.30",
        "Pass": boot_res['converged']
    })

    return pd.DataFrame(flat_results)
