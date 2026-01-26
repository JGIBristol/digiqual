import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Error Function
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

# Helper Functions

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

def _check_bootstrap_convergence(df: pd.DataFrame, input_cols: List[str], outcome_col: str) -> Dict:
    """
    Checks 'Convergence' using Bootstrap Resampling.
    Calculates relative width of the Confidence Interval (CI) at the centroid.
    [cite_start]Ref: [cite: 305, 817]
    """
    n_bootstraps = 100
    n_samples = len(df)
    X = df[input_cols].values
    y = df[outcome_col].values

    test_point = np.median(X, axis=0).reshape(1, -1)
    predictions = []

    for _ in range(n_bootstraps):
        X_res, y_res = resample(X, y, n_samples=n_samples)
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        model.fit(X_res, y_res)
        predictions.append(model.predict(test_point)[0])

    predictions = np.array(predictions)
    lower = np.percentile(predictions, 2.5)
    upper = np.percentile(predictions, 97.5)
    ci_width = upper - lower
    relative_width = ci_width / np.abs(np.mean(predictions))

    return {
        "bootstrap_iterations": n_bootstraps,
        "ci_width_at_centroid": round(ci_width, 4),
        "relative_ci_width": round(relative_width, 4),
        "converged": relative_width < 0.10
    }


def sample_sufficiency(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str
) -> pd.DataFrame:
    """
    Performs statistical tests on sampling sufficiency.

    Runs 3 checks:
    1. Input Space Coverage (Gaps)
    2. Model Fit Stability (CV Score)
    3. Bootstrap Convergence (CI Width)

    Args:
        df (pd.DataFrame): Clean data from validate_simulation.
        input_cols (List[str]): List of input variable names.
        outcome_col (str): Signal column name.

    Returns:
        pd.DataFrame: A table containing pass/fail metrics for each test.
    """
    # 1. Validate simulation data
    df_clean, df_removed = validate_simulation(df, input_cols, outcome_col)

    if not df_removed.empty:
        raise ValidationError(
            f"Data contained {len(df_removed)} invalid rows. "
            "Please clean inputs using `validate_simulation()`."
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
            "Pass": res['sufficient_coverage']
        })

    # Model Fit Results
    flat_results.append({
        "Test": "Model Fit (CV)",
        "Variable": outcome_col,
        "Metric": "Mean R2 Score",
        "Value": fit_res['mean_r2_score'],
        "Pass": fit_res['stable_fit']
    })

    # Bootstrap Results
    flat_results.append({
        "Test": "Bootstrap Convergence",
        "Variable": outcome_col,
        "Metric": "Relative CI Width",
        "Value": boot_res['relative_ci_width'],
        "Pass": boot_res['converged']
    })

    return pd.DataFrame(flat_results)
