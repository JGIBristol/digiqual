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

    Examples:
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

    # Force everything to be a number. Text becomes NaN.
    subset_numeric = subset.apply(pd.to_numeric, errors='coerce')

    # A row is only valid if EVERY required column has a real number in it
    mask_valid = subset_numeric.notna().all(axis=1)

    df_clean = subset_numeric.loc[mask_valid].copy()
    df_removed = df.loc[~mask_valid].copy()

    if len(df_clean) < 10:
        raise ValidationError(
            f"Too few valid rows remaining ({len(df_clean)}) after cleaning. "
            "Analysis requires at least 10 valid data points."
        )

    return df_clean, df_removed


#### Helper Functions for sample_sufficiency() ####

def _check_input_coverage(df: pd.DataFrame, input_cols: List[str], max_gap_ratio: float = 0.20) -> Dict:
    """
    Evaluates if the input space is sampled densely enough without excessively large gaps.
    Calculates the maximum distance between adjacent sorted points as a ratio of the total range.
    """
    results = {}
    for col in input_cols:
        sorted_vals = np.sort(df[col].values)
        gaps = np.diff(sorted_vals)
        data_range = sorted_vals[-1] - sorted_vals[0]

        if data_range == 0:
            calc_gap_ratio = 0.0
        else:
            calc_gap_ratio = np.max(gaps) / data_range

        results[col] = {
            "min": float(sorted_vals[0]),
            "max": float(sorted_vals[-1]),
            "max_gap_ratio": round(calc_gap_ratio, 4),
            "sufficient_coverage": calc_gap_ratio < max_gap_ratio
        }
    return results

def _check_model_fit(df: pd.DataFrame, input_cols: List[str], outcome_col: str, min_r2_score: float = 0.50) -> Dict:
    """
    Checks if a basic surrogate model can capture a meaningful signal-to-noise relationship.
    Uses a 3rd-degree polynomial and cross-validation to ensure the fit is stable.
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
        "stable_fit": np.mean(scores) > min_r2_score
    }

def _check_bootstrap_convergence(
    df: pd.DataFrame, input_cols: List[str], outcome_col: str,
    n_bootstraps: int = 100, max_avg_width: float = 0.15, max_tail_width: float = 0.30
) -> Dict:
    """
    Evaluates the stability of model predictions across different random sub-samples of the data.
    Ensures that adding or removing points does not wildly change the predicted outcome.
    """
    X = df[input_cols].values
    y = df[outcome_col].values
    n_samples = len(df)

    probe_points = np.percentile(X, [10, 50, 90], axis=0)
    all_predictions = []

    for _ in range(n_bootstraps):
        X_res, y_res = resample(X, y, n_samples=n_samples)
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X_res, y_res)
        preds = model.predict(probe_points)
        all_predictions.append(preds)

    all_predictions = np.array(all_predictions)

    stds = np.std(all_predictions, axis=0)
    means = np.abs(np.mean(all_predictions, axis=0))
    relative_widths = stds / (means + 1e-6)

    avg_rel_width = np.mean(relative_widths)
    max_rel_width = np.max(relative_widths)

    is_converged = avg_rel_width < max_avg_width and max_rel_width < max_tail_width

    return {
        "bootstrap_iterations": n_bootstraps,
        "avg_relative_width": round(avg_rel_width, 4),
        "max_relative_width": round(max_rel_width, 4),
        "avg_converged": bool(avg_rel_width < max_avg_width),
        "max_converged": bool(max_rel_width < max_tail_width),
        "converged": bool(is_converged)
    }


#### Main Function: sample_sufficiency() ####

def sample_sufficiency(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    skip_validation: bool = False,
    max_gap_ratio: float = 0.20,
    min_r2_score: float = 0.50,
    max_avg_width: float = 0.15,
    max_tail_width: float = 0.30
) -> pd.DataFrame:
    """
    Performs a suite of statistical diagnostics to evaluate if the current sample size is sufficient.

    This function tests input space coverage, basic model fit (signal-to-noise),
    and prediction stability via bootstrapping. It uses user-defined thresholds
    to determine if the sampling passes the sufficiency criteria required for reliable PoD analysis.

    Args:
        df (pd.DataFrame): The simulation dataset containing inputs and outcomes.
        input_cols (List[str]): A list of the input parameter column names.
        outcome_col (str): The name of the outcome/signal column.
        skip_validation (bool, optional): If True, skips the initial data cleaning step. Defaults to False.
        max_gap_ratio (float, optional): The maximum allowable gap between data points as a fraction of the total range. Defaults to 0.20.
        min_r2_score (float, optional): The minimum cross-validated R-squared score required to pass the fit test. Defaults to 0.50.
        max_avg_width (float, optional): The maximum allowable average relative width of the bootstrap predictions. Defaults to 0.15.
        max_tail_width (float, optional): The maximum allowable relative width at the tail ends (10th and 90th percentiles) of the predictions. Defaults to 0.30.

    Returns:
        pd.DataFrame: A formatted table detailing the results of each diagnostic test,
                      including the variable tested, the calculated metric, the target threshold,
                      and a boolean 'Pass' status.

    Examples:
        ```python
        import pandas as pd
        from digiqual.diagnostics import sample_sufficiency

        # Assume 'df' is a loaded DataFrame of simulation results
        df = pd.DataFrame({
            'Length': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Signal': [2.1, 4.0, 6.2, 8.1, 9.9, 12.0, 14.1, 15.9, 18.2, 20.0]
        })

        # Run diagnostics with custom stricter thresholds
        results_df = sample_sufficiency(
            df=df,
            input_cols=['Length'],
            outcome_col='Signal',
            max_gap_ratio=0.15,  # Require tighter spacing
            min_r2_score=0.70    # Require a stronger signal fit
        )

        print(results_df)
        ```
    """

    if not skip_validation:
        df_clean, df_removed = validate_simulation(df, input_cols, outcome_col)
        if not df_removed.empty:
            print(f"Note: {len(df_removed)} invalid rows were dropped automatically.")
    else:
        df_clean = df

    if len(df_clean) < 10:
        raise ValidationError(
            f"Insufficient valid data ({len(df_clean)} rows). "
            "Diagnostics require at least 10 valid data points."
        )

    # Pass the custom thresholds into the helpers
    coverage_res = _check_input_coverage(df_clean, input_cols, max_gap_ratio)
    fit_res = _check_model_fit(df_clean, input_cols, outcome_col, min_r2_score)
    boot_res = _check_bootstrap_convergence(df_clean, input_cols, outcome_col, 100, max_avg_width, max_tail_width)

    flat_results = []

    for col, res in coverage_res.items():
        flat_results.append({
            "Test": "Input Coverage",
            "Variable": col,
            "Metric": "Max Gap Ratio",
            "Value": res['max_gap_ratio'],
            "Threshold": f"< {max_gap_ratio:.2f}",
            "Pass": res['sufficient_coverage']
        })

    flat_results.append({
        "Test": "Model Fit (CV)",
        "Variable": outcome_col,
        "Metric": "Mean R2 Score",
        "Value": fit_res['mean_r2_score'],
        "Threshold": f"> {min_r2_score:.2f}",
        "Pass": fit_res['stable_fit']
    })

    flat_results.append({
        "Test": "Bootstrap Convergence",
        "Variable": outcome_col,
        "Metric": "Avg CV (Rel Std Dev)",
        "Value": boot_res['avg_relative_width'],
        "Threshold": f"< {max_avg_width:.2f}",
        "Pass": boot_res['avg_converged']
    })

    flat_results.append({
        "Test": "Bootstrap Convergence",
        "Variable": outcome_col,
        "Metric": "Max CV (Rel Std Dev)",
        "Value": boot_res['max_relative_width'],
        "Threshold": f"< {max_tail_width:.2f}",
        "Pass": boot_res['max_converged']
    })

    return pd.DataFrame(flat_results)
