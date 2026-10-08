import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .defaults import MAX_ALLOWED_VIF, MAX_AVG_CV, MAX_GAP_RATIO, MAX_MAX_CV, MIN_R2_SCORE

logger = logging.getLogger(__name__)

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
        import numpy as np
        import pandas as pd
        # 12 rows, one of which has a non-numeric input
        length = list(np.linspace(1.0, 5.0, 12))
        length[3] = 'BadValue'
        df = pd.DataFrame({'Length': length, 'Signal': np.linspace(0.5, 2.0, 12)})

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

def _check_input_coverage(df: pd.DataFrame, input_cols: List[str], max_gap_ratio: float = MAX_GAP_RATIO) -> Dict:
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

def check_input_coverage(df: pd.DataFrame, input_cols: List[str], max_gap_ratio: float = MAX_GAP_RATIO) -> Dict:
    """
    Largest gap between neighbouring sorted values of each input, as a fraction of its range.

    Public wrapper of the Input Coverage diagnostic, returning per-input details
    (``max_gap_ratio``, ``sufficient_coverage``, the gap location, min and max).
    """
    return _check_input_coverage(df, input_cols, max_gap_ratio)


def bootstrap_convergence_trace(
    make_model,
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 100,
    percentiles=(10, 50, 90),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Running bootstrap stability of a model's predictions, for a convergence plot.

    Refits ``make_model()`` to ``n_boot`` bootstrap resamples of ``(X, y)`` and
    predicts at the given percentiles of the inputs. After each resample it records
    the average and largest prediction spread so far, measured with `relative_spread`
    (as the Bootstrap Convergence diagnostic does).

    Args:
        make_model (Callable[[], estimator]): Returns a fresh, unfitted model.
        X (np.ndarray): Training inputs.
        y (np.ndarray): Training responses.
        n_boot (int): Number of bootstrap resamples.
        percentiles: Input percentiles used as probe points.
        seed (int): Seed for the resampling.

    Returns:
        dict: ``iterations``, ``running_avg`` and ``running_max`` arrays.
    """
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    y = np.asarray(y, dtype=float)
    probe = np.percentile(X, list(percentiles), axis=0)
    rng = np.random.default_rng(seed)

    preds, running_avg, running_max = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        preds.append(make_model().fit(X[idx], y[idx]).predict(probe))
        widths = relative_spread(np.array(preds), y)
        running_avg.append(float(np.mean(widths)))
        running_max.append(float(np.max(widths)))
    return {
        "iterations": np.arange(1, n_boot + 1),
        "running_avg": np.array(running_avg),
        "running_max": np.array(running_max),
    }


def _check_model_fit(df: pd.DataFrame, input_cols: List[str], outcome_col: str, min_r2_score: float = MIN_R2_SCORE) -> Dict:
    """
    Checks if a basic surrogate model can capture a meaningful signal-to-noise relationship.
    Uses a 3rd-degree polynomial and cross-validation to ensure the fit is stable.
    """
    X = df[input_cols]
    y = df[outcome_col]

    model = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())

    k = 10 if len(df) > 50 else 5
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    return {
        "model_type": "Polynomial (deg=3)",
        "cv_folds": k,
        "mean_r2_score": round(np.mean(scores), 4),
        "stable_fit": np.mean(scores) > min_r2_score
    }

def relative_spread(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Spread of bootstrap predictions at each probe point, relative to the spread of the data.

    ``std(predictions) / std(y)``, per probe point. Dividing by the scatter of the
    observed response, rather than by the size of the mean prediction, makes the
    measure independent of where the response's zero is: a signal in dB that sits
    near 0, or one with a large constant offset, is judged on the same footing.
    """
    predictions = np.asarray(predictions, dtype=float)
    y_scale = float(np.std(y))
    if y_scale <= 0:
        y_scale = 1.0
    return np.std(predictions, axis=0) / y_scale


def _check_bootstrap_convergence(
    df: pd.DataFrame, input_cols: List[str], outcome_col: str,
    n_bootstraps: int = 100, max_avg_cv: float = MAX_AVG_CV, max_max_cv: float = MAX_MAX_CV
) -> Dict:
    """
    Evaluates the stability of model predictions across different random sub-samples of the data.
    Ensures that adding or removing points does not wildly change the predicted outcome.

    A quadratic polynomial is refitted to bootstrap resamples and evaluated at the 10th,
    50th and 90th percentiles of the inputs. The spread of those predictions is divided
    by the standard deviation of the response (see `relative_spread`).
    """
    X = df[input_cols].values
    y = df[outcome_col].values
    n_samples = len(df)

    probe_points = np.percentile(X, [10, 50, 90], axis=0)
    all_predictions = []

    # Seeded so repeated diagnostics on the same data give the same result
    rng = np.random.default_rng(42)

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, n_samples, replace=True)
        X_res, y_res = X[idx], y[idx]

        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
        model.fit(X_res, y_res)
        preds = model.predict(probe_points)
        all_predictions.append(preds)

    all_predictions = np.array(all_predictions)

    relative_widths = relative_spread(all_predictions, y)

    avg_rel_width = np.mean(relative_widths)
    max_rel_width = np.max(relative_widths)

    is_converged = avg_rel_width < max_avg_cv and max_rel_width < max_max_cv

    return {
        "bootstrap_iterations": n_bootstraps,
        "avg_relative_width": round(avg_rel_width, 4),
        "max_relative_width": round(max_rel_width, 4),
        "avg_converged": bool(avg_rel_width < max_avg_cv),
        "max_converged": bool(max_rel_width < max_max_cv),
        "converged": bool(is_converged)
    }




def _check_collinearity(df: pd.DataFrame, input_cols: List[str], max_allowed_vif: float = MAX_ALLOWED_VIF) -> Dict[str, float]:
    """
    Computes the Variance Inflation Factor (VIF) of each input against the others.

    Returns the VIF per input; the pass/fail comparison against ``max_allowed_vif``
    is made by `sample_sufficiency`, which reports it.
    """
    vifs = {}
    if len(input_cols) > 1:
        for col in input_cols:
            other_cols = [c for c in input_cols if c != col]
            X = df[other_cols].values
            y = df[col].values

            try:
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                if r2 >= 1.0 - 1e-10:
                    vif = float('inf')
                else:
                    vif = 1.0 / (1.0 - r2)
            except Exception:
                vif = float('inf')

            vifs[col] = round(vif, 4)
    else:
        for col in input_cols:
            vifs[col] = 1.0

    return vifs


#### Main Function: sample_sufficiency() ####

def sample_sufficiency(
    df: pd.DataFrame,
    input_cols: List[str],
    outcome_col: str,
    skip_validation: bool = False,
    max_gap_ratio: float = MAX_GAP_RATIO,
    min_r2_score: float = MIN_R2_SCORE,
    max_avg_cv: float = MAX_AVG_CV,
    max_max_cv: float = MAX_MAX_CV,
    max_allowed_vif: float = MAX_ALLOWED_VIF
) -> pd.DataFrame:
    """
    Performs a suite of statistical diagnostics to evaluate if the current sample size is sufficient.

    This function tests input space coverage, basic model fit (signal-to-noise),
    prediction stability via bootstrapping, and multicollinearity. It uses user-defined thresholds
    to determine if the sampling passes the sufficiency criteria required for reliable PoD analysis.

    Args:
        df (pd.DataFrame): The simulation dataset containing inputs and outcomes.
        input_cols (List[str]): A list of the input parameter column names.
        outcome_col (str): The name of the outcome/signal column.
        skip_validation (bool, optional): If True, skips the initial data cleaning step. Defaults to False.
        max_gap_ratio (float, optional): The maximum allowable gap between data points as a fraction of the total range. Defaults to 0.20.
        min_r2_score (float, optional): The minimum cross-validated R-squared score required to pass the fit test. Defaults to 0.50.
        max_avg_cv (float, optional): The maximum allowable average spread of the bootstrap predictions, relative to the standard deviation of the response. Defaults to 0.15.
        max_max_cv (float, optional): The maximum allowable relative width at any of the three probe points (10th, 50th and 90th percentiles of the inputs). Defaults to 0.30.
        max_allowed_vif (float, optional): The maximum allowable Variance Inflation Factor (VIF) to detect multicollinearity. Defaults to 5.0.

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
            logger.warning(f"Note: {len(df_removed)} invalid rows were dropped automatically.")
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
    boot_res = _check_bootstrap_convergence(df_clean, input_cols, outcome_col, 100, max_avg_cv, max_max_cv)
    vif_res = _check_collinearity(df_clean, input_cols, max_allowed_vif)

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
        "Threshold": f"< {max_avg_cv:.2f}",
        "Pass": boot_res['avg_converged']
    })

    flat_results.append({
        "Test": "Bootstrap Convergence",
        "Variable": outcome_col,
        "Metric": "Max CV (Rel Std Dev)",
        "Value": boot_res['max_relative_width'],
        "Threshold": f"< {max_max_cv:.2f}",
        "Pass": boot_res['max_converged']
    })

    for col, vif in vif_res.items():
        flat_results.append({
            "Test": "Collinearity Check",
            "Variable": col,
            "Metric": "VIF",
            "Value": vif,
            "Threshold": f"< {max_allowed_vif:.2f}",
            "Pass": bool(vif <= max_allowed_vif)
        })

    return pd.DataFrame(flat_results)

