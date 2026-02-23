import pytest
import pandas as pd
import numpy as np
from digiqual.diagnostics import (
    sample_sufficiency,
    _check_input_coverage,
    _check_model_fit,
    _check_bootstrap_convergence
)

# --- Fixtures ---

@pytest.fixture
def basic_df():
    """Creates a 'perfect' dataset that should pass all checks."""
    np.random.seed(123)
    n = 100
    df = pd.DataFrame({
        'Length': np.linspace(0, 10, n),
        'Angle': np.linspace(-90, 90, n),
        'Roughness': np.linspace(0, 1, n)
    })

    df['Signal'] = 2 * df['Length'] + 10
    return df


# --- Part 1: Unit Tests for Helper Functions ---

def test_check_input_coverage_pass():
    """Test that evenly spaced data passes coverage."""
    # 50 points evenly spaced between 0 and 10.
    df = pd.DataFrame({'A': np.linspace(0, 10, 50)})

    res = _check_input_coverage(df, ['A'])

    # FIX: Use implicit truthiness (Satisfies Ruff E712)
    assert res['A']['sufficient_coverage']
    assert res['A']['max_gap_ratio'] < 0.2

def test_check_input_coverage_fail():
    """Test that data with a large gap fails coverage."""
    # Data missing the middle chunk (3 to 7)
    df = pd.DataFrame({'A': [0, 1, 2, 8, 9, 10]})

    res = _check_input_coverage(df, ['A'])

    assert not res['A']['sufficient_coverage']
    assert res['A']['max_gap_ratio'] > 0.2

def test_check_model_fit_pass():
    """Test that a clean quadratic relationship yields a high R2."""
    np.random.seed(123)
    df = pd.DataFrame({'A': np.linspace(0, 10, 50)})
    df['Signal'] = df['A']**2 + 5

    res = _check_model_fit(df, ['A'], 'Signal')

    assert res['stable_fit']
    assert res['mean_r2_score'] > 0.95

def test_check_model_fit_fail():
    """Test that pure noise yields a low/negative R2."""
    np.random.seed(123)
    df = pd.DataFrame({
        'A': np.linspace(0, 10, 50),
        'Signal': np.random.normal(0, 10, 50) # Pure noise
    })

    res = _check_model_fit(df, ['A'], 'Signal')

    assert not res['stable_fit']
    assert res['mean_r2_score'] < 0.5

def test_check_bootstrap_convergence_pass():
    """Test that Large N + Low Noise = Narrow Confidence Interval."""
    np.random.seed(123)
    n = 500
    df = pd.DataFrame({'A': np.linspace(0, 10, n)})
    df['Signal'] = 3 * df['A'] + np.random.normal(0, 0.1, n)

    res = _check_bootstrap_convergence(df, ['A'], 'Signal')

    assert res['converged']
    # FIX: Changed 'relative_ci_width' to 'max_relative_width'
    assert res['max_relative_width'] < 0.10

def test_check_bootstrap_convergence_fail():
    """Test that Small N + High Noise = Wide Confidence Interval."""
    np.random.seed(123)
    n = 20
    df = pd.DataFrame({'A': np.linspace(0, 10, n)})
    df['Signal'] = 3 * df['A'] + np.random.normal(0, 10, n)

    res = _check_bootstrap_convergence(df, ['A'], 'Signal')

    assert not res['converged']
    # FIX: Changed 'relative_ci_width' to 'max_relative_width'
    assert res['max_relative_width'] > 0.10



def test_bootstrap_detects_heteroskedasticity():
    """Test that convergence fails if the 'tail' is unstable even if the mean is okay."""
    np.random.seed(123)
    n = 100
    X = np.linspace(0, 10, n)

    # Noise increases significantly with X (Heteroskedasticity)
    noise = np.random.normal(0, 1) * (X / 2)
    df = pd.DataFrame({'A': X, 'Signal': 2 * X + noise + 10})

    res = _check_bootstrap_convergence(df, ['A'], 'Signal')

    # The 10th percentile might be stable, but the 90th should be wild
    assert res['probe_results']['90th_percentile_rel_width'] > res['probe_results']['10th_percentile_rel_width']

    # Ensure the max_relative_width is the one driving the 'converged' status
    assert res['max_relative_width'] == max(res['probe_results'].values())


# --- Part 2: Integration Tests for Main Function ---

def test_sample_sufficiency_happy_path(basic_df):
    """Test that the main function returns a clean pass for good data."""
    results = sample_sufficiency(basic_df, ['Length', 'Angle'], 'Signal')

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert results['Pass'].all()

def test_sample_sufficiency_fail_coverage():
    """Test that a large gap triggers a failure."""
    # N=12 to pass validation, but gap remains large
    group1 = np.linspace(0, 2, 6)
    group2 = np.linspace(8, 10, 6)

    df = pd.DataFrame({
        'A': np.concatenate([group1, group2]),
        'Signal': np.random.rand(12) + 10
    })

    results = sample_sufficiency(df, ['A'], 'Signal')

    assert not results['Pass'].all()

    coverage_fail = results[
        (results['Test'] == 'Input Coverage') &
        (results['Variable'] == 'A')
    ]

    assert not coverage_fail['Pass'].values[0]

def test_sample_sufficiency_fail_stability():
    """Test that noisy data fails Model Fit."""
    np.random.seed(123)
    df = pd.DataFrame({
        'A': np.linspace(0, 10, 20),
        'Signal': np.random.normal(0, 10, 20)
    })
    df['Signal'] = df['Signal'].abs() + 1

    results = sample_sufficiency(df, ['A'], 'Signal')

    fit_row = results[results['Test'] == "Model Fit (CV)"]
    assert not fit_row['Pass'].values[0]
