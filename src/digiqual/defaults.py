"""
Default settings shared by the library and the app.

The GUI reads these instead of hard-coding its own copies, so the two can't drift
apart. Changing a value here changes the default everywhere.
"""

__all__ = [
    "MAX_GAP_RATIO",
    "MIN_R2_SCORE",
    "MAX_AVG_CV",
    "MAX_MAX_CV",
    "MAX_ALLOWED_VIF",
    "MAX_POLY_DEGREE",
    "N_CV_FOLDS",
    "KRIGING_MAX_SAMPLES",
    "DEFAULT_N_BOOT",
    "CONFIDENCE_LEVELS",
    "TARGET_PODS",
    "MC_SAMPLES",
    "BOOTSTRAP_MC_SAMPLES",
    "N_THRESHOLD_POINTS",
]

# Diagnostic thresholds (see `SimulationStudy.diagnose` / `sample_sufficiency`)
MAX_GAP_RATIO = 0.20
"""Input coverage: largest allowed gap, as a fraction of each input's range."""
MIN_R2_SCORE = 0.50
"""Model fit: minimum 10-fold cross-validated R² (also the Model Diagnostics pass mark)."""
MAX_AVG_CV = 0.15
"""Bootstrap stability: largest allowed average prediction spread (relative to the response std)."""
MAX_MAX_CV = 0.30
"""Bootstrap stability: largest allowed prediction spread at any probe point."""
MAX_ALLOWED_VIF = 5.0
"""Collinearity: largest allowed Variance Inflation Factor of any input."""

# Model selection
MAX_POLY_DEGREE = 10
"""Highest polynomial degree among the candidate models."""
N_CV_FOLDS = 10
"""Folds used to cross-validate the candidate models."""
# Kriging is only fitted up to this many samples (every fit is O(N^3)).
KRIGING_MAX_SAMPLES = 1000
"""Kriging is only fitted up to this many samples (every fit is O(N³))."""

# Uncertainty quantification
DEFAULT_N_BOOT = 1000
"""Default number of bootstrap iterations for confidence bounds."""
CONFIDENCE_LEVELS = (50, 90, 95, 99)
"""Confidence levels (%) whose bounds every bootstrap computes."""
TARGET_PODS = (50, 90, 95, 99)
"""Target PoDs (%) in the reliability (aX/Y) table."""

# Monte Carlo samples for integrating out nuisance parameters: the PoD curve itself,
# and each bootstrap refit. Fewer per refit keeps a 1000-iteration bootstrap fast; the
# bounds then carry slightly more MC noise than the curve (so the curve can sit a
# hair outside a narrow band where the nuisances matter little).
MC_SAMPLES = 3000
"""Monte Carlo samples used to integrate out nuisance parameters for the PoD curve."""
BOOTSTRAP_MC_SAMPLES = 500
"""Monte Carlo samples per bootstrap refit (fewer, to keep the bootstrap fast)."""

# Number of thresholds in the pre-computed threshold spectrum
N_THRESHOLD_POINTS = 100
"""Number of thresholds in the pre-computed threshold spectrum (`compute_pod_spectrum`)."""
