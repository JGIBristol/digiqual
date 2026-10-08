"""
Default settings shared by the library and the app.

The GUI reads these instead of hard-coding its own copies, so the two can't drift
apart. Changing a value here changes the default everywhere.
"""

# Diagnostic thresholds (see `SimulationStudy.diagnose` / `sample_sufficiency`)
MAX_GAP_RATIO = 0.20
MIN_R2_SCORE = 0.50
MAX_AVG_CV = 0.15
MAX_MAX_CV = 0.30
MAX_ALLOWED_VIF = 5.0

# Model selection
MAX_POLY_DEGREE = 10
N_CV_FOLDS = 10
# Kriging is only fitted up to this many samples (every fit is O(N^3)).
KRIGING_MAX_SAMPLES = 1000

# Uncertainty quantification
DEFAULT_N_BOOT = 1000
CONFIDENCE_LEVELS = (50, 90, 95, 99)
TARGET_PODS = (50, 90, 95, 99)

# Number of thresholds in the pre-computed threshold spectrum
N_THRESHOLD_POINTS = 100
