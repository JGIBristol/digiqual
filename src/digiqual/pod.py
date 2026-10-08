import logging
import os
import warnings
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kriging provenance tags
#
# The Kriging surrogate follows two papers and the UQLab toolbox they used.
# Comments and docstrings below cite where each choice comes from:
#
#   [M25]       Malkiel, Croxford & Wilcox (2025), "A generalized method for the
#               reliability assessment of safety-critical inspection",
#               Proc. R. Soc. A 481: 20240654.
#   [M25-code]  The MATLAB reference code for [M25]
#               (materials/malkiel25/Code_Generalized_a_hat_vs_a_method.m).
#   [M26]       Malkiel, Croxford & Wilcox (2026), "A comprehensive investigation
#               of flexible and multi-dimensional simulation-based PoD analysis",
#               NDT&E Int. 159: 103596.
#   [UQLab]     A UQLab Kriging default (UQLab_Rel2.1.0, Kriging module). Both
#               papers used UQLab with its defaults, so these are inherited.
#   [digiqual]  A deliberate deviation from the references. The rationale is
#               given inline and in docs/kriging_metamodeling.qmd.
# ---------------------------------------------------------------------------

# Numerical jitter added to the Kriging covariance diagonal. Observation noise is
# NOT represented here: it is learned by the WhiteKernel term in every candidate
# kernel (see `build_kriging_candidate_kernels`), so this only needs to be large
# enough to keep the Cholesky factorisation stable. [UQLab] uses the same idea
# (Corr.Nugget = 1e-10).
KRIGING_JITTER = 1e-10

# Optimiser restarts for the Kriging likelihood. [UQLab] uses a hybrid genetic
# algorithm; [digiqual] uses scikit-learn's multi-start L-BFGS-B instead.
KRIGING_RESTARTS_FULL = 10
KRIGING_RESTARTS_CV = 5


class ScaledGaussianProcessRegressor(GaussianProcessRegressor):
    """
    Gaussian Process regressor that standardises its inputs before fitting.

    [UQLab] (``Scaling = true``, the default used by [M25-code] and [M26])
    standardises every input with the experimental-design mean and standard
    deviation, ``u = (x - mean(x)) / std(x)``, and fits the GP in ``u``. This
    makes the length scales dimensionless (in units of each input's standard
    deviation), so the length-scale bounds and initial values mean the same
    thing whatever the physical units of the inputs.

    The scaling is computed in ``fit`` unless it has been frozen with
    `freeze_scaling` (used for bootstrap refits, so every resample shares the
    full-data scaling, and therefore the full-data length scales keep their
    meaning). ``predict`` and ``sample_y`` accept inputs in physical units.
    Every other behaviour, including all constructor arguments, is inherited
    from scikit-learn's `GaussianProcessRegressor`.
    """

    def freeze_scaling(self, x_mean: np.ndarray, x_std: np.ndarray) -> "ScaledGaussianProcessRegressor":
        """Fixes the input scaling so that ``fit`` reuses it instead of recomputing it."""
        self._frozen_scaling = (np.asarray(x_mean, dtype=float), np.asarray(x_std, dtype=float))
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Maps physical inputs to the standardised inputs the kernel operates on."""
        return (np.atleast_2d(X) - self.x_mean_) / self.x_std_

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        frozen = getattr(self, "_frozen_scaling", None)
        if frozen is not None:
            self.x_mean_, self.x_std_ = frozen
        else:
            self.x_mean_ = X.mean(axis=0)
            std = X.std(axis=0)
            # A constant input carries no information; keep it finite rather than divide by zero.
            self.x_std_ = np.where(std > 0, std, 1.0)
        return super().fit(self.transform_X(X), y)

    def predict(self, X, return_std=False, return_cov=False):
        return super().predict(self.transform_X(X), return_std=return_std, return_cov=return_cov)

    def sample_y(self, X, n_samples=1, random_state=0):
        return super().sample_y(self.transform_X(X), n_samples=n_samples, random_state=random_state)


def build_kriging_candidate_kernels(n_features: int) -> dict[str, Any]:
    """
    Builds the candidate covariance kernels evaluated for the Kriging surrogate.

    Every candidate has the form ``C * R(u, u') + WhiteKernel``, evaluated on
    standardised inputs ``u`` (see `ScaledGaussianProcessRegressor`):

    - ``C`` is the process variance sigma^2 ([M26] Eq. 3).
    - ``R`` is the correlation function, with one length scale per input
      (anisotropic, [M26] Eq. 5; [UQLab] ``Corr.Type = 'Ellipsoidal'``).
      The families are those compared in [M26] Sec. 4.3.1: exponential
      (Matérn 1/2), Matérn 3/2, Matérn 5/2 and Gaussian. [digiqual] omits
      UQLab's "linear" correlation because scikit-learn does not provide it.
    - ``WhiteKernel`` is the observation-noise (nugget) variance tau^2, learned
      by maximum likelihood alongside the length scales. This follows
      [M25-code] (``Regression.SigmaNSQ = 'auto'``). [M26] instead treats the
      response as deterministic and interpolates it exactly; a learned nugget
      reduces to that case when the data has no scatter.

    Learning the nugget is essential when the chosen inputs do not fully
    determine the response (e.g. a model trained on Area and Offset when the
    response also depends on flaw shape). If the noise were fixed too small, the
    likelihood would be maximised by shrinking a length scale until the surface
    interpolates the scatter. That produces a spiky surface whose slices revert
    to the prior mean away from data and whose bootstrap refits are unstable.

    Length-scale initial value (1) and lower bound (1e-3) follow [UQLab]. The
    upper bound is 1e3 rather than UQLab's 10 [digiqual], so that an input with
    no influence can take a length scale far longer than its range (an
    effectively flat direction) instead of sitting on the bound.

    The kernels are intended for use with ``normalize_y=True``, so ``C`` and the
    noise level are expressed relative to the variance of the training response.

    Args:
        n_features (int): Number of input dimensions.

    Returns:
        dict[str, Kernel]: Mapping of human-readable kernel name to an unfitted
        scikit-learn kernel.
    """
    ls = np.ones(n_features)
    ls_bounds = (1e-3, 1e3)

    def _amplitude() -> C:
        return C(1.0, (1e-5, 1e6))

    def _noise() -> WhiteKernel:
        return WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e1))

    return {
        'Exponential (Matern 1/2)': _amplitude() * Matern(length_scale=ls, length_scale_bounds=ls_bounds, nu=0.5) + _noise(),
        'Matern 3/2': _amplitude() * Matern(length_scale=ls, length_scale_bounds=ls_bounds, nu=1.5) + _noise(),
        'Matern 5/2': _amplitude() * Matern(length_scale=ls, length_scale_bounds=ls_bounds, nu=2.5) + _noise(),
        'Gaussian (RBF)': _amplitude() * RBF(length_scale=ls, length_scale_bounds=ls_bounds) + _noise(),
    }


def build_kriging_gpr(kernel: Any, n_restarts: int = KRIGING_RESTARTS_FULL) -> ScaledGaussianProcessRegressor:
    """
    Builds an unfitted Kriging model with the production settings.

    Standardised inputs [UQLab], standardised response (``normalize_y=True``)
    [digiqual: scikit-learn has no ordinary-Kriging trend, so the prior mean is
    the sample mean rather than a GLS-estimated constant as in [M26] Eq. 6],
    hyperparameters by maximum likelihood [M26] Sec. 2.1.2, and multi-start
    L-BFGS-B [digiqual].
    """
    return ScaledGaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=KRIGING_JITTER,
        normalize_y=True,
        random_state=42,
    )


def build_fixed_kernel_gpr(
    kernel: Any,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
) -> ScaledGaussianProcessRegressor:
    """
    Builds a Kriging model that reuses already-optimised hyperparameters.

    Used wherever a fitted Kriging surrogate is refitted to resampled data
    (PoD bootstrap confidence bounds, the GUI bootstrap convergence trace).
    The optimiser is disabled, so only the posterior is recomputed, and the
    settings match the production model: ``normalize_y=True`` and only
    numerical jitter on the diagonal, because the noise variance is already
    part of ``kernel`` (its WhiteKernel term).

    [M25] Sec. 2i re-estimates the selected model's parameters on every
    bootstrap resample. [digiqual] freezes the Kriging hyperparameters instead,
    because re-optimising them 1000 times is too slow. The bootstrap interval
    therefore omits hyperparameter uncertainty.

    The input scaling should be frozen too, so that the length scales keep the
    meaning they had in the full-data fit. Pass the full-data ``x_mean`` and
    ``x_std``, or pass the model's ``model_params_`` dictionary as ``kernel``.

    Args:
        kernel (Kernel | dict): A fitted kernel (``gpr.kernel_``), or the
            ``model_params_`` dict of a Kriging model from
            `fit_all_robust_mean_models` (keys ``kernel``, ``x_mean``, ``x_std``).
        x_mean (np.ndarray, optional): Full-data input means.
        x_std (np.ndarray, optional): Full-data input standard deviations.

    Returns:
        ScaledGaussianProcessRegressor: An unfitted regressor ready for ``.fit()``.
    """
    if isinstance(kernel, dict):
        x_mean = kernel.get('x_mean', x_mean)
        x_std = kernel.get('x_std', x_std)
        kernel = kernel['kernel']
    gpr = ScaledGaussianProcessRegressor(
        kernel=kernel, alpha=KRIGING_JITTER, normalize_y=True, optimizer=None
    )
    if x_mean is not None and x_std is not None:
        gpr.freeze_scaling(x_mean, x_std)
    return gpr


def _kriging_loo_matrix(gpr: GaussianProcessRegressor, X_2d: np.ndarray, y: np.ndarray):
    """
    Returns the LOO matrix B and the response in the kernel's working units.

    B is the top-left m x m block of the inverse of the augmented matrix
    S = [[K + alpha*I, F], [F^T, 0]] with F = 1 (ordinary Kriging), as in
    [M26] Eq. 12-13 (after Dubrule 1983). [UQLab] computes the same matrix in
    ``uq_Kriging_calc_KFold`` as R^-1 (I - F (F^T R^-1 F)^-1 F^T R^-1).
    """
    y_flat = np.asarray(y, dtype=np.float64).flatten()
    m = len(y_flat)

    # Work in the same (possibly normalised) units the kernel was optimised in.
    if getattr(gpr, "normalize_y", False):
        y_mean = float(np.ravel(gpr._y_train_mean)[0])
        y_scale = float(np.ravel(gpr._y_train_std)[0])
    else:
        y_mean, y_scale = 0.0, 1.0
    y_work = (y_flat - y_mean) / y_scale

    X_work = gpr.transform_X(X_2d) if hasattr(gpr, "transform_X") else np.atleast_2d(X_2d)
    K = gpr.kernel_(X_work)
    alpha = gpr.alpha if isinstance(gpr.alpha, (int, float)) else 1e-6

    S = np.zeros((m + 1, m + 1))
    S[:m, :m] = K + np.eye(m) * alpha
    S[:m, m] = 1.0
    S[m, :m] = 1.0
    B_mm = np.linalg.inv(S)[:m, :m]
    return B_mm, y_work, y_mean, y_scale


def compute_kriging_loo_residuals(
    gpr: GaussianProcessRegressor,
    X_2d: np.ndarray,
    y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes exact LOO predictions, variances, standardized residuals e_i,
    and the outlier factor gamma, following Malkiel et al. (2026).

    With B the top-left block of S^-1, S = [[K + alpha*I, F], [F^T, 0]]:

    - LOO mean ([M26] Eq. 10): mu_{-i} = -sum_{j != i} (B_ij / B_ii) y_j
    - LOO variance ([M26] Eq. 11): sigma_{-i}^2 = 1 / B_ii
    - Standardised residual ([M26] Eq. 19): e_i = (y_i - mu_{-i}) / sigma_{-i}
    - Outlier factor ([M26] Sec. 2.2.5.2): gamma = max(1, max|e_i| / 3)

    As in [M26], the hyperparameters are those fitted to the full data set; they
    are not re-optimised with point i removed.

    K is the fitted kernel evaluated at the training inputs, so for the models built by
    `fit_all_robust_mean_models` it already includes the learned noise (WhiteKernel)
    variance on its diagonal. The LOO variance is therefore the predictive variance of a
    new noisy observation, which is the correct scale for standardizing observed residuals.

    In [M26], gamma scales the Kriging interpolation standard deviation used to draw
    the GP sample paths for the PoD uncertainty bound (Fig. 10b shows the residuals
    divided by gamma, so the worst one lands on 3). digiqual computes its PoD bounds
    by bootstrap ([M25]) instead, so gamma is reported as a diagnostic and is not
    applied to the PoD [digiqual].

    If the model was fitted with ``normalize_y=True``, the kernel hyperparameters are in
    normalised units. The calculation is then done on the normalised response and the
    LOO means and standard deviations are converted back to the original units. Inputs
    are standardised first if the model does so (`ScaledGaussianProcessRegressor`).

    Args:
        gpr (GaussianProcessRegressor): A fitted scikit-learn Gaussian Process model.
        X_2d (np.ndarray): 2D matrix of input training coordinates (N_samples, N_features).
        y (np.ndarray): 1D array of observed responses (N_samples,).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - loo_means: Array of Leave-One-Out predicted mean responses.
            - loo_stds: Array of Leave-One-Out predicted standard deviations.
            - std_residuals: Array of standardized LOO residuals e_i = (y_i - mu_{-i}) / sigma_{-i}.
            - gamma: Outlier factor gamma = max(1.0, max|e_i| / 3.0).

    Examples:
        ```python
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from digiqual.pod import compute_kriging_loo_residuals

        X = np.linspace(0, 5, 20).reshape(-1, 1)
        y = 2.0 * X.flatten() + np.random.normal(0, 0.1, 20)

        gpr = GaussianProcessRegressor()
        gpr.fit(X, y)

        loo_means, loo_stds, std_res, gamma = compute_kriging_loo_residuals(gpr, X, y)
        print(f"Outlier factor gamma: {gamma:.3f}")
        ```
    """
    y_flat = np.asarray(y, dtype=np.float64).flatten()
    m = len(y_flat)
    X_2d = np.atleast_2d(X_2d)

    try:
        B_mm, y_work, y_mean, y_scale = _kriging_loo_matrix(gpr, X_2d, y_flat)
        diag_B = np.diag(B_mm)
        diag_B = np.where(np.abs(diag_B) < 1e-12, 1e-12, diag_B)

        # [M26] Eq. 10, written as y_i - (B y)_i / B_ii to vectorise the sum over j != i.
        loo_means = y_work - (B_mm @ y_work) / diag_B
        # [M26] Eq. 11
        loo_stds = np.sqrt(np.maximum(1e-10, 1.0 / diag_B))

        loo_means = loo_means * y_scale + y_mean
        loo_stds = loo_stds * y_scale

    except np.linalg.LinAlgError:
        loo_means = y_flat.copy()
        loo_stds = np.ones(m) * (np.std(y_flat) if np.std(y_flat) > 0 else 1.0)

    residuals = y_flat - loo_means
    std_residuals = residuals / np.maximum(loo_stds, 1e-10)

    max_abs_e = float(np.max(np.abs(std_residuals))) if len(std_residuals) > 0 else 0.0
    gamma = max(1.0, max_abs_e / 3.0)

    return loo_means, loo_stds, std_residuals, gamma


def compute_kriging_group_loo_means(
    gpr: GaussianProcessRegressor,
    X_2d: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """
    Leave-one-group-out Kriging means, for data that contains repeated points.

    Each group (e.g. all copies of one original observation in a bootstrap
    resample) is removed together, so a point is never predicted from its own
    duplicate. With I the indices of a group and B as in
    `compute_kriging_loo_residuals`, the prediction is
    mu_{-I} = -(B_II)^-1 B_{I,rest} y_rest, the block form used by [UQLab]
    ``uq_Kriging_calc_KFold``. It reduces to [M26] Eq. 10 when every group has
    one member.

    Args:
        gpr (GaussianProcessRegressor): A fitted Kriging model.
        X_2d (np.ndarray): Training inputs (N_samples, N_features).
        y (np.ndarray): Training responses (N_samples,).
        groups (np.ndarray): Integer group label per sample.

    Returns:
        np.ndarray: The leave-one-group-out mean for every sample, in response units.
    """
    y_flat = np.asarray(y, dtype=np.float64).flatten()
    groups = np.asarray(groups)
    try:
        B_mm, y_work, y_mean, y_scale = _kriging_loo_matrix(gpr, np.atleast_2d(X_2d), y_flat)
    except np.linalg.LinAlgError:
        return y_flat.copy()

    By = B_mm @ y_work
    loo = np.empty_like(y_work)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        # -(B_II)^-1 B_{I,rest} y_rest  ==  y_I - (B_II)^-1 (B y)_I
        loo[idx] = y_work[idx] - np.linalg.solve(B_mm[np.ix_(idx, idx)], By[idx])
    return loo * y_scale + y_mean

#### Mean Model - Robust Regression (Polynomial + Kriging) ####

def select_cv_winner(
    cv_scores: dict[tuple[str, Any], float],
    cv_se: dict[tuple[str, Any], float] | None = None,
) -> tuple[str, Any]:
    """
    Picks the expectation model using the one-standard-error rule.

    [M25] Sec. 4b(i) (Fig. 5) chose the 3rd-order polynomial over the 4th- and
    5th-order polynomials and Kriging, whose 10-fold CV errors were almost
    identical, "since it is the simplest one among those with smaller errors";
    [M25-code] records the standard error of each CV estimate
    (``std(fold MSEs) / sqrt(k)``). [digiqual] turns that judgement into a fixed
    rule: find the model with the lowest mean CV MSE, then choose the *simplest*
    model whose CV MSE is no more than one standard error above it. Complexity
    order is Polynomial 1, 2, ..., then Kriging (the most flexible).

    Args:
        cv_scores (dict): ``(model_type, params)`` -> mean CV MSE.
        cv_se (dict, optional): ``(model_type, params)`` -> standard error of that
            CV MSE. Without it, the strict minimum is returned.

    Returns:
        tuple: The key of the selected model.
    """
    best_key = min(cv_scores, key=cv_scores.get)
    if not cv_se or not np.isfinite(cv_se.get(best_key, np.nan)):
        return best_key
    threshold = cv_scores[best_key] + cv_se[best_key]

    def complexity(key):
        model_type, params = key
        return (0, params) if model_type == 'Polynomial' else (1, 0)

    for key in sorted(cv_scores, key=complexity):
        if cv_scores[key] <= threshold:
            return key
    return best_key


def fit_all_robust_mean_models(
    X: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 10
) -> tuple[dict[tuple[str, Any], Any], dict[tuple[str, Any], float], tuple[str, Any]]:
    """
    Fits all polynomial models (and optionally Kriging) and returns them for caching.

    Instead of fitting models, selecting the best, and throwing the rest away,
    this function evaluates all candidates via k-fold Cross Validation (CV) and
    then fits *every* model to the full dataset. This allows the application to
    instantly swap between different model structures without recalculating.

    The procedure combines both references:

    1. Polynomials of degree 1 to ``max_degree`` are scored by k-fold CV MSE
       ([M25] Eq. 2.7-2.8, k = 10).
    2. Kriging (only for N <= 1000, [digiqual]) is fitted once to the full data
       for each kernel from `build_kriging_candidate_kernels`. The kernel with the
       lowest normalised LOO MSE ([M26] Eq. 10-14, hyperparameters from the full
       fit) is kept, as in [M26] Sec. 2.2.2.
    3. The chosen kernel alone is then scored by the same k-fold CV as the
       polynomials, with its hyperparameters re-optimised in every fold, so that
       Kriging and the polynomials are compared on equal terms ([M25] Fig. 5).
    4. The overall winner is picked by the one-standard-error rule in
       `select_cv_winner` ([M25] Sec. 4b(i), formalised by [digiqual]).

    The fitted models carry the extra attributes ``cv_se_`` (standard error of
    every CV score) and, for Kriging, ``kernel_loo_scores_``, ``best_kernel_name_``,
    ``loo_means_``, ``loo_residuals_`` and ``outlier_scale_factor_``.

    Args:
        X (np.ndarray): 1D array or 2D matrix of input variable values.
        y (np.ndarray): 1D array of outcome values (e.g., signal response).
        max_degree (int, optional): The maximum polynomial degree to test. Defaults to 10.
        n_folds (int, optional): Number of folds for Cross Validation. Defaults to 10.

    Returns:
        tuple[dict, dict, tuple]:
            - `fitted_models`: A dictionary mapping a key like `('Polynomial', 3)` to the fully trained scikit-learn model.
            - `cv_scores`: A dictionary mapping the same keys to their Cross-Validation MSE scores.
            - `cv_winner_key`: The key of the model selected by the one-standard-error rule.

    Examples:
        ```python
        import numpy as np
        X = np.linspace(0, 10, 50)
        y = 3 * X + np.random.normal(0, 1, 50)

        models, scores, best_key = fit_all_robust_mean_models(X, y)

        print(f"The best model was: {best_key}")

        # Instantly retrieve the degree-4 polynomial without refitting
        poly_4 = models[('Polynomial', 4)]
        ```
    """
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)

    y = np.asarray(y, dtype=np.float64).flatten()

    fitted_models = {}
    cv_scores = {}
    cv_se = {}
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _record_cv(key, neg_fold_scores):
        fold_mse = -np.asarray(neg_fold_scores)
        cv_scores[key] = float(np.mean(fold_mse))  # [M25] Eq. 2.8
        # Standard error of the CV estimate, as computed in [M25-code]
        cv_se[key] = float(np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))) if len(fold_mse) > 1 else float('nan')

    # 1. Evaluate & Fit Polynomials ([M25] Sec. 2c; [digiqual] Ridge on standardised features instead of OLS)
    for d in range(1, max_degree + 1):
        model = make_pipeline(
            PolynomialFeatures(degree=d),
            StandardScaler(),
            Ridge(alpha=0.1, random_state=42)
        )
        _record_cv(('Polynomial', d), cross_val_score(model, X_2d, y, cv=cv, scoring='neg_mean_squared_error'))

        model.fit(X_2d, y)
        model.model_type_ = 'Polynomial'
        model.model_params_ = d
        fitted_models[('Polynomial', d)] = model

    # 2. Kriging. [digiqual] skipped above 1000 samples: every fit is O(N^3).
    n_samples = len(y)
    if n_samples <= 1000:
        candidate_kernels = build_kriging_candidate_kernels(X_2d.shape[1])
        var_y = float(np.var(y)) if np.var(y) > 0 else 1.0

        fitted_candidates = {}
        kernel_loo_scores = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            # 2a. Pick the correlation family by LOO MSE ([M26] Sec. 2.2.2, Eq. 10-14)
            for kname, kernel in candidate_kernels.items():
                try:
                    gpr_cand = build_kriging_gpr(kernel).fit(X_2d, y)
                    loo_means, _, _, _ = compute_kriging_loo_residuals(gpr_cand, X_2d, y)
                    kernel_loo_scores[kname] = float(np.mean((y - loo_means) ** 2) / var_y)  # [M26] Eq. 14
                    fitted_candidates[kname] = gpr_cand
                except Exception as e:  # noqa: BLE001 - candidate kernel fit can fail in many ways
                    logger.warning("Kriging candidate kernel '%s' failed to fit/score: %s", kname, e)

            best_kernel_name = min(kernel_loo_scores, key=kernel_loo_scores.get) if kernel_loo_scores else None

            # 2b. k-fold CV of the chosen kernel, on equal terms with the polynomials ([M25] Fig. 5)
            if best_kernel_name is not None:
                try:
                    _record_cv(('Kriging', None), cross_val_score(
                        build_kriging_gpr(candidate_kernels[best_kernel_name], n_restarts=KRIGING_RESTARTS_CV),
                        X_2d, y, cv=cv, scoring='neg_mean_squared_error'
                    ))
                except Exception as e:  # noqa: BLE001 - leave Kriging out rather than abort the whole fit
                    logger.warning("Kriging k-fold CV failed for kernel '%s': %s", best_kernel_name, e)
                    best_kernel_name = None

        if best_kernel_name is not None:
            best_kriging_gpr = fitted_candidates[best_kernel_name]

            loo_means, _loo_stds, std_residuals, gamma = compute_kriging_loo_residuals(best_kriging_gpr, X_2d, y)
            best_kriging_gpr.loo_means_ = loo_means
            best_kriging_gpr.loo_residuals_ = std_residuals          # [M26] Eq. 19
            best_kriging_gpr.outlier_scale_factor_ = gamma           # [M26] Sec. 2.2.5.2 (diagnostic only)
            best_kriging_gpr.best_kernel_name_ = best_kernel_name
            best_kriging_gpr.kernel_loo_scores_ = kernel_loo_scores
            best_kriging_gpr.kernel_cv_scores_ = kernel_loo_scores   # backwards-compatible alias
            best_kriging_gpr.model_type_ = 'Kriging'
            best_kriging_gpr.model_params_ = {
                'kernel': best_kriging_gpr.kernel_,
                'x_mean': best_kriging_gpr.x_mean_,
                'x_std': best_kriging_gpr.x_std_,
            }
            fitted_models[('Kriging', None)] = best_kriging_gpr
    else:
        print(f"Skipping Kriging evaluation to prevent timeout (Dataset N={n_samples} > 1000).")

    # 3. Overall winner by the one-standard-error rule ([M25] Sec. 4b(i))
    cv_winner_key = select_cv_winner(cv_scores, cv_se)
    for model in fitted_models.values():
        model.cv_se_ = cv_se

    return fitted_models, cv_scores, cv_winner_key


def generate_latex_equation(model: Any, feature_names: list, outcome_name: str = "y") -> str:
    """
    Extracts a LaTeX formatted equation from a fitted Polynomial Pipeline.
    """
    if getattr(model, 'model_type_', None) != 'Polynomial':
        return "Equation not available for Kriging models (Gaussian Process)."

    poly = model.named_steps['polynomialfeatures']
    # Changed from linearregression to ridge to match our new pipeline step
    lr = model.named_steps['ridge']

    terms = poly.get_feature_names_out(feature_names)
    import numpy as np
    coefs = lr.coef_[0] if lr.coef_.ndim > 1 else lr.coef_
    intercept = lr.intercept_[0] if np.ndim(lr.intercept_) > 0 else lr.intercept_

    latex_outcome = outcome_name.replace("_", "\\_")
    equation = f"{latex_outcome} = {intercept:.4g}"

    import re
    for coef, term in zip(coefs, terms):
        if term == "1" or abs(coef) < 1e-7:  # Skip intercept or terms shrunk near zero
            continue

        formatted_term = term.replace(" ", " \\cdot ")
        formatted_term = re.sub(r'\^(\d+)', r'^{\1}', formatted_term)
        formatted_term = formatted_term.replace("_", "\\_")

        sign = "+" if coef > 0 else "-"
        equation += f" {sign} {abs(coef):.4g} \\cdot {formatted_term}"

    return equation

def plot_model_selection(
    cv_scores: dict,
    used_key: tuple | None = None,
    cv_winner_key: tuple | None = None,
    cv_se: dict | None = None
) -> Any:
    """
    Generates a normalized bar chart of the Bias-Variance Tradeoff from CV scores,
    alongside a sorted table of the exact MSE values in best-fit order.

    Bars are colour-coded to distinguish the CV winner, the user-forced model
    (when different from the CV winner), and all other candidates. When the
    standard errors are given, the one-standard-error threshold used by
    `select_cv_winner` is drawn as a dashed line: the winner is the simplest
    model whose bar is below it.

    Args:
        cv_scores (dict): Dictionary mapping ``(model_type, params)`` tuples to
            their Cross-Validation MSE scores.
        used_key (tuple | None): The ``(type, params)`` key of the model that was
            actually used for the PoD calculation. If ``None``, the bar with the
            lowest MSE is treated as the used model.
        cv_winner_key (tuple | None): The ``(type, params)`` key of the CV winner.
            If ``None``, falls back to the bar with the lowest MSE.
        cv_se (dict | None): Standard error of each CV score (the models'
            ``cv_se_`` attribute). Optional.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    # --- 1. Build ordered labels and MSE values ---
    poly_keys = [k for k in cv_scores if k[0] == 'Polynomial']
    poly_degrees = sorted([k[1] for k in poly_keys])

    labels = []
    mses = []
    keys = []

    for d in poly_degrees:
        labels.append(f"Poly {d}")
        mses.append(cv_scores[('Polynomial', d)])
        keys.append(('Polynomial', d))

    if ('Kriging', None) in cv_scores:
        labels.append("Kriging")
        mses.append(cv_scores[('Kriging', None)])
        keys.append(('Kriging', None))

    # Normalise by minimum error
    min_mse = min(mses)
    normalized_mses = [m / min_mse for m in mses]

    # Resolve which bar is the CV winner and which is the used model
    if cv_winner_key is None:
        cv_winner_key = keys[int(np.argmin(mses))]
    if used_key is None:
        used_key = cv_winner_key

    forced = (used_key != cv_winner_key)

    # --- 2. Assign bar colours ---
    colours = []
    for k in keys:
        if k == cv_winner_key:
            colours.append('crimson')
        elif k == used_key:
            colours.append('#ff7f0e')
        else:
            colours.append('#1f77b4')

    # --- 3. Build the sorted MSE table ---
    def _name(key):
        return f"Poly {key[1]}" if key[0] == 'Polynomial' else "Kriging"

    show_se = bool(cv_se) and any(np.isfinite(cv_se.get(k, np.nan)) for k in cv_scores)
    sorted_scores = sorted(cv_scores.items(), key=lambda item: item[1])
    table_data = []
    for key, score in sorted_scores:
        row = [_name(key), f"{score:.3g}"]
        if show_se:
            se = cv_se.get(key, np.nan)
            row.append(f"{se:.2g}" if np.isfinite(se) else "-")
        table_data.append(row)

    # --- 4. Create figure ---
    fig, (ax_plot, ax_table) = plt.subplots(
        1, 2, figsize=(10, 5.5), gridspec_kw={'width_ratios': [2.2, 1]}
    )

    # --- Bar Chart ---
    ax_plot.bar(labels, normalized_mses, color=colours, edgecolor='black', alpha=0.85)

    y_limit = 6
    ax_plot.set_ylim(0, y_limit)

    # Add arrows for cut-off bars
    for i, val in enumerate(normalized_mses):
        if val > y_limit:
            ax_plot.annotate(
                '',
                xy=(i, y_limit),
                xytext=(i, y_limit - 0.4),
                arrowprops={
                    'facecolor': 'black',
                    'shrink': 0.05,
                    'width': 2,
                    'headwidth': 8
                }
            )

    ax_plot.axhline(1.0, color='red', linestyle='-.', linewidth=1.5)

    # One-standard-error threshold ([M25] Sec. 4b(i), see select_cv_winner)
    best_key = keys[int(np.argmin(mses))]
    se_threshold = None
    if cv_se and np.isfinite(cv_se.get(best_key, np.nan)):
        se_threshold = (min_mse + cv_se[best_key]) / min_mse
        ax_plot.axhline(se_threshold, color='grey', linestyle='--', linewidth=1.2)
    ax_plot.set_title('Model Selection: Bias-Variance Tradeoff', fontweight='bold')
    ax_plot.set_ylabel('Error / Min Error [-]')
    ax_plot.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax_plot.tick_params(axis='x', rotation=45)

    # Legend placed cleanly above the plot
    from matplotlib.lines import Line2D
    winner_label = 'Selected (1-SE rule)' if se_threshold is not None else 'Selected (min CV error)'
    legend_handles = [mpatches.Patch(color='crimson', label=winner_label)]
    if forced:
        legend_handles.append(mpatches.Patch(color='#ff7f0e', label='Used (Override)'))
    legend_handles.append(mpatches.Patch(color='#1f77b4', label='Other Candidates'))
    legend_handles.append(Line2D([0], [0], color='red', linestyle='-.', label='Min CV error'))
    if se_threshold is not None:
        legend_handles.append(Line2D([0], [0], color='grey', linestyle='--', label='Min + 1 SE'))

    # Inside the axes, in the empty space above the bars (y-limit is 6x the minimum)
    ax_plot.legend(handles=legend_handles, fontsize=9, loc='upper right', ncol=2, framealpha=0.9)

    # --- Table ---
    ax_table.axis('off')
    ax_table.set_title('CV MSE\n(lowest first)', fontweight='bold')

    # Bounding box [x0, y0, width, height] prevents table from expanding into the title
    col_labels = ["Model", "CV MSE", "± SE"] if show_se else ["Model", "CV MSE"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        bbox=[0, 0.08, 1, 0.77],
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style table rows: bold headers and lowest-MSE row, highlight selected and used models
    cv_winner_name = _name(cv_winner_key)
    used_name = _name(used_key)
    lowest_name = _name(best_key)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            continue
        cell_label = table_data[row - 1][0]
        if cell_label == lowest_name:
            cell.set_text_props(weight='bold')
        if cell_label == cv_winner_name:
            cell.set_facecolor('#ffcccc')
        if forced and cell_label == used_name:
            cell.set_facecolor('#ffe0b2')

    note = "Bold: lowest CV MSE.  Red: selected."
    if forced:
        note += "\nOrange: used (override)."
    ax_table.text(0.5, 0.0, note, ha='center', va='bottom', fontsize=8.5,
                  style='italic', transform=ax_table.transAxes)

    fig.tight_layout()
    return fig


#### Variance Model - Kernel Smoothing ####

def optimise_bandwidth(
    X: np.ndarray,
    residuals: np.ndarray,
    min_ratio: float = 0.01,
    max_ratio: float = 0.5
) -> float:
    """
    Finds the optimal kernel smoothing bandwidth using Leave-One-Out Cross-Validation (LOO-CV).

    This function automatically determines the best smoothing window (sigma) for the
    variance model. It evaluates different bandwidths by predicting the squared residual
    of each point using a Gaussian weighted average of all *other* points, selecting
    the bandwidth that minimizes the Mean Squared Error (MSE) of these predictions.

    Args:
        X (np.ndarray): A 1D array of the original input locations (e.g., flaw sizes).
        residuals (np.ndarray): The raw residuals calculated from the mean model
            (differences between observed y and predicted mean y).
        min_ratio (float, optional): The lower bound for the optimizer's search space,
            defined as a fraction of the total range of X (X.max() - X.min()). Defaults to 0.01.
        max_ratio (float, optional): The upper bound for the optimizer's search space,
            defined as a fraction of the total range of X. Defaults to 0.5.

    Returns:
        float: The optimal smoothing bandwidth in the absolute units of X.

    Examples:
        ```python
        import numpy as np

        # 1. Generate dummy input data and simulated residuals
        X = np.linspace(0, 10, 50)
        # Simulate heteroscedastic noise (variance increases with X)
        residuals = np.random.normal(0, X * 0.5, size=50)

        # 2. Find the optimal bandwidth
        optimal_bw = optimize_bandwidth(X, residuals)
        print(f"Optimal Bandwidth: {optimal_bw:.4f}")
        ```
    """
    from scipy.spatial.distance import cdist
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)
    sq_residuals = residuals.flatten() ** 2
    data_range = np.max(X_2d.max(axis=0) - X_2d.min(axis=0))

    def loo_cv_objective(bw: float) -> float:
        # Calculate euclidean distance matrix
        dists = cdist(X_2d, X_2d, metric='euclidean')

        # Calculate Gaussian weights
        weights = stats.norm.pdf(dists, loc=0, scale=bw)

        # Leave-One-Out: Set diagonal to zero so a point doesn't predict itself
        np.fill_diagonal(weights, 0)

        # Normalize weights so each row sums to 1
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0] = 1e-10  # Prevent division by zero
        weights = weights / row_sums.reshape(-1, 1)

        # Predict the squared residuals
        preds = weights @ sq_residuals

        # Return the Mean Squared Error of the variance prediction
        return float(np.mean((sq_residuals - preds) ** 2))

    # Define the search bounds for the optimizer
    bounds = (data_range * min_ratio, data_range * max_ratio)

    # Run a bounded scalar optimization to find the minimum LOO-CV error
    res = minimize_scalar(loo_cv_objective, bounds=bounds, method='bounded')

    return float(res.x)


def fit_variance_model(
    X: np.ndarray,
    y: np.ndarray,
    mean_model: Any,
    auto_bandwidth: bool = True,
    bandwidth_ratio: float = 0.1
) -> tuple[np.ndarray, float]:
    """
    Calculates residuals and defines the smoothing bandwidth for variance estimation.

    This function acts as the setup phase for modeling heteroscedasticity ([M25] Sec. 2e).
    It computes the residuals from the provided mean model and establishes the smoothing
    bandwidth either via automated Cross-Validation or a fixed user-defined ratio.

    For polynomial mean models the residuals are the in-sample residuals, as in [M25]
    Eq. 2.9. For Kriging models they are the leave-one-out residuals ``y - loo_means_``
    [digiqual]. A Kriging model with a learned nugget partly fits the scatter at the
    training points, so its in-sample residuals understate the scatter, which would make
    the PoD curve too steep. The LOO residuals measure how far a new observation falls
    from a prediction made without it.

    The Kriging outlier factor gamma ([M26] Sec. 2.2.5.2) is *not* applied here. In [M26]
    it widens the Kriging interpolation uncertainty used for the PoD bound only, and the
    PoD curve itself is unchanged; digiqual's bound comes from the bootstrap, so gamma is
    kept as a diagnostic (see `compute_kriging_loo_residuals`).

    Args:
        X (np.ndarray): The 1D array of original input data (e.g., parameter of interest).
        y (np.ndarray): The 1D array of original outcome data (e.g., signal response).
        mean_model (Any): A fitted scikit-learn estimator (e.g., Pipeline or
            GaussianProcessRegressor) that implements a `.predict()` method.
        auto_bandwidth (bool, optional): If True, dynamically calculates the optimal
            bandwidth using Leave-One-Out Cross-Validation. If False, falls back to
            the fixed `bandwidth_ratio`. Defaults to True.
        bandwidth_ratio (float, optional): The kernel smoothing window size as a
            fraction of the data range (X.max() - X.min()). Only used if
            `auto_bandwidth` is False. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, float]:
            - residuals: Differences between `y` and the mean model predictions
              (leave-one-out predictions for Kriging).
            - bandwidth: The selected smoothing window size (in absolute units of X).

    Examples:
        ```python
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # 1. Setup dummy data and a basic mean model
        X = np.linspace(0, 10, 50)
        y = 2.5 * X + np.random.normal(0, 1, 50)

        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)

        # 2. Extract residuals and optimized bandwidth
        residuals, bandwidth = fit_variance_model(
            X, y,
            mean_model=model,
            auto_bandwidth=True
        )

        print(f"Calculated Bandwidth: {bandwidth:.4f}")
        ```
    """
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)
    y = np.asarray(y, dtype=np.float64).flatten()

    loo_means = getattr(mean_model, 'loo_means_', None)
    if getattr(mean_model, 'model_type_', None) == 'Kriging':
        if loo_means is None or len(loo_means) != len(y):
            loo_means, _, _, _ = compute_kriging_loo_residuals(mean_model, X_2d, y)
        residuals = y - loo_means
    else:
        residuals = y - mean_model.predict(X_2d)

    if auto_bandwidth:
        print("   -> Optimizing bandwidth via LOO-CV...")
        bandwidth = optimise_bandwidth(X_2d, residuals)
    else:
        data_range = np.max(X_2d.max(axis=0) - X_2d.min(axis=0))
        bandwidth = data_range * bandwidth_ratio

    return residuals, bandwidth


def predict_local_std(
    X: np.ndarray,
    residuals: np.ndarray,
    X_eval: np.ndarray,
    bandwidth: float
) -> np.ndarray:
    """
    Estimates the local standard deviation using Gaussian Kernel Smoothing.

    This implements a Nadaraya-Watson estimator specifically for the squared
    residuals to model how noise varies across the input domain (heteroscedasticity).
    """
    from .cpp_fallback import predict_local_std_fast
    return predict_local_std_fast(X, residuals, X_eval, bandwidth)


#### Residual Distribution Fitting ####

def infer_best_distribution(
    residuals: np.ndarray,
    X: np.ndarray,
    bandwidth: float
) -> tuple[str, tuple]:
    """
    Selects the best statistical distribution for the standardized residuals using AIC.

    This function normalizes residuals by their local standard deviation (Z-scores)
    and tests them against a suite of candidate distributions (Normal, Gumbel,
    Logistic, Laplace, t-Student).

    Args:
        residuals (np.ndarray): Raw residuals from the mean model.
        X (np.ndarray): Input locations for the residuals.
        bandwidth (float): Bandwidth used for local standardization.

    Returns:
        tuple[str, tuple]:
            - best_name: The SciPy name of the best-fitting distribution (e.g., 'norm').
            - best_params: The fitted parameters for that distribution (e.g., loc, scale).

    Examples:
        ```python
        dist_name, dist_params = infer_best_distribution(residuals, X, bandwidth)
        print(f"Best distribution: {dist_name}")
        ```
    """
    local_std = predict_local_std(X, residuals, X, bandwidth)
    z_scores = residuals.flatten() / local_std.flatten()

    candidates = [
        "norm",         # Gaussian (Classical standard)
        "gumbel_r",     # Right-skewed Extreme Value
        "gumbel_l",     # Left-skewed Extreme Value (Common for cracks)
        "weibull_min",  # Weibull Minimum (Malkiel 2025)
        "weibull_max",  # Weibull Maximum
        "gamma",        # Gamma distribution
        "expon",        # Exponential distribution
        "logistic",     # Heavier tails than Normal
        "laplace",      # Sharper peak, heavy tails
        "t",            # Student's t (Very robust to outliers)
        "beta",         # Beta distribution
        "uniform",      # Uniform distribution
    ]

    best_aic = np.inf
    best_result = ("norm", (0, 1))

    for dist_name in candidates:
        try:
            dist_obj = getattr(stats, dist_name)
            params = dist_obj.fit(z_scores)
            log_likelihood = np.sum(np.log(dist_obj.pdf(z_scores, *params)))

            k = len(params)
            aic = 2*k - 2*log_likelihood

            if aic < best_aic:
                best_aic = aic
                best_result = (dist_name, params)
        except Exception as e:  # noqa: BLE001 - distribution fit can fail in many ways
            logger.debug("Distribution '%s' failed to fit: %s", dist_name, e)
            continue

    return best_result


#### PoD Generation and Bootstrap Intervals ####

def compute_pod_curve(
    X_eval: np.ndarray,
    mean_model: Any,
    X: np.ndarray,
    residuals: np.ndarray,
    bandwidth: float,
    dist_info: tuple[str, tuple],
    threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Probability of Detection (PoD) curve.

    Combines the Mean Model, Variance Model, and Error Distribution to compute
    the probability that the signal exceeds the threshold at every point in `X_eval`.

    Args:
        X_eval (np.ndarray): The grid points to calculate PoD for.
        mean_model (Any): The fitted sklearn mean response model.
        X (np.ndarray): Original input data (needed for variance prediction).
        residuals (np.ndarray): Original residuals (needed for variance prediction).
        bandwidth (float): Smoothing bandwidth.
        dist_info (tuple[str, tuple]): The (name, params) of the error distribution.
        threshold (float): The detection threshold value.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - pod_curve: Array of probabilities [0, 1] for each point in X_eval.
            - mean_curve: Array of mean signal response values for X_eval.

    Examples:
        ```python
        # Assuming we have fitted models (mean_model) and data (X, residuals)
        # Calculate the PoD curve for a threshold of 0.5
        pod, mean_resp = compute_pod_curve(
            X_eval=np.linspace(0, 10, 100),
            mean_model=mean_model,
            X=X,
            residuals=residuals,
            bandwidth=1.5,
            dist_info=('norm', (0, 1)),
            threshold=0.5
        )
        ```
    """
    dist_name, dist_params = dist_info

    X_eval_2d = np.atleast_2d(X_eval).T if np.asarray(X_eval).ndim == 1 else np.asarray(X_eval)
    mean_curve = mean_model.predict(X_eval_2d)
    sigma_curve = predict_local_std(X, residuals, X_eval_2d, bandwidth)

    z_threshold = (threshold - mean_curve) / sigma_curve

    dist_obj = getattr(stats, dist_name)
    pod_curve = 1 - dist_obj.cdf(z_threshold, *dist_params)

    return pod_curve, mean_curve



def _single_bootstrap_step(
    X_2d, y, X_eval, threshold, model_type, model_params,
    bandwidth, dist_info, nuisance_ranges, n_samples,
    feature_names=None, poi_names=None, nuisance_dists=None,
    seed=None, n_mc_samples=500
):
    """Internal helper to process a single bootstrap iteration."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Resample indices
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_samples, n_samples, replace=True)
    else:
        idx = np.random.choice(n_samples, n_samples, replace=True)
    X_res_2d = X_2d[idx]
    y_res = y[idx]

    # Fit Mean Model with regularized Ridge regression to prevent bumpy intervals
    if model_type == 'Polynomial':
        from sklearn.linear_model import Ridge  # <-- regularized bootstrap model
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        mean_model = make_pipeline(
            PolynomialFeatures(model_params),
            StandardScaler(),
            Ridge(alpha=0.1, random_state=42)
        )
    elif model_type == 'Kriging':
        # Frozen hyperparameters and input scaling ([digiqual], see build_fixed_kernel_gpr)
        mean_model = build_fixed_kernel_gpr(model_params)

    mean_model.fit(X_res_2d, y_res)

    if model_type == 'Kriging':
        # LOO residuals, as in fit_variance_model. All copies of a resampled point are
        # left out together so a point is never predicted from its own duplicate.
        res_res = y_res - compute_kriging_group_loo_means(mean_model, X_res_2d, y_res, groups=idx)
    else:
        res_res = y_res - mean_model.predict(X_res_2d)

    dist_name, _dist_params = dist_info
    try:
        local_std_res = predict_local_std(X_res_2d, res_res, X_res_2d, bandwidth)
        local_std_res = np.maximum(local_std_res, 1e-10)
        z_res = res_res.flatten() / local_std_res.flatten()
        dist_obj = getattr(stats, dist_name)
        new_params = dist_obj.fit(z_res)
        new_dist_info = (dist_name, new_params)
    except Exception as e:  # noqa: BLE001 - fall back to original distribution on any fit failure
        logger.debug("Bootstrap re-fit of distribution '%s' failed, reusing original: %s", dist_name, e)
        new_dist_info = dist_info

    from .integration import compute_multi_dim_pod
    pod_curve, _ = compute_multi_dim_pod(
        X_eval, nuisance_ranges or {}, mean_model, X_res_2d, res_res,
        bandwidth, new_dist_info, threshold, n_mc_samples=n_mc_samples,
        feature_names=feature_names, poi_names=poi_names, nuisance_dists=nuisance_dists
    )
    return pod_curve




def bootstrap_pod_ci(
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    threshold: float,
    model_type: str,
    model_params: Any,
    bandwidth: float,
    dist_info: tuple[str, tuple],
    n_boot: int = 1000,
    nuisance_ranges: dict | None = None,
    n_jobs: int | None = None,
    feature_names: list | None = None,
    poi_names: list | None = None,
    confidence_levels: list | None = None,
    nuisance_dists: dict | None = None,
    progress_callback: Any = None,
    n_mc_samples: int = 500
) -> tuple[np.ndarray, np.ndarray] | dict:
    """
    Estimates Confidence Bounds for the PoD curve via Bootstrapping.

    This function resamples the original data with replacement `n_boot` times.
    For each resample, it refits the Mean Model (dynamically rebuilding either
    a Polynomial or Kriging model), recalculates residuals, and generates a new PoD curve.
    If Kriging is selected, the optimizer is disabled during bootstrapping to remain
    computationally tractable [digiqual]: each resample reuses the fitted kernel (including
    its learned WhiteKernel noise level) and the full-data input scaling via
    `build_fixed_kernel_gpr`, so only the posterior mean is recomputed. [M25] Sec. 2i
    re-estimates the model on every resample, so this interval does not include the
    uncertainty in the Kriging hyperparameters. Kriging residuals for the variance model
    are leave-one-out residuals, matching `fit_variance_model`.

    Args:
        model_params (Any): The polynomial degree, or for Kriging the model's
            ``model_params_`` dict (``kernel``, ``x_mean``, ``x_std``).
    """
    import gc

    n_samples = len(y)
    X_2d = np.atleast_2d(X).T if np.asarray(X).ndim == 1 else np.asarray(X)

    total_cores = os.cpu_count() or 4
    if n_jobs is None or n_jobs == -1:
        n_jobs_actual = max(1, total_cores - 2)
    elif n_jobs == 1:
        n_jobs_actual = 1
    else:
        n_jobs_actual = min(max(1, n_jobs), total_cores)

    print(f"   -> [Bootstrap] Running {n_boot} iterations on {n_jobs_actual} worker core(s)...", flush=True)

    N_eval_len = len(X_eval)
    pod_matrix = np.empty((n_boot, N_eval_len))

    chunk_size = 50
    for b_start in range(0, n_boot, chunk_size):
        b_end = min(b_start + chunk_size, n_boot)
        n_chunk = b_end - b_start

        if n_jobs_actual > 1:
            chunk_results = Parallel(n_jobs=n_jobs_actual, backend="multiprocessing", verbose=0)(
                delayed(_single_bootstrap_step)(
                    X_2d, y, X_eval, threshold, model_type, model_params,
                    bandwidth, dist_info, nuisance_ranges, n_samples,
                    feature_names, poi_names, nuisance_dists,
                    seed=b_start + i, n_mc_samples=n_mc_samples
                ) for i in range(n_chunk)
            )
        else:
            chunk_results = [
                _single_bootstrap_step(
                    X_2d, y, X_eval, threshold, model_type, model_params,
                    bandwidth, dist_info, nuisance_ranges, n_samples,
                    feature_names, poi_names, nuisance_dists,
                    seed=b_start + i, n_mc_samples=n_mc_samples
                ) for i in range(n_chunk)
            ]

        for i, res in enumerate(chunk_results):
            pod_matrix[b_start + i] = res

        completed = b_end
        pct = int((completed / n_boot) * 100)
        print(f"   -> [Bootstrap Progress] Completed {completed}/{n_boot} iterations ({pct}%)...", flush=True)

        if progress_callback is not None:
            try:
                progress_callback(completed, n_boot)
            except Exception as e:  # noqa: BLE001 - user-supplied callback, must not abort bootstrap
                logger.warning("Progress callback raised an exception: %s", e)

        del chunk_results
        gc.collect()

    if confidence_levels is None:
        return np.percentile(pod_matrix, 2.5, axis=0), np.percentile(pod_matrix, 97.5, axis=0)

    bounds = {}
    for cl in confidence_levels:
        low_p = (100.0 - cl) / 2.0
        high_p = 100.0 - low_p
        bounds[cl] = (np.percentile(pod_matrix, low_p, axis=0), np.percentile(pod_matrix, high_p, axis=0))
    return bounds


def calculate_reliability_point(
    X_eval: np.ndarray,
    ci_lower: np.ndarray,
    target_pod: float = 0.90
) -> float:
    """
    Calculates the defect size (a90/95) where the Lower Confidence Bound
    crosses the target reliability threshold (usually 0.90).

    Args:
        X_eval (np.ndarray): The evaluation grid points.
        ci_lower (np.ndarray): The lower confidence bound curve (y values).
        target_pod (float, optional): Target reliability level. Defaults to 0.90.

    Returns:
        float: The interpolated x-value, or np.nan if not reached.

    Examples:
        ```python
        a90_95 = calculate_reliability_point(X_eval, lower_ci, target_pod=0.90)
        print(f"a90/95 point: {a90_95:.2f}")
        ```
    """
    # Check if the curve actually reaches the target
    if np.max(ci_lower) < target_pod:
        return np.nan

    # Interpolate to find exact crossing point
    # We swap args because we are solving for X given Y=0.90
    monotonic_ci = np.maximum.accumulate(ci_lower)
    return float(np.interp(target_pod, monotonic_ci, X_eval))


def calculate_sobol_indices(mean_model: Any, feature_names: list, data_df, n_samples: int = 1024) -> dict | None:
    """
    Calculates the Total-Order Sobol sensitivity index for the fitted mean model.
    Optimized for speed by disabling second-order interaction matrices.
    """
    try:
        from SALib.analyze import sobol as salib_analyze
        from SALib.sample import sobol as salib_sample
    except ImportError:
        print("Warning: SALib not found. Skipping Sobol index calculation.")
        return None

    # 1. Define the bounds for each feature
    bounds = []
    for col in feature_names:
        bounds.append([float(data_df[col].min()), float(data_df[col].max())])

    problem = {
        'num_vars': len(feature_names),
        'names': feature_names,
        'bounds': bounds
    }

    # 2. FAST SAMPLING: explicitly disable second-order calculations
    X_sample = salib_sample.sample(problem, n_samples, calc_second_order=False)

    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(-1, 1)

    y_sample = mean_model.predict(X_sample)

    # 3. Analyze the results (also disabling second-order here)
    import warnings
    with warnings.catch_warnings():
        # Ignore divide-by-zero warnings if the predicted surface is perfectly flat
        warnings.simplefilter("ignore", RuntimeWarning)
        Si = salib_analyze.analyze(problem, y_sample.flatten(), print_to_console=False, calc_second_order=False)

    # 4. Extract ONLY the Total-Order effect (ST) and clamp numerical noise
    results = {}
    for i, name in enumerate(feature_names):
        raw_val = float(Si['ST'][i])

        # --- Clamp the value strictly between 0.0 and 1.0 ---
        # This prevents Monte Carlo approximation noise from showing > 100% or < 0%
        clamped_val = max(0.0, min(1.0, raw_val))

        results[name] = clamped_val

    return results
