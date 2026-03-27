import pytest
import numpy as np
import pandas as pd
import scipy.stats as stats
from digiqual.integration import build_integration_space, compute_marginal_pod
from unittest.mock import MagicMock, patch

def test_build_integration_space_custom_dist():
    ref_data = pd.DataFrame({'Angle': [0, 10], 'Roughness': [0, 1]})
    dists = {'Angle': stats.norm(loc=5, scale=2)}
    
    mc_matrix = build_integration_space(
        nuisance_cols=['Angle'],
        reference_data=ref_data,
        nuisance_dists=dists,
        n_mc_samples=100
    )
    
    assert mc_matrix.shape == (100, 1)
    # The mean should be relatively close to 5
    assert 3 < np.mean(mc_matrix[:, 0]) < 7

def test_build_integration_space_safetynet(capsys):
    ref_data = pd.DataFrame({'Angle': [-10.0, 10.0]})
    # Intentionally missing the distribution for 'Angle'
    
    mc_matrix = build_integration_space(
        nuisance_cols=['Angle'],
        reference_data=ref_data,
        nuisance_dists=None,
        n_mc_samples=100
    )
    
    # Should catch the warning output
    captured = capsys.readouterr()
    assert "Warning: No distribution provided" in captured.out
    
    assert mc_matrix.shape == (100, 1)
    # Uniform between -10 and 10 => mean ~ 0, min >= -10, max <= 10
    assert np.min(mc_matrix[:, 0]) >= -10.0
    assert np.max(mc_matrix[:, 0]) <= 10.0

@patch("digiqual.pod.predict_local_std")
def test_compute_marginal_pod(mock_std):
    # Dummy mock setup
    mock_std.return_value = np.array([1.0, 1.0])
    
    mock_mean_model = MagicMock()
    mock_mean_model.predict.return_value = np.array([10.0, 10.0])
    
    X_eval_grid = np.array([1.0, 2.0]) # 2 PoI evaluation points
    mc_samples = np.array([[5.0], [6.0]]) # 2 MC samples per point
    X_orig = np.array([[1.0, 5.0]])
    residuals = np.array([0.0])
    
    result = compute_marginal_pod(
        X_eval_grid=X_eval_grid,
        mean_model=mock_mean_model,
        bandwidth=0.5,
        dist_info=('norm', (0, 1)),
        threshold=10.0,
        mc_samples=mc_samples,
        X_orig=X_orig,
        residuals=residuals
    )
    
    assert result.shape == (len(X_eval_grid),)
    # Mean is exactly threshold = 10, std = 1.0 -> z = 0
    # True norm cdf at 0 is 0.5. So conditional PoDs = 1 - 0.5 = 0.5
    # Marginal is an average of the conditionals, which are all 0.5
    assert np.all(result == 0.5)
