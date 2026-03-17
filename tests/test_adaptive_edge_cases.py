import pandas as pd
import numpy as np
from unittest.mock import patch
from digiqual.adaptive import (
    _filter_by_graveyard,
    _fill_gaps,
    _sample_uncertainty,
    generate_targeted_samples,
    _execute_simulation,
    run_adaptive_search
)

def test_filter_by_graveyard():
    cands = pd.DataFrame({'A': [1.0, 5.0, 9.0]})
    grave = pd.DataFrame({'A': [4.9]})
    # 5.0 is very close to 4.9, should be filtered if threshold is say 0.05
    # max - min = 9.0 - 1.0 = 8.0. 
    # Normalised 4.9 is 3.9/8. 5.0 is 4.0/8. Diff is 0.1/8 = 0.0125.
    filtered = _filter_by_graveyard(cands, grave, ['A'], threshold=0.05)
    assert len(filtered) == 2
    assert 5.0 not in filtered['A'].values

    # Test empty inputs
    assert _filter_by_graveyard(pd.DataFrame(), grave, ['A']).empty
    assert len(_filter_by_graveyard(cands, None, ['A'])) == 3

def test_fill_gaps_graveyard():
    df = pd.DataFrame({'A': [0.0, 10.0]})
    grave = pd.DataFrame({'A': [5.0]})
    res = _fill_gaps(df, 'A', ['A'], n=5, graveyard=grave, threshold=0.05)
    assert not res.empty

def test_sample_uncertainty_graveyard():
    df = pd.DataFrame({'A': [0, 5, 10], 'Signal': [1, 2, 3]})
    grave = pd.DataFrame({'A': [5.0]})
    res = _sample_uncertainty(df, ['A'], 'Signal', n=2, graveyard=grave)
    assert not res.empty

    # Test n <= 0
    res0 = _sample_uncertainty(df, ['A'], 'Signal', n=0)
    assert res0.empty

    # Test aggressive grave
    grave_all = pd.DataFrame({'A': np.linspace(0, 10, 100)})
    res_empty = _sample_uncertainty(
        df, ['A'], 'Signal', n=2, graveyard=grave_all, threshold=1.0
    )
    assert res_empty.empty

def test_generate_targeted_samples_graveyard_and_empty():
    group1 = np.linspace(0, 2, 8)
    group2 = np.linspace(8, 10, 8)
    df = pd.DataFrame({
        'A': np.concatenate([group1, group2]),
        'Signal': np.concatenate([group1, group2])
    })
    grave = pd.DataFrame({'A': [5.0]})
    
    # Hit graveyard log message
    res = generate_targeted_samples(df, ['A'], 'Signal', failed_data=grave)
    assert not res.empty
    
    # Hit empty new_samples_list (no failures)
    df_perfect = pd.DataFrame({
        'A': np.linspace(0, 10, 20),
        'Signal': np.linspace(0, 10, 20)
    })
    res2 = generate_targeted_samples(df_perfect, ['A'], 'Signal')
    assert res2.empty
    
    with patch("digiqual.adaptive._fill_gaps", return_value=pd.DataFrame()), \
         patch("digiqual.adaptive._sample_uncertainty", return_value=pd.DataFrame()):
         assert generate_targeted_samples(df, ['A'], 'Signal').empty

@patch("digiqual.adaptive.subprocess.run")
def test_execute_simulation_read_csv_exception(mock_run):
    samples = pd.DataFrame({'A': [1]})
    
    # os.path.exists is true, but read_csv fails
    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", side_effect=Exception):
         res = _execute_simulation(samples, "cmd", ['A'], "in.csv", "out.csv")
         assert res.empty

@patch("digiqual.adaptive.generate_lhs")
@patch("digiqual.adaptive._execute_simulation")
def test_run_adaptive_search_edge_cases(mock_exec, mock_lhs):
    mock_lhs.return_value = pd.DataFrame({'A': [1]})
    
    # mock_exec returns empty
    mock_exec.return_value = pd.DataFrame()
    cmd = "cmd"
    ranges = {'A': (0,10)}
    with patch(
        "digiqual.adaptive.validate_simulation", 
        return_value=(pd.DataFrame(), pd.DataFrame())
    ), patch(
        "digiqual.adaptive.generate_targeted_samples", 
        return_value=pd.DataFrame()
    ):
        _ = run_adaptive_search(cmd, ['A'], 'Signal', ranges, max_iter=1)
    
    # max_hours hit
    with patch("time.time", side_effect=[0, 10000, 10000]):
         existing_df = pd.DataFrame({
             'A': np.linspace(0, 10, 15),
             'Signal': np.linspace(0, 10, 15)
         })
         _ = run_adaptive_search(
             cmd, ['A'], 'Signal', ranges,
             existing_data=existing_df, max_iter=2, max_hours=1.0
         )

    # new_samples empty break 
    with patch(
        "digiqual.adaptive.validate_simulation", 
        return_value=(pd.DataFrame(), pd.DataFrame())
    ), patch(
        "digiqual.adaptive.generate_targeted_samples", 
        return_value=pd.DataFrame()
    ):
         run_adaptive_search(
             cmd, ['A'], 'Signal', ranges,
             existing_data=pd.DataFrame({'A': [1]}), max_iter=1
         )
         
    # new_results empty in loop
    valid_data_df = pd.DataFrame({
        'A': np.linspace(0, 10, 15),
        'Signal': np.linspace(0, 10, 15)
    })
    with patch(
        "digiqual.adaptive.validate_simulation", 
        return_value=(valid_data_df, pd.DataFrame())
    ), patch(
        "digiqual.adaptive.generate_targeted_samples", 
        return_value=pd.DataFrame({'A': [2]})
    ), patch(
        "digiqual.adaptive._execute_simulation", 
        return_value=pd.DataFrame()
    ):
         run_adaptive_search(
             cmd, ['A'], 'Signal', ranges,
             existing_data=valid_data_df, max_iter=1
         )
