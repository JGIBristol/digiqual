import pytest
import pandas as pd
from digiqual.sampling import generate_lhs

def test_generate_lhs_missing_cols():
    ranges_df = pd.DataFrame([{'Name': 'A', 'Min': 0}]) # Missing 'Max'
    with pytest.raises(ValueError, match="missing columns"):
        generate_lhs(10, ranges_df)

def test_generate_lhs_bad_name_type():
    ranges_df = pd.DataFrame([{'Name': 123, 'Min': 0, 'Max': 10}])
    with pytest.raises(ValueError, match="must contain only character strings"):
        generate_lhs(10, ranges_df)

def test_generate_lhs_bad_min_max_type():
    ranges_df = pd.DataFrame([{'Name': 'A', 'Min': 'zero', 'Max': 10}])
    with pytest.raises(ValueError, match="must be strictly numeric"):
        generate_lhs(10, ranges_df)

def test_generate_lhs_empty_dict():
    df = generate_lhs(10, {})
    assert df.empty

def test_generate_lhs_empty_df():
    df = generate_lhs(10, pd.DataFrame())
    assert df.empty

def test_generate_lhs_bad_type():
    with pytest.raises(TypeError, match="must be a Dictionary or a DataFrame"):
        generate_lhs(10, "Not a dict or df")

def test_generate_lhs_min_greater_than_max():
    ranges_df = pd.DataFrame([{'Name': 'A', 'Min': 10, 'Max': 0}])
    with pytest.raises(ValueError, match="Bounds error: 'Min' must be strictly lower than 'Max'"):
        generate_lhs(10, ranges_df)


def test_reorder_max_min():
    from digiqual.sampling import reorder_max_min
    import numpy as np

    # Create a simple DataFrame of coordinates with clusters at (0,0) and (1,1)
    df = pd.DataFrame({
        'x': [0.0, 0.1, 0.9, 1.0],
        'y': [0.0, 0.05, 0.85, 1.0]
    })
    
    reordered = reorder_max_min(df)
    assert len(reordered) == len(df)
    
    # The first point chosen is index 0. The second point should be from the (1,1) cluster (either idx 2 or 3)
    # since it is far away, rather than index 1 (which is close to index 0).
    pt0 = reordered.iloc[0].values
    pt1 = reordered.iloc[1].values
    dist = np.linalg.norm(pt1 - pt0)
    assert dist > 1.0

    # Test edge cases (small DataFrames)
    df_small = pd.DataFrame({'x': [1.0]})
    assert len(reorder_max_min(df_small)) == 1
    assert reorder_max_min(pd.DataFrame()).empty
