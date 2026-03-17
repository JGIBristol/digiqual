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
