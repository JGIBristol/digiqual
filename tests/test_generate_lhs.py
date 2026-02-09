import pytest
import pandas as pd
from digiqual.sampling import generate_lhs

# 1. Test that the function works correctly under normal conditions
def test_generate_lhs_success():
    # Setup inputs
    vars_df = pd.DataFrame(
        [
            {"Name": "Var1", "Min": 0, "Max": 10},
            {"Name": "Var2", "Min": 100, "Max": 200},
        ]
    )

    # Run function (Updated argument name to 'ranges')
    result = generate_lhs(n=5, ranges=vars_df, seed=42)

    # Assertions (Checks)
    assert len(result) == 5
    assert list(result.columns) == ["Var1", "Var2"]
    assert result["Var1"].min() >= 0
    assert result["Var1"].max() <= 10
    assert not result.isnull().values.any()


# 2. Test that the function correctly catches the "Min > Max" error
def test_generate_lhs_bad_bounds():
    # Setup bad inputs (Min is 50, Max is 10)
    bad_df = pd.DataFrame([{"Name": "BadVar", "Min": 50, "Max": 10}])

    # We expect this to raise a ValueError
    with pytest.raises(ValueError, match="strictly lower"):
        generate_lhs(n=5, ranges=bad_df)

# 3. Test that non-numeric Min/Max triggers an error
def test_generate_lhs_non_numeric():
    # Setup bad inputs: 'Min' is a string "Ten" instead of number 10
    bad_df = pd.DataFrame([
        {'Name': 'Var1', 'Min': "Ten", 'Max': 20}
    ])

    # We expect a ValueError with the specific message we wrote
    with pytest.raises(ValueError, match="strictly numeric"):
        generate_lhs(n=5, ranges=bad_df)

# 4. Test that non-string Names trigger an error
def test_generate_lhs_bad_names():
    # Setup bad inputs: 'Name' is a number (123) instead of a string
    bad_df = pd.DataFrame([
        {'Name': 123, 'Min': 10, 'Max': 20}
    ])

    # We expect a ValueError about character strings
    with pytest.raises(ValueError, match="character strings"):
        generate_lhs(n=5, ranges=bad_df)

# --- NEW TESTS (Already correct) ---

def test_generate_lhs_dict_input():
    """Test generate_lhs with the new Dictionary input format."""
    ranges = {
        "Length": (0.0, 10.0),
        "Angle": (-45.0, 45.0)
    }

    df = generate_lhs(n=10, ranges=ranges, seed=42)

    assert len(df) == 10
    assert list(df.columns) == ["Length", "Angle"]
    assert df["Length"].min() >= 0.0
    assert df["Length"].max() <= 10.0
    assert df["Angle"].min() >= -45.0
    assert df["Angle"].max() <= 45.0

def test_generate_lhs_dataframe_input():
    """Test generate_lhs with the legacy DataFrame input format."""
    ranges_df = pd.DataFrame([
        {"Name": "Length", "Min": 0.0, "Max": 10.0},
        {"Name": "Angle", "Min": -45.0, "Max": 45.0}
    ])

    df = generate_lhs(n=10, ranges=ranges_df, seed=42)

    assert len(df) == 10
    assert list(df.columns) == ["Length", "Angle"]

def test_generate_lhs_invalid_input_type():
    """Test that it raises TypeError for invalid input types (e.g. list)."""
    with pytest.raises(TypeError):
        generate_lhs(n=10, ranges=["Invalid", "List"])

def test_generate_lhs_empty_input():
    """Test graceful handling of empty inputs."""
    assert generate_lhs(10, {}).empty
    assert generate_lhs(10, pd.DataFrame()).empty
