import pytest
import pandas as pd
from digiqual.lhs_design import generate_lhs_design


# 1. Test that the function works correctly under normal conditions
def test_generate_lhs_design_success():
    # Setup inputs
    vars_df = pd.DataFrame(
        [
            {"Name": "Var1", "Min": 0, "Max": 10},
            {"Name": "Var2", "Min": 100, "Max": 200},
        ]
    )

    # Run function
    result = generate_lhs_design(n=5, vars_df=vars_df, seed=42)

    # Assertions (Checks)
    assert len(result) == 5
    assert list(result.columns) == ["Var1", "Var2"]
    assert result["Var1"].min() >= 0
    assert result["Var1"].max() <= 10
    assert not result.isnull().values.any()  # Check for no missing values


# 2. Test that the function correctly catches the "Min > Max" error
def test_generate_lhs_design_bad_bounds():
    # Setup bad inputs (Min is 50, Max is 10)
    bad_df = pd.DataFrame([{"Name": "BadVar", "Min": 50, "Max": 10}])

    # We expect this to raise a ValueError
    with pytest.raises(ValueError, match="strictly lower"):
        generate_lhs_design(n=5, vars_df=bad_df)

# 3. Test that non-numeric Min/Max triggers an error
def test_generate_lhs_design_non_numeric():
    # Setup bad inputs: 'Min' is a string "Ten" instead of number 10
    bad_df = pd.DataFrame([
        {'Name': 'Var1', 'Min': "Ten", 'Max': 20}
    ])

    # We expect a ValueError with the specific message we wrote
    with pytest.raises(ValueError, match="strictly numeric"):
        generate_lhs_design(n=5, vars_df=bad_df)

# 4. Test that non-string Names trigger an error
def test_generate_lhs_design_bad_names():
    # Setup bad inputs: 'Name' is a number (123) instead of a string
    bad_df = pd.DataFrame([
        {'Name': 123, 'Min': 10, 'Max': 20}
    ])

    # We expect a ValueError about character strings
    with pytest.raises(ValueError, match="character strings"):
        generate_lhs_design(n=5, vars_df=bad_df)
