import pytest
import pandas as pd
import os
import subprocess
from unittest.mock import patch
from digiqual.executors import PythonExecutor, CLIExecutor, MatlabExecutor

@pytest.fixture
def sample_df():
    return pd.DataFrame({'Length': [1.0, 2.0], 'Angle': [10.0, 20.0]})

# --- PythonExecutor Tests ---
def test_python_executor_success(sample_df):
    def dummy_solver(row):
        return row['Length'] + row['Angle']

    executor = PythonExecutor(solver_func=dummy_solver, outcome_col='Signal')
    res = executor.run(sample_df)

    assert not res.empty
    assert 'Signal' in res.columns
    assert res['Signal'].iloc[0] == 11.0

def test_python_executor_crash(sample_df):
    def crash_solver(row):
        raise ValueError("Math error")

    executor = PythonExecutor(solver_func=crash_solver, outcome_col='Signal')
    res = executor.run(sample_df)

    # Should catch the error and return an empty DataFrame
    assert res.empty

# --- CLIExecutor Tests ---
@patch("subprocess.run")
def test_cli_executor_success(mock_subprocess, sample_df, tmp_path):
    input_csv = str(tmp_path / "in.csv")
    output_csv = str(tmp_path / "out.csv")

    # Create a dummy output file simulating the solver's work
    result_df = sample_df.copy()
    result_df['Signal'] = [0.99, 0.88]
    result_df.to_csv(output_csv, index=False)

    executor = CLIExecutor("echo {input} {output}", input_csv, output_csv)
    df = executor.run(sample_df)

    assert not df.empty
    assert os.path.exists(input_csv)
    mock_subprocess.assert_called_once()
    assert df['Signal'].iloc[0] == 0.99

@patch("subprocess.run")
def test_cli_executor_crash(mock_subprocess, sample_df, tmp_path):
    mock_subprocess.side_effect = subprocess.CalledProcessError(returncode=1, cmd="fail")
    executor = CLIExecutor("fail", str(tmp_path / "in.csv"), str(tmp_path / "out.csv"))

    df = executor.run(sample_df)
    assert df.empty

def test_cli_executor_missing_output(sample_df, tmp_path):
    # Runs successfully but doesn't produce the output file
    executor = CLIExecutor("echo running...", str(tmp_path / "in.csv"), str(tmp_path / "missing.csv"))
    df = executor.run(sample_df)
    assert df.empty

# --- MatlabExecutor Tests ---
def test_matlab_executor_initialization():
    executor = MatlabExecutor(wrapper_name="my_script.m") # Deliberately add .m to test the stripper

    # Check if the command template was formatted correctly
    expected_cmd = 'matlab -batch "my_script(\'{input}\', \'{output}\')" -nosplash -nodesktop'
    assert executor.command_template == expected_cmd


@patch("subprocess.run")
def test_cli_executor_auto_stitching(mock_subprocess, sample_df, tmp_path):
    """Test that CLIExecutor stitches input columns back if the solver drops them."""
    input_csv = str(tmp_path / "in.csv")
    output_csv = str(tmp_path / "out.csv")

    # Simulate solver output that ONLY has the 'Signal' column (dropped Length & Angle)
    result_df = pd.DataFrame({'Signal': [0.99, 0.88]})
    result_df.to_csv(output_csv, index=False)

    executor = CLIExecutor("echo {input} {output}", input_csv, output_csv)
    df = executor.run(sample_df)

    # Assertions
    assert not df.empty
    assert 'Signal' in df.columns
    assert 'Length' in df.columns # Proves auto-stitching worked!
    assert 'Angle' in df.columns
    assert df['Length'].iloc[0] == 1.0 # Proves the data matches the original inputs

@patch("subprocess.run")
def test_cli_executor_mismatch_failure(mock_subprocess, sample_df, tmp_path):
    """Test that CLIExecutor fails gracefully if row counts mismatch and inputs are missing."""
    input_csv = str(tmp_path / "in.csv")
    output_csv = str(tmp_path / "out.csv")

    # Simulate solver output missing inputs AND returning 3 rows instead of 2
    result_df = pd.DataFrame({'Signal': [0.99, 0.88, 0.77]})
    result_df.to_csv(output_csv, index=False)

    executor = CLIExecutor("echo {input} {output}", input_csv, output_csv)
    df = executor.run(sample_df)

    # Proves it aborted safely and returned an empty DataFrame instead of crashing
    assert df.empty
