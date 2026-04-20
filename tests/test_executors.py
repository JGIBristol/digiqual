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
