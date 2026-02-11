"""
Mock Solver Script (solver.py)

This script mimics an external engineering tool (like Ansys or Abaqus).
It takes an input CSV, runs a "physics" calculation, and writes an output CSV.

Usage:
    python solver.py <input_file> <output_file>
"""
import sys
import pandas as pd
import numpy as np

def run_simulation(input_path, output_path):
    # 1. Read Inputs
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Solver Error: Could not read {input_path}. {e}")
        sys.exit(1)

    # 2. The "Physics" (Signal = 2*Length + Noise)
    signal = (df['Length'] * 2.0) - (0.1 * np.abs(df['Angle']))
    noise_scale = 0.5 + (0.5 * df['Length'])
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(df))

    df['Signal'] = np.abs(signal + noise)

    # 3. Write Outputs
    try:
        df.to_csv(output_path, index=False)
        print(f"Solver: Successfully processed {len(df)} rows.")
    except Exception as e:
        print(f"Solver Error: Could not write {output_path}. {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure correct arguments
    if len(sys.argv) != 3:
        print("Usage: python solver.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    run_simulation(input_csv, output_csv)
