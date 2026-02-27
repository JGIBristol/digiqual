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

    # --- THE TRAP: Create a "Danger Zone" to trigger the Graveyard ---
    # If Length > 8 and Angle > 30, the "part breaks" and returns no signal.
    danger_mask = (df['Length'] > 8.0) & (df['Angle'] > 30.0)
    df.loc[danger_mask, 'Signal'] = None

    failed_count = danger_mask.sum()

    # 3. Write Outputs
    try:
        df.to_csv(output_path, index=False)
        if failed_count > 0:
            print(f"Solver: Processed {len(df)} rows ({failed_count} crashed in the danger zone).")
        else:
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
