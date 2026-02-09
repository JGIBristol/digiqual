"""
digiqual Demo: Automated Optimisation Loop
Run this script to see the package automatically generate and refine a design.
"""

from digiqual.core import SimulationStudy

def main():
    print("--- 1. Setting up Mock Physics Solver ---")
    # This command mimics an external simulation software.
    # It reads a CSV, calculates Signal = Length * 2 + Noise, and writes a CSV.
    solver_cmd = (
        "python -c "
        "'import pandas as pd, numpy as np; "
        "df=pd.read_csv(\"{input}\"); "
        "df[\"Signal\"] = df[\"Length\"] * 2.0 + np.random.normal(0, 0.5, len(df)); "
        "df.to_csv(\"{output}\", index=False)'"
    )

    print("--- 2. Initializing Study ---")
    ranges = {
        "Length": (0.0, 10.0),
        "Angle": (-45.0, 45.0)
    }

    study = SimulationStudy(
        input_cols=["Length", "Angle"],
        outcome_col="Signal"
    )

    print("--- 3. Running optimise() ---")
    # This will:
    # 1. Generate 10 LHS points
    # 2. Run the mock solver
    # 3. Diagnose quality
    # 4. Add 5 more points if quality is low (looping up to 3 times)
    study.optimise(
        command=solver_cmd,
        ranges=ranges,
        n_start=20,
        n_step=10,
        max_iter=3
    )

    print("\n--- 4. Results ---")
    print(f"Final Dataset Size: {len(study.data)} rows")
    print(study.data.head())

    print("\n--- 5. Visualisation ---")
    # Run PoD analysis on the generated data
    try:
        study.pod(poi_col="Length", threshold=15.0)
        study.visualise(show=False)
    except Exception as e:
        print(f"Could not plot: {e}")

if __name__ == "__main__":
    main()
