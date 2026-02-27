"""
digiqual Demo: Automated Optimisation Loop
Run this script to see the package automatically generate and refine a design.
"""
import sys
import os
from digiqual.core import SimulationStudy

def main():
    print("--- 1. Setting up Solver Command ---")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    solver_script = os.path.join(current_dir, "solver.py")
    solver_cmd = f"{sys.executable} {solver_script} {{input}} {{output}}"

    print(f"Solver Command: {solver_cmd}")

    print("\n--- 2. Initializing Study ---")
    ranges = {
        "Length": (0.0, 10.0),
        "Angle": (-45.0, 45.0)
    }

    study = SimulationStudy(
        input_cols=["Length", "Angle"],
        outcome_col="Signal"
    )

    print("\n[!] NOTE: The mock solver is rigged to fail if Length > 8.0 AND Angle > 30.0.")
    print("    Watch the console output to see the Graveyard actively block these regions!")

    print("\n--- 3. Running optimise() ---")
    study.optimise(
        command=solver_cmd,
        ranges=ranges,
        n_start=50,
        n_step=20,
        max_iter=100
    )

    print("\n--- 4. Results ---")
    print(f"Final Dataset Size: {len(study.data)} valid rows")

    print("\n--- 5. Visualisation ---")
    try:
        # We assume a threshold of 15.0 based on our physics (max length 10 * 2 = 20)
        study.pod(poi_col="Length", threshold=7.5)
        study.visualise(show=True)
    except Exception as e:
        print(f"Could not plot: {e}")

if __name__ == "__main__":
    main()
