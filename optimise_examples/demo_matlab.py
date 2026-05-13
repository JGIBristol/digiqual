from digiqual.core import SimulationStudy
from digiqual.executors import MatlabExecutor
import os
from pathlib import Path

def run_matlab_demo():
    # --- NEW: Force Python to run in this script's directory ---
    script_dir = Path(__file__).parent
    os.chdir(script_dir)


    print("Starting DigiQual MATLAB Executor Demo...\n")

    # 1. Configure the Study
    ranges = {'Length': (0.0, 10.0), 'Angle': (-45.0, 45.0), 'Roughness': (0.0, 1.0)}
    study = SimulationStudy()

    # 2. Setup the MATLAB Executor
    # We just provide the name of the MATLAB function (without the .m extension)
    # The executor will automatically build the complex headless terminal command!
    print("Setting up MATLAB Executor...")
    executor = MatlabExecutor(wrapper_name="dummy_matlab_solver")

    # 3. Run the Automated Optimisation Loop
    print("\n--- Kicking off Auto-Pilot ---")
    study.optimise(
        executor=executor,
        ranges=ranges,
        outcome_col='Signal',
        n_start=40,
        n_step=10,
        max_iter=3
    )

    # 4. Run PoD Analysis and Visualise
    print("\n--- Optimisation Complete. Running PoD ---")
    try:
        # Running the 3D surface with Monte Carlo integration on all available cores
        _ = study.pod(
            poi_col=["Length", "Angle"],
            nuisance_col="Roughness",
            threshold=15.0,
            n_jobs=-1
        )
        study.visualise(show=True)
    except Exception as e:
        print(f"\n[!] PoD Analysis failed. Error: {e}")

if __name__ == "__main__":
    run_matlab_demo()
