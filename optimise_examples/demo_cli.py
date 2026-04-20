from digiqual.core import SimulationStudy
from digiqual.executors import CLIExecutor

import os

def run_cli_demo():
    print("Starting DigiQual CLI Executor Demo...\n")

    # 1. Configure the Study
    ranges = {'Length': (0.0, 10.0), 'Angle': (-45.0, 45.0),'Roughness': (0, 1)}
    study = SimulationStudy(input_cols=['Length', 'Angle','Roughness'], outcome_col='Signal')

    # 2. Setup the CLI Executor
    # Figure out exactly where THIS script (demo_cli.py) lives on the computer
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the full path to the dummy script sitting next to it
    dummy_script_path = os.path.join(current_dir, "dummy_fea.py")

    # Inject that absolute path into our command template
    cmd_template = f"python {dummy_script_path} {{input}} {{output}}"

    print(f"Setting up CLI Executor with command: '{cmd_template}'")
    executor = CLIExecutor(command_template=cmd_template)

    # 3. Run the Automated Optimisation Loop
    print("\n--- Kicking off Auto-Pilot ---")
    study.optimise(
        executor=executor,
        ranges=ranges,
        n_start=20,
        n_step=10,
        max_iter=3
    )

    # 4. Run PoD Analysis and Visualise
    print("\n--- Optimisation Complete. Running PoD ---")
    try:
        _ = study.pod(poi_col=["Length", "Angle"], nuisance_col="Roughness", threshold=15.0,n_jobs=-1)
        study.visualise(show=True)
    except Exception as e:
        print(f"\n[!] PoD Analysis failed. Error: {e}")

if __name__ == "__main__":
    run_cli_demo()
