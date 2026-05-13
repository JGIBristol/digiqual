import numpy as np
from digiqual.core import SimulationStudy
from digiqual.executors import PythonExecutor

# 1. Define our "External" Solver
def mock_sensor_model(row):
    """
    A realistic simulated physics model with a complex cubic trend,
    a length-angle interaction, roughness attenuation, heteroscedastic noise,
    and skewed errors.
    """
    length = row['Length']
    angle = row['Angle']
    roughness = row['Roughness']

    # 1. THE DEAD ZONE (Trigger Graveyard Tracking)
    if 4.0 < length < 6.0 and abs(angle) > 30:
        return np.nan

    # 2. BASE SIGNAL (Cubic Trend + Interaction + Attenuation)
    # Roughness absorbs and scatters the signal, lowering the mean response.
    base_signal = (
        5.0
        + (3.0 * length)
        - (0.8 * (length ** 2))
        + (0.1 * (length ** 3))
        + (angle * 0.1)
        - (0.05 * length * abs(angle))
        - (roughness * 5.0)
    )

    # 3. HETEROSCEDASTIC, NON-NORMAL NOISE
    # Roughness also makes the signal noisier and harder to read.
    noise_scale = 0.5 + (length * 0.4) + (roughness * 1.0)

    noise = np.random.gumbel(loc=0, scale=noise_scale)
    noise -= (noise_scale * 0.57721)

    return base_signal + noise

def run_demo():
    print("Starting DigiQual Live Demo...\n")

    # 2. Configure the Study with the new Roughness parameter
    outcome = 'Signal'
    ranges = {
        'Length': (0.0, 10.0),
        'Angle': (-45.0, 45.0),
        'Roughness': (0.0, 1.0)
    }

    study = SimulationStudy()

    print("Setting up the Python Executor...")
    executor = PythonExecutor(solver_func=mock_sensor_model, outcome_col=outcome)

    # 3. Run the Automated Optimisation Loop
    # We increased the start points slightly to account for the extra dimension
    print("\n--- Kicking off Auto-Pilot Optimisation ---")
    study.optimise(
        executor=executor,
        ranges=ranges,
        outcome_col=outcome,
        n_start=40,
        n_step=10,
        max_iter=10
    )

    print("\n--- Optimisation Complete ---")
    print(f"Total valid simulations collected: {len(study.data)}")

    # 4. Run PoD Analysis (The Ultimate Test)
    print("\n--- Running Probability of Detection (PoD) Analysis ---")
    try:
        # We define Length and Angle as the PoIs to get the 3D surface plots,
        # but we pass Roughness as a nuisance parameter to trigger the Monte Carlo integration!
        _ = study.pod(
            poi_col=["Length", "Angle"],
            nuisance_col="Roughness",
            threshold=20.0,
            n_jobs=-1
        )

        # 5. Visualise the Results
        print("\nGenerating plots... (Close the plot windows to end the script)")
        study.visualise(show=True)
    except Exception as e:
        print(f"\n[!] PoD Analysis failed. Error: {e}")

if __name__ == "__main__":
    run_demo()
