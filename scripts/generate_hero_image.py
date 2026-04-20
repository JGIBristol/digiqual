import os
import matplotlib.pyplot as plt
import numpy as np
from digiqual.core import SimulationStudy
from digiqual.executors import PythonExecutor

# Define the location for the image
materials_dir = os.path.join('docs', 'materials')
image_path = os.path.join(materials_dir, 'hero_pod_surface.png')

# Create the materials directory if it doesn't exist
if not os.path.exists(materials_dir):
    os.makedirs(materials_dir)
    print(f"Created directory: {materials_dir}")

# Define the "External" Solver
def complex_physics_model(row):
    length = row['Length']
    angle = row['Angle']

    # 1. Base Signal (Cubic Trend + Interaction)
    base_signal = (
        5.0
        + (3.0 * length)
        - (0.8 * (length ** 2))
        + (0.1 * (length ** 3))
        + (angle * 0.1)
        - (0.05 * length * abs(angle))
    )

    # 2. Heteroscedastic, Non-Normal Noise (Gumbel)
    noise_scale = 0.5 + (length * 0.4)
    noise = np.random.gumbel(loc=0, scale=noise_scale)
    noise -= (noise_scale * 0.57721) # Correct for non-zero mean

    return base_signal + noise

def generate_hero_image():
    print("Generating the PoD data for the hero image...")
    # 1. Configure the Study
    ranges = {'Length': (0.0, 10.0), 'Angle': (-45.0, 45.0)}
    study = SimulationStudy(input_cols=['Length', 'Angle'], outcome_col='Signal')
    executor = PythonExecutor(solver_func=complex_physics_model, outcome_col='Signal')

    # 2. Run the Automated Optimisation Loop
    print("Running optimization...")
    study.optimise(
        executor=executor,
        ranges=ranges,
        n_start=40,      # Slightly more starting points for a denser grid
        n_step=10,
        max_iter=3
    )

    # 3. Run PoD Analysis and Visualise
    print("Running PoD analysis...")
    try:
        _ = study.pod(poi_col=["Length", "Angle"], threshold=30.0, n_jobs=-1)

        # 4. Generate and Save the Plot
        print("Saving plot...")
        study.visualise(show=False) # Plot to buffer
        plt.savefig(image_path, dpi=150, bbox_inches='tight') # Save with good resolution
        plt.close() # Clean up memory
        print(f"Successfully saved image to: {image_path}")

    except Exception as e:
        print(f"\n[!] Image generation failed. Error: {e}")

if __name__ == "__main__":
    generate_hero_image()
