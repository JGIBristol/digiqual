import sys
import pandas as pd
import numpy as np
import time
import math

def run_simulation(input_csv, output_csv):
    """
    This script simulates an external C++ or Fortran physics solver.
    It now includes the complex cubic trends, interactions, and skewed noise.
    """
    print(f"   [Dummy FEA] Booting up... Reading {input_csv}")

    # 1. Read the inputs provided by DigiQual
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("   [Dummy FEA] Error: Input file not found!")
        sys.exit(1)

    # 2. Simulate heavy computation time
    time.sleep(1.0)

    # 3. Do the "Physics"
    signals = []
    for _, row in df.iterrows():
        length = row['Length']
        angle = row['Angle']

        # Safely grab Roughness if it exists, otherwise default to 0.0
        # This prevents crashes if you run the older 2-variable demo script!
        roughness = row.get('Roughness', 0.0)

        # A) THE DEAD ZONE (Trigger Graveyard Tracking)
        if 4.0 < length < 6.0 and abs(angle) > 30:
            signals.append(np.nan)
            continue # Skip the rest of the math for this row

        # B) BASE SIGNAL (Cubic Trend + Interaction + Attenuation)
        base_signal = (
            5.0
            + (3.0 * length)
            - (0.5 * (length ** 2))
            + (0.1 * (length ** 3))
            + (angle * 0.1)
            - (math.sin(math.radians(angle*2))*10)
            - (0.05 * length * abs(angle))
            - (roughness * 5.0)
        )

        # C) HETEROSCEDASTIC, NON-NORMAL NOISE
        noise_scale = 0.5 + (length * 0.4) + (roughness * 1.0)

        noise = np.random.gumbel(loc=0, scale=noise_scale)
        noise -= (noise_scale * 0.57721)

        signals.append(base_signal + noise)

    df['Signal'] = signals

    # 4. Save the results back to the hard drive
    print(f"   [Dummy FEA] Solving complete. Saving to {output_csv}")
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # The OS passes the command line arguments to sys.argv
    # sys.argv[1] will be {input}, sys.argv[2] will be {output}
    if len(sys.argv) != 3:
        print("Usage: python dummy_fea.py <input_csv> <output_csv>")
        sys.exit(1)

    run_simulation(sys.argv[1], sys.argv[2])
