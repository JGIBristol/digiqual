import numpy as np
import pandas as pd
from scipy.stats import qmc

def generate_synthetic_ndt_data(n_samples=1000, seed=42):
    """
    Generates a 7-parameter synthetic dataset for testing PoD algorithms.
    """
    np.random.seed(seed)

    bounds = {
        "Defect_Length": (0.0, 10.0),
        "Defect_Depth": (1.0, 5.0),
        "Sensor_Angle": (-15.0, 15.0),
        "Sensor_Liftoff": (0.5, 3.0),
        "Surface_Roughness": (1.0, 6.0),
        "Material_Temp": (10.0, 40.0),
        "Inspection_Speed": (5.0, 20.0)
    }

    # 1. Generate Latin Hypercube Samples
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    lhs_samples = sampler.random(n=n_samples)

    # 2. Scale samples
    df = pd.DataFrame()
    for i, (name, (min_val, max_val)) in enumerate(bounds.items()):
        df[name] = lhs_samples[:, i] * (max_val - min_val) + min_val

    # 3. Define the physical polynomial
    signal_pure = (
        5.0 +
        3.5 * df["Defect_Length"] +
        1.2 * df["Defect_Depth"] +
        0.15 * (df["Defect_Length"] * df["Defect_Depth"]) +
        -0.05 * (df["Sensor_Angle"] ** 2) +
        -2.5 * df["Sensor_Liftoff"] +
        -0.3 * df["Surface_Roughness"] +
        0.02 * df["Material_Temp"] +
        -0.1 * df["Inspection_Speed"]
    )

    # 4. FIXED: Add heteroscedastic noise safely
    # We use np.abs() to scale with magnitude, and np.maximum() to enforce a floor > 0
    noise_std = np.maximum(0.1, 0.5 + (0.1 * np.abs(signal_pure)))
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n_samples)

    # 5. Calculate final outcome
    df["Signal_Amplitude"] = signal_pure + noise

    return df.round(3)

# Generate the data
df_synthetic = generate_synthetic_ndt_data(n_samples=1500)

# Save it
df_synthetic.to_csv("synthetic_7_param_data.csv", index=False)
print("Data generated successfully! Saved to 'synthetic_7_param_data.csv'.")
