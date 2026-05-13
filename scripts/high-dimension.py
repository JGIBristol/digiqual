import numpy as np
import pandas as pd

from digiqual import SimulationStudy
from digiqual.sampling import generate_lhs

# --- 1. Define the Physics ---
def apply_ndt_physics(df):
    """
    Applies the synthetic NDT physics model to a DataFrame.
    """
    # Define the physical polynomial
    signal_pure = (
        5.0 +
        1.5 * (df["Defect_Length"]**2) +
        0.5 * (df["Defect_Depth"]) +
        0.15 * (df["Defect_Length"] * df["Defect_Depth"]) +
        -0.05 * (df["Sensor_Angle"] ** 2) +
        -2.5 * df["Sensor_Liftoff"] +
        -0.3 * df["Surface_Roughness"] +
        0.02 * df["Material_Temp"] +
        -0.1 * df["Inspection_Speed"]
    )

    # Add heteroscedastic noise (noise scales with signal magnitude)
    noise_std = np.maximum(0.1, 0.5 + (0.1 * np.abs(signal_pure)))
    noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df))

    return signal_pure + noise

# --- 2. Setup Variable Ranges & Generate Data ---
vars_df = pd.DataFrame([
    {"Name": "Defect_Length",    "Min": 0.0,  "Max": 10.0},
    {"Name": "Defect_Depth",     "Min": 1.0,  "Max": 5.0},
    {"Name": "Sensor_Angle",     "Min": -15.0, "Max": 15.0},
    {"Name": "Sensor_Liftoff",   "Min": 0.5,  "Max": 3.0},
    {"Name": "Surface_Roughness","Min": 1.0,  "Max": 6.0},
    {"Name": "Material_Temp",    "Min": 10.0, "Max": 40.0},
    {"Name": "Inspection_Speed", "Min": 5.0,  "Max": 20.0}
])

# Use generate_lhs instead of manual qmc scaling
df_initial = generate_lhs(ranges=vars_df, n=1500)

# Calculate outcome signal
df_initial["Signal_Amplitude"] = apply_ndt_physics(df_initial)

# --- 3. Initialize SimulationStudy ---
input_cols = vars_df["Name"].tolist()
study = SimulationStudy()
study.add_data(df_initial,outcome_col="Signal_Amplitude")

# --- 4. Diagnosis
print("Running Diagnostics...")
study.diagnose()



# --- 5. Reliability (PoD) Analysis ---
# Marginalising environmental/noise variables
_ = study.pod(
    poi_col=["Defect_Length"],
    # nuisance_col=["Material_Temp", "Inspection_Speed", "Surface_Roughness"],
    slice_values={
        "Sensor_Angle": 0.0,
        "Defect_Depth": 3.0,
    },
    threshold=25.0,
    n_jobs=-1,
    n_boot = 0
)

# --- 6. Visualise ---
study.visualise()
