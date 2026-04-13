import numpy as np
import pandas as pd

def generate_complex_polynomial_dataset(num_samples: int = 1000, filename: str = "example_4D_poly_data.csv") -> pd.DataFrame:
    """
    Generates a synthetic dataset with 2 PoIs and 2 Nuisance parameters.
    The underlying relationship is a polynomial curve to test model fitting.
    """
    print(f"Generating synthetic polynomial dataset with {num_samples} samples...")
    np.random.seed(42)

    # ==========================================
    # 1. Parameters of Interest (PoIs)
    # ==========================================
    defect_length = np.random.uniform(1.0, 10.0, num_samples)
    defect_angle = np.random.uniform(-45.0, 45.0, num_samples)

    # ==========================================
    # 2. Nuisance Parameters
    # ==========================================
    sensor_liftoff = np.random.uniform(0.0, 1.5, num_samples)
    surface_roughness = np.random.uniform(1.0, 6.0, num_samples)

    # ==========================================
    # 3. Physics Simulation (Polynomial Model)
    # ==========================================
    # PoI 1: Quadratic relationship for length (e.g., Signal = 0.6*L^2 + 1.2*L)
    base_signal = 0.6 * (defect_length ** 2) + 1.2 * defect_length

    # PoI 2: Quadratic penalty for angle (inverted parabola, max at 0 degrees)
    angle_effect = 1.0 - 0.0004 * (defect_angle ** 2)

    # Interaction Term: The angle matters more for longer defects
    interaction = 0.02 * defect_length * np.abs(defect_angle)

    # Nuisance Effects (Exponential decay and linear attenuation)
    liftoff_effect = np.exp(-0.8 * sensor_liftoff)
    roughness_effect = 1.0 - (surface_roughness * 0.03)

    # Combine into the deterministic polynomial surface
    deterministic_signal = ((base_signal * angle_effect) - interaction) * liftoff_effect * roughness_effect

    # Add Measurement Noise
    noise_scale = 1.5 + (surface_roughness * 0.3)
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=num_samples)

    # Final Signal
    signal_amplitude = deterministic_signal + noise

    # ==========================================
    # 4. Package and Save
    # ==========================================
    df = pd.DataFrame({
        "Defect_Length": np.round(defect_length, 3),
        "Defect_Angle": np.round(defect_angle, 3),
        "Sensor_Liftoff": np.round(sensor_liftoff, 3),
        "Surface_Roughness": np.round(surface_roughness, 3),
        "Signal_Amplitude": np.round(signal_amplitude, 3)
    })

    # Prevent negative signals
    df["Signal_Amplitude"] = df["Signal_Amplitude"].clip(lower=0.1)

    df.to_csv(filename, index=False)
    print(f"Success! Saved to '{filename}'.")
    print("\nDataset Preview:")
    print(df.head())

    return df

if __name__ == "__main__":
    generate_complex_polynomial_dataset()
