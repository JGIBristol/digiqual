import pandas as pd
import numpy as np

def generate_fake_data(filename="initial_data.csv", n=50):
    """Generates a small dataset that might FAIL diagnostics (for testing the 'Fix' loop)."""
    np.random.seed(42)

    # 1. Generate Inputs (Small N = likely gaps)
    df = pd.DataFrame({
        'Length': np.random.uniform(0, 10, n),
        'Angle': np.random.uniform(-45, 45, n)
    })

    # 2. Physics & Noise
    base_signal = (df['Length'] * 2.0) - (0.1 * df['Angle'].abs())
    noise_scale = 0.5 + (0.1 * df['Length'])
    noise = np.random.normal(loc=0, scale=noise_scale, size=n)

    df['Signal'] = np.abs(base_signal + noise)

    df.to_csv(filename, index=False)
    print(f"✅ Created '{filename}' with {n} rows (likely to have issues).")


def updated_data(filename="sufficient_data.csv", n=200):
    """Generates a large dataset that should PASS all diagnostics."""
    np.random.seed(999) # Different seed

    # 1. Generate Inputs (Large N = good coverage)
    df = pd.DataFrame({
        'Length': np.random.uniform(0, 10, n),
        'Angle': np.random.uniform(-45, 45, n)
    })

    # 2. Physics & Noise
    base_signal = (df['Length'] * 2.0) - (0.1 * df['Angle'].abs())
    noise_scale = 0.5 + (0.1 * df['Length'])
    noise = np.random.normal(loc=0, scale=noise_scale, size=n)

    df['Signal'] = np.abs(base_signal + noise)

    df.to_csv(filename, index=False)
    print(f"✅ Created '{filename}' with {n} rows (should pass checks).")

if __name__ == "__main__":
    # You can comment out the one you don't want, or run both
    generate_fake_data()
    updated_data()
