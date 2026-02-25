import pandas as pd
import numpy as np

def generate_fake_data(filename="app/initial_data.csv", n=25):
    """Fails due to massive Gaps and Skewed Heteroskedasticity."""
    # 1. Deliberate Gap (0-2 and 8-10)
    lengths = np.concatenate([np.random.uniform(0, 2, 12), np.random.uniform(8, 10, 13)])
    df = pd.DataFrame({
        'Length': lengths,
        'Angle': np.random.uniform(-45, 45, n)
    })

    # 2. Monotonic Physics + Skewed Gamma Noise
    # As Length increases, the 'scale' of the Gamma noise increases (Heteroskedasticity)
    base_signal = 10.0 + 1.5 * df['Length'] + 0.2 * (df['Length']**2)

    # Non-normal noise: Gamma distribution is always positive and skewed
    noise_scale = 0.5 + (0.8 * df['Length'])
    noise = np.random.gamma(shape=2.0, scale=noise_scale, size=n)

    df['Signal'] = base_signal + noise
    df.to_csv(filename, index=False)
    print(f"✅ Created '{filename}' (N={n}). Should fail Gap and Bootstrap.")

def updated_data(filename="app/sufficient_data.csv", n=1500):
    """Passes because high N overcomes the skewed noise."""
    df = pd.DataFrame({
        'Length': np.random.uniform(0, 10, n),
        'Angle': np.random.uniform(-45, 45, n)
    })

    base_signal = 10.0 + 1.5 * df['Length'] + 0.2 * (df['Length']**2)
    noise_scale = 0.5 + (0.8 * df['Length'])
    noise = np.random.gamma(shape=2.0, scale=noise_scale, size=n)

    df['Signal'] = base_signal + noise
    df.to_csv(filename, index=False)
    print(f"✅ Created '{filename}' (N={n}). Should pass all tests.")

if __name__ == "__main__":
    generate_fake_data()
    updated_data()
