# Quick Start

DigiQual offers two ways to work: the **Functional Approach** (great for specific tasks) and the **Class-Based Approach** (recommended for full study management).

## Option 2: Class-Based Approach (Streamlined)

Use the `SimulationStudy` manager to handle the entire lifecycle: storage, diagnostics, and active refinement. This allows you to automatically fix issues in your design.

In this example, we will intentionally feed the study "bad" data (with a large gap) to see how it identifies and fixes the problem.

```python
import numpy as np
import pandas as pd
import digiqual as dq
import digiqual.plotting as plot
import matplotlib.pyplot as plt

# --- Define the Physics ---
# We use a lambda or function to ensure we apply the SAME physics later
def apply_physics(df):
    # 1. Base Signal: Quadratic trend (2*Length + 0.5*Length^2)
    # 2. Angle Penalty: Misalignment (-0.1*Angle) reduces signal
    signal = 10.0 + (2.0 * df['Length']) + (0.5 * df['Length']**2) - (0.1 * np.abs(df['Angle']))

    # 3. Heteroscedastic Noise: Higher roughness = More scatter
    noise_scale = 0.5 + (1.5 * df['Roughness'])
    noise = np.random.normal(loc=0, scale=noise_scale, size=len(df))

    return signal + noise

# --- Create Flawed Data (Gap between 3mm and 7mm) ---
df1 = pd.DataFrame({
    'Length': np.random.uniform(0.1, 3.0, 20),
    'Angle': np.random.uniform(-10, 10, 20),
    'Roughness': np.random.uniform(0, 0.5, 20)
})

df2 = pd.DataFrame({
    'Length': np.random.uniform(7.0, 10.0, 20),
    'Angle': np.random.uniform(-10, 10, 20),
    'Roughness': np.random.uniform(0, 0.5, 20)
})

df = pd.concat([df1, df2], ignore_index=True)
df['Signal'] = apply_physics(df)

# --- Initialize Study ---
study = dq.SimulationStudy(
    input_cols=["Length", "Angle", "Roughness"],
    outcome_col="Signal"
)
study.add_data(df)

# --- Diagnose & Adapt ---
# Check health (Will fail "Input Coverage")
print(study.diagnose())

# Generate active learning samples (20 points to fill the gap)
new_samples = study.refine(n_points=20)

if not new_samples.empty:
    print(f"\nGenerated {len(new_samples)} new samples to fix data issues.")

    # Apply the exact same physics model to the new points
    new_samples['Signal'] = apply_physics(new_samples)

    # Add back to study
    study.add_data(new_samples)

    print("\nRefinement Complete. Re-running diagnostics...")
    print(study.diagnose())

# --- Running the PoD Analysis ---
# Run the full reliability pipeline in a single call. This fits the models, determines the distribution, and runs the bootstrap.
# Run Analysis (Threshold = 18 dB)
results = study.pod(poi_col="Length", threshold=18.0)

print(f"Selected Degree: {results['mean_model'].best_degree_}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Signal Model (Physics)
local_std = dq.pod.predict_local_std(
    results['X_train'], results['residuals'], results['X_eval'], results['bandwidth']
)

plot.plot_signal_model(
    X_train=results['X_train'],
    y_train=results['y_train'],
    X_eval=results['X_eval'],
    mean_curve=results['curves']['mean_response'],
    threshold=results['threshold'],
    local_std=local_std,
    ax=ax1
)

# Plot PoD Curve (Reliability)
plot.plot_pod_curve(
    X_eval=results['X_eval'],
    pod_curve=results['curves']['pod'],
    ci_lower=results['curves']['ci_lower'],
    ci_upper=results['curves']['ci_upper'],
    target_pod=0.90,
    ax=ax2
)

plt.show()

```
