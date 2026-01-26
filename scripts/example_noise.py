from numpy.random import default_rng
import pandas as pd
import digiqual as dq

rng = default_rng(123)

# Define variables
vars_df = pd.DataFrame([
    {"Name": "Length", "Min": 0.0, "Max": 10.0},
    {"Name": "Angle", "Min": -90.0, "Max": 90.0},
    {"Name": "Roughness", "Min": 0.0, "Max": 1.0},
])

#### INSUFFICIENT DATA (Small N, Gaps, High Noise) ####

df_bad = dq.generate_lhs(n=50, seed=42, vars_df=vars_df)

# Artificially create a "Gap" failure
df_bad = df_bad[~df_bad['Length'].between(4.0, 6.0)].copy()

# Create a noisy Signal to trigger Model Fit & Bootstrap failure
# Signal is weak, Noise is strong (Standard Dev = 5)
noise = rng.normal(loc=0, scale=5, size=len(df_bad))
df_bad['Signal'] = (df_bad['Length'] * 0.5) + (df_bad['Roughness'] * 2) + 10 + noise

# Run Diagnostics
df_bad_clean, _ = dq.validate_simulation(df_bad, ["Length", "Angle", "Roughness"], "Signal")

results_bad = dq.sample_sufficiency(
    df=df_bad_clean,
    input_cols=["Length", "Angle", "Roughness"],
    outcome_col="Signal"
)

# Display failures
print(results_bad)
