import numpy as np
import pandas as pd
from digiqual.core import SimulationStudy

print("Generating synthetic non-linear data...")
# 1. Generate Non-linear Data (Sigmoid Curve)
# This shape is difficult for polynomials but perfect for Kriging.
np.random.seed(42)
flaw_sizes = np.linspace(0.1, 10.0, 150)

# Sigmoid function: plateaus at the top and bottom
true_responses = 20 / (1 + np.exp(-1.5 * (flaw_sizes - 5)))
# Add noise that scales slightly with the flaw size
noise = np.random.normal(0, 1.0 + 0.1 * flaw_sizes, size=len(flaw_sizes))
responses = true_responses + noise

df = pd.DataFrame({
        'Flaw_Size': flaw_sizes,
        'Response': responses
})

# 3. Initialize the Study
print("Initializing SimulationStudy...")
study = SimulationStudy(input_cols=['Flaw_Size'], outcome_col='Response')
study.add_data(df)
study.diagnose()

# 4. Run the PoD Analysis
# We use a threshold that intersects the middle of our S-Curve (e.g., 10.0)
# Using 100 bootstrap iterations so it runs relatively quickly for testing
print("\n--- Running PoD Analysis ---")
results = study.pod(poi_col='Flaw_Size', threshold=10.0, n_boot=100)

# 5. Show the Final Visualizations
print("\n--- Generating Visualizations ---")
study.visualise()
