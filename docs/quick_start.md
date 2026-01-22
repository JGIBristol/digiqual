# Quick Start

## Generating an Experimental Design

Create a Latin Hypercube design for a simulation study involving defect size (a) and roughness (σR).

```python
import pandas as pd
from digiqual.sampling import generate_lhs

# Define your variables and bounds
vars_df = pd.DataFrame({
    "Name": ["Defect_Size", "Roughness_RMS"],
    "Min": [0.1, 50],
    "Max": [3.0, 100]
})

# Generate 50 samples
design = generate_lhs_design(n=50, vars_df=vars_df, seed=42)
print(design.head())
```

## Validating Simulation Data

Once you have your simulation results, ensure they are ready for PoD analysis.

```python
from digiqual.diagnostics import validate_simulation

# Assume 'results_df' is your dataframe containing simulation outputs
report = validate_data(
    df=results_df,
    input_cols=["Defect_Size", "Roughness_RMS"],
    outcome_col="Signal_Amplitude"
)

if report["valid"]:
    print("Data is ready for analysis!")
else:
    print(f"Validation failed: {report['message']}")
```
