# Quick Start

## Generating an Experimental Design

Create a Latin Hypercube design for a simulation study involving defect size ($a$), angle ($\theta$) and roughness ($\sigma_{R}$).

```python
import pandas as pd
from digiqual.sampling import generate_lhs

# Define your variables and bounds
vars_df = pd.DataFrame(
    [
        {"Name": "Length", "Min": 0.1, "Max": 10},
        {"Name": "Angle", "Min": -90, "Max": 90},
        {"Name": "Roughness", "Min": 0, "Max": 1},
    ]
)

# Generate 1000 samples
df = generate_lhs(n=1000, seed=123, vars_df=vars_df)
print(df.head())
```

## Validating Simulation Data

Once you have your simulation results, ensure they are ready for PoD analysis.

```python
from digiqual.diagnostics import validate_simulation

# For this example, we can take the dataframe
#  we created above and add a synthetic result

df['Signal'] = df['Length']*df['Roughness']

report = validate_simulation(
    df=df,
    input_cols=["Length","Angle","Roughness"],
    outcome_col="Signal
)

if report["valid"]:
    print("Data is ready for analysis!")
else:
    print(f"Validation failed: {report['message']}")
```
