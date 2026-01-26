# Quick Start

## Generating an Experimental Design

Create a Latin Hypercube design for a simulation study involving defect size ($a$), angle ($\theta$) and roughness ($\sigma_{R}$).

``` python
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

``` python
from digiqual.diagnostics import validate_simulation
from numpy.random import default_rng

# Here we create a Signal variable that we would usually
# collect from the simulations. We add some noise to
# showcase the validate_simulation function.

rng=default_rng(123)

df['Signal'] = (df['Length'] * df['Roughness']) + rng.uniform(-1, 1, size=len(df))

df_clean, df_removed = validate_simulation(
    df=df,input_cols=["Length", "Angle", "Roughness"],outcome_col="Signal")

len(df_clean)
len(df_removed)
```
