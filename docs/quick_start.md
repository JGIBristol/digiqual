# Quick Start

DigiQual offers two ways to work: the Functional Approach (great for specific tasks) and the Class-Based Approach (recommended for full study management).

## Option 1: Functional Approach (Manual Control)

Use individual functions if you just need to generate samples or check a dataframe you already have.

### Generating an Experimental Design

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

### Validating Simulation Data

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

### Checking Sample Sufficiency
We have validated data so now we want to check if we have enough samples to produce an accurate PoD Curve.

``` python
ss = sample_sufficiency(
    df=df_clean,
    input_cols=["Length", "Angle", "Roughness"],
    outcome_col="Signal"
)
print(ss)
```

## Option 2: Class-Based Approach (Streamlined)

Use the `SimulationStudy` manager to handle data storage, validation, and diagnostics in one place. This prevents errors caused by passing the wrong dataframe between functions.

```python
from numpy.random import default_rng
import pandas as pd
import digiqual as dq

rng = default_rng(123)  # instantiate a Generator (seeded for reproducibility)

#### Data Creation ####

# Create sample framework using generate_lhs()
vars_df = pd.DataFrame(
    [
        {"Name": "Length", "Min": 0.1, "Max": 10},
        {"Name": "Angle", "Min": -90, "Max": 90},
        {"Name": "Roughness", "Min": 0, "Max": 1},
    ]
)

df = dq.generate_lhs(n=1000, seed=123, vars_df=vars_df)

rng=default_rng(123)

# Create a fake signal output column with some noise
df['Signal'] = (df['Length'] * df['Roughness']) + rng.uniform(-1, 1, size=len(df))

#### Class - SimulationStudy ####

# 1. Define the study
study = dq.SimulationStudy(
    input_cols=["Length", "Roughness"],
    outcome_col="Signal"
)

# 2. Add the Data
study.add_data(df)

# 3. Validate & Test
study.diagnose()
```
