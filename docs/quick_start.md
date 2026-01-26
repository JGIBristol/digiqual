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

### Initialise & Load

```python
import digiqual as dq
import pandas as pd

# 1. Define the study once
study = dq.SimulationStudy(
    input_cols=["Length", "Angle", "Roughness"],
    outcome_col="Signal"
)

# 2. Add your raw data (Simulating the load here)
# You can call add_data() multiple times; the class manages the merging.
df = pd.read_csv("my_simulation_results.csv")
study.add_data(df)
```

### Run Diagnostics
The manager automatically validates your data before running diagnostics.

```python
# 3. Diagnose quality
# Returns a report on Coverage, Model Fit, and Bootstrap Convergence
report = study.diagnose()
print(report)
```
