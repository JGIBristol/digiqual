# Quick Start

DigiQual offers two ways to work: the **Functional Approach** (great for specific tasks) and the **Class-Based Approach** (recommended for full study management).

## Option 1: Functional Approach (Manual Control)

In this scenario, we generate a high-quality Latin Hypercube design. We will see that the active learning module confirms the design is sufficient and requires no further sampling.

### Generating an Experimental Design

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

### Validating Simulation Data

Once you have your simulation results, ensure they are ready for PoD analysis.

``` python
from digiqual.diagnostics import validate_simulation
from numpy.random import default_rng

# Create a fake signal output column with some noise
rng = default_rng(123)
df['Signal'] = (df['Length'] * df['Roughness']) + rng.uniform(-1, 1, size=len(df))

df_clean, df_removed = validate_simulation(
    df=df,
    input_cols=["Length", "Angle", "Roughness"],
    outcome_col="Signal"
)

print(f"Valid rows: {len(df_clean)}")
print(f"Dropped rows: {len(df_removed)}")
```

### Checking Sample Sufficiency
We have validated data so now we want to check if we have enough samples to produce an accurate PoD Curve.

``` python
from digiqual.diagnostics import sample_sufficiency

ss = sample_sufficiency(
    df=df_clean,
    input_cols=["Length", "Angle", "Roughness"],
    outcome_col="Signal"
)
print(ss)
```

### Adaptive Refinement Check
We now run the targeted sampler. Because `generate_lhs` provides good coverage by default, we expect the adaptive module to return an empty result, confirming no more work is needed.

```python
from digiqual.adaptive import generate_targeted_samples

new_samples = generate_targeted_samples(
    df=df_clean,
    input_cols=["Length", "Angle","Roughness"],
    outcome_col="Signal",
    n_new_per_fix=5
)
```


## Option 2: Class-Based Approach (Streamlined)

Use the `SimulationStudy` manager to handle the entire lifecycle: storage, diagnostics, and active refinement. This allows you to automatically fix issues in your design.

In this example, we will intentionally feed the study "bad" data (with a large gap) to see how it identifies and fixes the problem.

```python
import numpy as np
import pandas as pd
import digiqual as dq

# --- 1. Create a "Flawed" Dataset ---
# We simulate a scenario where we forgot to simulate lengths between 3.0 and 7.0
# Range is 0 to 10, so a gap of 4.0 is huge (40% of the domain).
df_part1 = pd.DataFrame({'Length': np.random.uniform(0, 3, 10), 'Angle': np.random.uniform(0, 45, 10)})
df_part2 = pd.DataFrame({'Length': np.random.uniform(7, 10, 10), 'Angle': np.random.uniform(0, 45, 10)})
df = pd.concat([df_part1, df_part2], ignore_index=True)

# Add a dummy signal
df['Signal'] = df['Length'] * 2 + 10

# --- 2. Initialize the Study ---
study = dq.SimulationStudy(
    input_cols=["Length", "Angle"],
    outcome_col="Signal"
)
study.add_data(df)

# --- 3. Diagnose & Adapt ---
print(study.diagnose())
# You will see 'Input Coverage' fails for 'Length' due to the gap.

# --- 4. Refine (The Fix) ---
# The study automatically generates points specifically to fill the gap.
new_points = study.refine(n_points=5)

print(new_points)

# Verify: The new Length values will be between 3.0 and 7.0
```
