from numpy.random import default_rng
import pandas as pd
import digiqual as dq

rng = default_rng(123)  # instantiate a Generator (seeded for reproducibility)

#### Data Creation ####

vars_df = pd.DataFrame(
    [
        {"Name": "Length", "Min": 0.1, "Max": 10},
        {"Name": "Angle", "Min": -90, "Max": 90},
        {"Name": "Roughness", "Min": 0, "Max": 1},
    ]
)

df = dq.generate_lhs(n=1000, seed=123, vars_df=vars_df)

rng=default_rng(123)

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
