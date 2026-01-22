from numpy.random import default_rng
import pandas as pd
import digiqual as dq

rng = default_rng()  # instantiate a Generator (seeded for reproducibility)

#### Data Creation ####

vars_df = pd.DataFrame(
    [
        {"Name": "Length", "Min": 0.1, "Max": 10},
        {"Name": "Angle", "Min": -90, "Max": 90},
        {"Name": "Roughness", "Min": 0, "Max": 1},
    ]
)

df = dq.generate_lhs(n=1000, seed=123, vars_df=vars_df)


#### Data Validation ####

# Here we create a Signal variable that we would usually collect from the simulations. We add some noise to showcase the validate_simulation function.
df['Signal'] = (df['Length'] * df['Roughness']) + rng.uniform(-1, 1, size=len(df))

df_clean, df_removed = dq.validate_simulation(df=df, input_cols=["Length", "Angle", "Roughness"], outcome_col="Signal")

len(df_clean)
len(df_removed)

#### Sample Sufficiency ####
