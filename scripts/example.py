import pandas as pd
import digiqual as dq

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

df['Signal'] = df['Length']*df['Roughness']

dq.validate_simulation(df=df,input_cols=["Length","Angle","Roughness"],outcome_col="Signal")


#### Sample Sufficiency ####
