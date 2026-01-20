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

df = dq.generate_lhs_design(n=1000, seed=123, vars_df=vars_df)

df['Signal'] = df['Length']*df['Roughness']
