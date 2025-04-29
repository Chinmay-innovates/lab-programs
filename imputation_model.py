import numpy as np
import pandas as pd

data = {'Score': [np.nan, 90, 80, 85, np.nan, 88, 92, np.nan, 87, 90, 80, 78]}

df = pd.DataFrame(data)

df['Score_Mean'] = df['Score'].fillna(df['Score'].mean())
df['Score_Median'] = df['Score'].fillna(df['Score'].median())
df['Score_Mode'] = df['Score'].fillna(df['Score'].mode()[0])

df.to_csv('imputation_model.csv', index=False)

print(df)
