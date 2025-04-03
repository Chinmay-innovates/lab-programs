import pandas as pd

df = pd.read_csv("book.csv")

print(df.isnull().sum())

# check for missing values
df.ffill(inplace=True)

# writing missing values
print(df.isnull().sum())
df.to_csv("book2.csv")
