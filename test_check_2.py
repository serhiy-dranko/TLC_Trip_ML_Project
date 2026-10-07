import pandas as pd
df = pd.read_parquet("data/processed/train.parquet")
print(df.columns)