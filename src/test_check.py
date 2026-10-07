import pandas as pd

df = pd.read_parquet("data/interim/trips_fhv_2026-01.parquet")

print(df.head(15))       
print(df.shape)          
print(df.dtypes)         
print(df.isna().mean())  