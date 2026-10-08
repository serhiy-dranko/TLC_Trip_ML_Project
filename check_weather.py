import pandas as pd
from pathlib import Path


import yaml
config_path = Path("config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

processed = Path(config["paths"]["processed"])


train = pd.read_parquet(processed / "train.parquet")
val = pd.read_parquet(processed / "val.parquet")

print("=" * 80)
print("WEATHER DATA ANALYSIS")
print("=" * 80)

# ========================================
# TRAIN SET
# ========================================
print("\n📊 TRAIN SET:")
print(f"Total rows: {len(train)}")

# Snowfall
print(f"\nSnowfall column:")
print(f"  Min: {train['snowfall'].min()}")
print(f"  Max: {train['snowfall'].max()}")
print(f"  Mean: {train['snowfall'].mean():.4f}")
print(f"  Rows with snowfall > 0: {(train['snowfall'] > 0).sum()}")
print(f"  Rows with snowfall == 0: {(train['snowfall'] == 0).sum()}")

# Precipitation
print(f"\nPrecipitation column:")
print(f"  Min: {train['precipitation'].min()}")
print(f"  Max: {train['precipitation'].max()}")
print(f"  Mean: {train['precipitation'].mean():.4f}")
print(f"  Rows with precipitation > 0: {(train['precipitation'] > 0).sum()}")
print(f"  Rows with precipitation == 0: {(train['precipitation'] == 0).sum()}")

# ========================================
# VALIDATION SET
# ========================================
print("\n" + "=" * 80)
print("📊 VALIDATION SET:")
print(f"Total rows: {len(val)}")

# Snowfall
print(f"\nSnowfall column:")
print(f"  Min: {val['snowfall'].min()}")
print(f"  Max: {val['snowfall'].max()}")
print(f"  Mean: {val['snowfall'].mean():.4f}")
print(f"  Rows with snowfall > 0: {(val['snowfall'] > 0).sum()}")
print(f"  Rows with snowfall == 0: {(val['snowfall'] == 0).sum()}")

# Precipitation
print(f"\nPrecipitation column:")
print(f"  Min: {val['precipitation'].min()}")
print(f"  Max: {val['precipitation'].max()}")
print(f"  Mean: {val['precipitation'].mean():.4f}")
print(f"  Rows with precipitation > 0: {(val['precipitation'] > 0).sum()}")
print(f"  Rows with precipitation == 0: {(val['precipitation'] == 0).sum()}")

# ========================================
# COMBINED ANALYSIS
# ========================================
print("\n" + "=" * 80)
print("🌡️ COMBINED ANALYSIS (TRAIN + VAL):")

combined = pd.concat([train, val])

print(f"\nSnowfall distribution:")
print(f"  Days with snow: {(combined['snowfall'] > 0).sum()}")
print(f"  Days without snow: {(combined['snowfall'] == 0).sum()}")
print(f"  Total: {len(combined)}")

# Top 10 snowfall values
print(f"\nTop 10 snowfall values:")
top_snow = combined['snowfall'].nlargest(10).values
if top_snow[0] > 0:
    print(f"  {top_snow}")
else:
    print(f"  All zeros - NO SNOW IN DATASET!")

# ========================================
# CONCLUSION
# ========================================
print("\n" + "=" * 80)
if (combined['snowfall'] > 0).sum() == 0: 
    print("❌ OUTPUT: There is NO snow in the dataset (snowfall > 0)") 
    print("It's normal if the data is from warm months!")
else: 
    print(f" ✅ CONCLUSION: There IS snow in the dataset!") 
    print(f" {(combined['snowfall'] > 0).sum()} term with snowfall > 0")

print("=" * 80)