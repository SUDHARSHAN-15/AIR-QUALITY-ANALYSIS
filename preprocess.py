# preprocess_india.py
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 1. Load hourly city data (main source)
# -------------------------------------------------
df_hour = pd.read_csv(RAW / "city_hour.csv")
df_hour['Datetime'] = pd.to_datetime(df_hour['Datetime'])
df_hour = df_hour.rename(columns={'Datetime': 'datetime_hour'})

# -------------------------------------------------
# 2. Clean missing values (Kaggle uses NaN already)
# -------------------------------------------------
pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3',
              'Benzene','Toluene','Xylene','AQI']
df_hour[pollutants] = df_hour[pollutants].fillna(0)

# -------------------------------------------------
# 3. Save hourly nationwide
# -------------------------------------------------
df_hour.to_csv(OUT / "india_hourly.csv", index=False)

# -------------------------------------------------
# 4. Daily averages per city
# -------------------------------------------------
daily = (df_hour
         .set_index('datetime_hour')
         .groupby('City')
         .resample('D')
         .mean(numeric_only=True)
         .reset_index())

daily.to_csv(OUT / "india_daily.csv", index=False)

print(f"Nation-wide data ready!")
print(f"   Hourly rows : {len(df_hour):,}")
print(f"   Daily  rows : {len(daily):,}")
print(f"Saved to {OUT}")