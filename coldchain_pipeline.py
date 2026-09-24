"""
ColdChain AI - core analytics pipeline
Prototype uses a clearly-labelled simulated cold-chain sensor dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA = Path("data/cold_chain_sensor_data.csv")
OUT = Path("data/cold_chain_scored.csv")

df = pd.read_csv(DATA, parse_dates=["timestamp"])

# Domain rule: strawberries prototype target band
LOWER_LIMIT = 2.0
UPPER_LIMIT = 8.0

df["out_of_range"] = (
    (df["temperature_c"] < LOWER_LIMIT) |
    (df["temperature_c"] > UPPER_LIMIT)
).astype(int)

# Features describing sensor behaviour
df = df.sort_values(["shipment_id", "sensor_position", "timestamp"])
g = df.groupby(["shipment_id", "sensor_position"], group_keys=False)

df["temp_change"] = g["temperature_c"].diff().fillna(0)
df["rolling_mean_60m"] = g["temperature_c"].transform(
    lambda x: x.rolling(6, min_periods=1).mean()
)
df["rolling_std_60m"] = g["temperature_c"].transform(
    lambda x: x.rolling(6, min_periods=1).std().fillna(0)
)
df["abs_deviation"] = (df["temperature_c"] - df["rolling_mean_60m"]).abs()

features = [
    "temperature_c", "humidity_pct", "temp_change",
    "rolling_std_60m", "abs_deviation"
]
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

model = IsolationForest(
    n_estimators=150,
    contamination=0.06,
    random_state=42
)
df["ai_anomaly"] = (model.fit_predict(X) == -1).astype(int)
df["anomaly_score"] = -model.score_samples(X)

# Explainable risk score: combines domain excursion + AI anomaly
df["risk_score"] = (
    0.55 * np.clip((df["temperature_c"] - UPPER_LIMIT).abs() / 6, 0, 1)
    + 0.30 * np.clip(df["rolling_std_60m"] / 2, 0, 1)
    + 0.15 * df["ai_anomaly"]
) * 100

df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[-1, 25, 55, 101],
    labels=["Low", "Medium", "High"]
)

df.to_csv(OUT, index=False)
print("Scored dataset saved to:", OUT)
print(df[["risk_level","ai_anomaly","out_of_range"]].value_counts().head())
