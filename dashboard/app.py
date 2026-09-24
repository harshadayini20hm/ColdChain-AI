import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="ColdChain AI", page_icon="🧊", layout="wide")
st.title("🧊 ColdChain AI")
st.caption("Intelligent Cold-Chain Temperature Excursion & Risk Analytics — prototype")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cold_chain_sensor_data.csv", parse_dates=["timestamp"])
    return df

df = load_data()
LOWER, UPPER = 2.0, 8.0

df["out_of_range"] = ((df["temperature_c"] < LOWER) | (df["temperature_c"] > UPPER)).astype(int)
df = df.sort_values(["shipment_id","sensor_position","timestamp"])
g = df.groupby(["shipment_id","sensor_position"], group_keys=False)
df["temp_change"] = g["temperature_c"].diff().fillna(0)
df["rolling_mean_60m"] = g["temperature_c"].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["rolling_std_60m"] = g["temperature_c"].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0))
df["abs_deviation"] = (df["temperature_c"] - df["rolling_mean_60m"]).abs()

features = ["temperature_c","humidity_pct","temp_change","rolling_std_60m","abs_deviation"]
X = df[features].replace([np.inf,-np.inf],np.nan).fillna(0)
model = IsolationForest(n_estimators=150, contamination=0.06, random_state=42)
df["ai_anomaly"] = (model.fit_predict(X) == -1).astype(int)
df["anomaly_score"] = -model.score_samples(X)
df["risk_score"] = (
    0.55*np.clip((df["temperature_c"]-UPPER).abs()/6,0,1)
    + 0.30*np.clip(df["rolling_std_60m"]/2,0,1)
    + 0.15*df["ai_anomaly"]
)*100
df["risk_level"] = pd.cut(df["risk_score"], bins=[-1,25,55,101], labels=["Low","Medium","High"])

# Sidebar
st.sidebar.header("Filters")
shipment = st.sidebar.selectbox("Shipment", ["All"] + sorted(df.shipment_id.unique().tolist()))
sensor = st.sidebar.selectbox("Sensor position", ["All"] + sorted(df.sensor_position.unique().tolist()))

f = df.copy()
if shipment != "All": f = f[f.shipment_id == shipment]
if sensor != "All": f = f[f.sensor_position == sensor]

# KPIs
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Shipments", f.shipment_id.nunique())
c2.metric("Observations", f"{len(f):,}")
c3.metric("Excursion Rate", f"{f.out_of_range.mean()*100:.1f}%")
c4.metric("AI Anomalies", f"{f.ai_anomaly.sum():,}")
c5.metric("High-Risk", f"{(f.risk_level=='High').sum():,}")

st.divider()

left,right = st.columns([2,1])
with left:
    st.subheader("Temperature Trend")
    plot = f.copy()
    if shipment == "All":
        plot = plot[plot.shipment_id == f.shipment_id.iloc[0]]
    fig = px.line(plot, x="timestamp", y="temperature_c", color="sensor_position",
                  title=f"Temperature profile — {plot.shipment_id.iloc[0]}")
    fig.add_hline(y=UPPER, line_dash="dash", annotation_text="Upper limit 8°C")
    fig.add_hline(y=LOWER, line_dash="dash", annotation_text="Lower limit 2°C")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Risk Distribution")
    rc = f["risk_level"].value_counts().reindex(["Low","Medium","High"]).fillna(0).reset_index()
    rc.columns = ["risk_level","count"]
    fig2 = px.bar(rc, x="risk_level", y="count", text="count")
    fig2.update_layout(height=430)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Sensor Comparison")
summary = f.groupby("sensor_position").agg(
    Avg_Temperature=("temperature_c","mean"),
    Max_Temperature=("temperature_c","max"),
    Excursion_Rate=("out_of_range","mean"),
    Anomaly_Rate=("ai_anomaly","mean"),
    Avg_Risk=("risk_score","mean")
).reset_index()
summary["Excursion_Rate"] *= 100
summary["Anomaly_Rate"] *= 100
st.dataframe(summary.round(2), use_container_width=True)

st.subheader("Top Risk Events")
top = f.sort_values("risk_score", ascending=False)[
    ["shipment_id","timestamp","sensor_position","temperature_c","humidity_pct","anomaly_score","risk_score","risk_level"]
].head(15)
st.dataframe(top.round(2), use_container_width=True)

st.info(
    "Decision logic: threshold-based temperature excursion + behavioural anomaly detection + "
    "risk scoring. High-risk observations should be investigated; the prototype does not claim "
    "to determine product safety or spoilage."
)
