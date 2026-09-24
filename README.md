# 🧊 ColdChain AI

## Intelligent Cold-Chain Temperature Excursion & Risk Analytics

ColdChain AI is a data analytics and machine-learning prototype for monitoring temperature-sensitive shipments. It combines threshold-based temperature excursion detection with unsupervised anomaly detection and an explainable risk score.

### Important data note
For the deadline-ready prototype, this repository uses a **simulated sensor dataset** generated specifically for demonstration. It is clearly labelled as simulated data and must not be presented as live operational data.

A public real-world strawberry cold-chain dataset that can be used for future validation is:
https://huggingface.co/datasets/NifferLi/Cold-Chain-Transportation-Strawberry

### Features
- Temperature excursion detection
- Multi-sensor comparison
- Rolling time-series features
- Isolation Forest anomaly detection
- Explainable risk score
- Shipment and sensor filtering
- Interactive Streamlit dashboard
- Operational insight framing: Fact → Insight → Risk → Action

### Technologies
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Jupyter

### Project structure
```text
ColdChain_AI/
├── data/
│   └── cold_chain_sensor_data.csv
├── notebooks/
│   └── ColdChain_AI_Analysis.ipynb
├── dashboard/
│   └── app.py
├── coldchain_pipeline.py
├── requirements.txt
└── README.md
```

### Setup
Create a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

### Run the analytics pipeline
```bash
python coldchain_pipeline.py
```

### Run the dashboard
```bash
streamlit run dashboard/app.py
```

### Methodology
1. Load and validate sensor data.
2. Apply a prototype operating band of 2°C–8°C.
3. Engineer temperature-change, rolling mean, rolling standard deviation and deviation features.
4. Detect behavioural anomalies with Isolation Forest.
5. Combine threshold and behavioural signals into a 0–100 risk score.
6. Visualize findings through an interactive dashboard.

### Limitations
This is an academic prototype. The simulated dataset is not a substitute for validated operational cold-chain data. Product safety, spoilage, regulatory compliance and shipment release decisions require validated domain procedures and qualified personnel.

### Resume-ready summary
Developed a cold-chain analytics prototype that combines time-series feature engineering, Isolation Forest anomaly detection and an interactive Streamlit dashboard to identify temperature excursions and prioritize shipment-level risk investigation.
