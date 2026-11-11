# 🚦 FlowCast: AI-Powered Smart Traffic Flow Prediction Platform

Predicts **traffic volume, congestion level, travel time and accident risk** per road segment
for the 25-segment Northline Corridor, using sensor, weather and calendar data, and serves
the results through an interactive Streamlit dashboard.

> **Status:** Step 1 of 10 complete (repo setup). See [docs/ROADMAP.md](docs/ROADMAP.md).

## Targets (PRD section 3.3)
| Task | Metric | Target |
|------|--------|--------|
| Volume forecast | MAPE | ≤ 12% |
| Congestion (4-class) | Macro-F1 | ≥ 0.80 |
| Accident risk | ROC-AUC | ≥ 0.75 |
| Deep vs classical | Volume RMSE | LSTM/GRU beats best classical |

## Quick start
```bash
git clone https://github.com/<your-username>/flowcast.git
cd flowcast
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_data.py     # verify raw data loads
python -m pytest -q              # run tests
```

## Project layout
```
data/raw/        source tables as delivered (read-only)
data/interim/    cleaned intermediates (generated)
data/processed/  analysis-ready dataset (generated)
src/             ingest -> clean -> features -> ml_models / dl_model -> evaluate
notebooks/       EDA, cleaning, modelling narratives
models/          persisted models + model cards
reports/         data-quality report + figures
dashboard/       Streamlit app
tests/           pytest suite
docs/            PRD, data dictionary, roadmap
```

## Data
Three raw tables for 1 Jan - 31 May 2025: `traffic_sensor_log.csv` (~178k rows),
`weather_observations.csv` (hourly, 3 stations), `calendar_events.csv` (daily).
See [docs/FlowCast_Data_Dictionary.docx](docs/FlowCast_Data_Dictionary.docx).

## License
MIT (add a LICENSE file if you want to publish).
