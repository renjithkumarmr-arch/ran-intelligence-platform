# RAN Intelligence Platform

A complete 5G RAN operations analytics project inspired by the structure and goals of the SD-WAN & WiFi Analytics dashboard. It turns cell telemetry into a NOC dashboard with multi-KPI health scoring, ML anomaly detection, 15-minute SLA-risk prediction, cell status, geospatial views and probable root-cause hints.

## Business questions answered
- Is the RAN healthy now?
- Which sites and sectors are degraded or critical?
- Where are congestion, RF, mobility or transport issues emerging?
- Which cells are likely to breach service thresholds in the next 15 minutes?
- What KPI is the probable root cause?

## Included
- Synthetic 5G NR telemetry generator for six sites and three sectors per site
- CSV ingestion contract for replacement with OSS/EMS/SMO data
- Configurable KPI thresholds and health-score weights
- Isolation Forest anomaly detection
- Random Forest next-window breach risk
- Dash/Plotly operations dashboard
- Site map, sector heatmap, KPI trends, live status and operations table
- Docker, Compose and pytest support

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open `http://localhost:8050`.

## Docker

```bash
docker compose up --build
```

## Data integration
Place `data/ran_telemetry.csv` with these columns:

`timestamp, site_id, site_name, cell_id, sector, technology, band, latitude, longitude, connected_users, availability_pct, dl_throughput_mbps, ul_throughput_mbps, latency_ms, packet_loss_pct, prb_utilization_pct, rrc_success_pct, handover_success_pct, sinr_db`

If the file is absent, demo data is generated automatically.

## Production integration pattern
1. Collect PM counters from vendor OSS/EMS, O-RAN SMO or a streaming bus.
2. Normalize vendor-specific counters to the data contract.
3. Store raw and aggregated measurements in a time-series store.
4. Schedule model retraining and version the model artifacts.
5. Expose alerts to ITSM, email, Teams or a NOC event bus.
6. Add authentication, RBAC, audit logs and secrets management before production.

## Configuration
Edit `config/config.yaml` to change service thresholds, weighting, refresh interval and prediction horizon.

## Tests

```bash
pytest -q
```

## Important note
This repository uses synthetic demo data and a demonstration ML workflow. Validate thresholds, labels and models against operator-specific SLA definitions and historical incidents before production use.
