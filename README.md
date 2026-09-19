markdown
# anomaly-agent

A lightweight, zero-trust anomaly detection engine for distributed systems, with a live web dashboard.

**Live Demo:** https://anomaly-agent-abc.vercel.app *(replace with your actual Vercel URL)*

## What it does

Learns the normal behavior of a system (CPU, memory, disk, network) and flags deviations before they cause an outage. Uses an Isolation Forest model for unsupervised detection and produces human-readable alerts with root-cause hints.

## Architecture
[Vercel: Frontend] ──HTTP/SSE──> [Render: FastAPI] ──> [Isolation Forest]
HTML/CSS/JS Python 3.12 scikit-learn

text

- **Frontend (Vercel):** Dashboard with live metrics, calibration, detection, and alert history.
- **Backend (Render):** FastAPI service. Trains at startup, persists the model to disk, exposes REST + SSE.
- **Detection:** Isolation Forest trained on synthetic baseline, retrainable on real metrics.
- **Reporting:** Per-feature deviation analysis with plain-English alerts.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service status and model state |
| `/collect` | GET | Real snapshot of server metrics |
| `/calibrate?count=30` | POST | Collect real snapshots and retrain |
| `/detect` | POST | Score a single snapshot |
| `/demo/{kind}` | GET | Demo: `kind` = `normal` or `anomaly` |
| `/alerts?limit=20` | GET | Recent anomaly alerts |
| `/stream` | GET | Server-Sent Events: live metrics |
| `/docs` | GET | Auto-generated interactive API docs |

## Features

- Unsupervised anomaly detection (no labeled data required)
- Root-cause hints: which metric deviated and by how much
- Model persistence across restarts (joblib)
- Real metrics via psutil (`/collect`, `/calibrate`)
- Alert log with history endpoint
- Live streaming dashboard via Server-Sent Events
- Synthetic demo mode for deterministic testing

## Project structure
.
├── app.py # FastAPI backend
├── requirements.txt
├── frontend/
│ └── index.html # Dashboard
├── src/
│ ├── agent.py # CLI loop
│ ├── collector.py # psutil-based metrics
│ ├── detector.py # Isolation Forest wrapper
│ └── reporter.py # Alert formatting
└── tests/ # Pytest suite

text

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
CLI mode

bash
python -m src.agent --warmup 30 --interval 5
