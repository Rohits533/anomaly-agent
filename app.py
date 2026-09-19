"""
FastAPI backend for anomaly-agent.
Exposes the anomaly detection logic as an API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.detector import AnomalyDetector
from src.reporter import Reporter

app = FastAPI(title="anomaly-agent API", version="1.0")

# Allow the Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll tighten this after Vercel gives us a URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for a simple demo (resets when service restarts)
detector = AnomalyDetector(contamination=0.05)
reporter = Reporter(threshold=0.0)
baseline_samples: List[dict] = []
baseline: dict = {}


class Snapshot(BaseModel):
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    net_sent_mb: float
    net_recv_mb: float


class TrainingBatch(BaseModel):
    snapshots: List[Snapshot]


@app.get("/health")
def health():
    return {"status": "ok", "trained": detector.is_fitted}


@app.post("/train")
def train(batch: TrainingBatch):
    """Train the detector on a batch of normal snapshots."""
    global baseline_samples, baseline
    baseline_samples = [s.dict() for s in batch.snapshots]
    if len(baseline_samples) < 10:
        return {"error": "Need at least 10 snapshots to train"}
    detector.fit(baseline_samples)
    # Compute simple mean baseline for the reporter
    keys = AnomalyDetector.FEATURES
    n = len(baseline_samples)
    baseline = {k: sum(s[k] for s in baseline_samples) / n for k in keys}
    return {"status": "trained", "samples": n}


@app.post("/detect")
def detect(snapshot: Snapshot):
    """Check a single snapshot for anomalies."""
    if not detector.is_fitted:
        return {"error": "Model not trained yet. Call /train first."}
    s = snapshot.dict()
    is_anomaly, score = detector.predict(s)
    alert = reporter.format_alert(s, score, baseline) if is_anomaly else None
    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "alert": alert,
    }
