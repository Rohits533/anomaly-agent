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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AnomalyDetector(contamination=0.05)
reporter = Reporter(threshold=0.0)
baseline: dict = {}


def make_synthetic_snapshot(cpu=10.0, mem=40.0):
    return {
        "timestamp": "2026-01-01T00:00:00",
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_percent": 50.0,
        "net_sent_mb": 1.0,
        "net_recv_mb": 2.0,
    }


def train_on_synthetic_baseline():
    """Train the detector immediately at startup with synthetic data.
    This means no user ever waits for training."""
    global baseline
    samples = []
    # Create 60 normal samples with slight natural variation
    for i in range(60):
        cpu = 10.0 + (i % 5) * 0.5
        mem = 40.0 + (i % 7) * 0.4
        samples.append(make_synthetic_snapshot(cpu=cpu, mem=mem))

    detector.fit(samples)

    keys = AnomalyDetector.FEATURES
    n = len(samples)
    baseline = {k: sum(s[k] for s in samples) / n for k in keys}
    print(f"[startup] Trained on {n} synthetic samples. Model ready.")


# Train once when the service starts
train_on_synthetic_baseline()


class Snapshot(BaseModel):
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    net_sent_mb: float
    net_recv_mb: float


@app.get("/health")
def health():
    return {"status": "ok", "trained": detector.is_fitted}


@app.post("/train")
def train():
    """Re-train on the synthetic baseline. Instant, no user data needed."""
    train_on_synthetic_baseline()
    return {"status": "trained", "samples": 60}


@app.post("/detect")
def detect(snapshot: Snapshot):
    if not detector.is_fitted:
        train_on_synthetic_baseline()
    s = snapshot.dict()
    is_anomaly, score = detector.predict(s)
    alert = reporter.format_alert(s, score, baseline) if is_anomaly else None
    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "alert": alert,
    }
