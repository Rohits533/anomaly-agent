"""
FastAPI backend for anomaly-agent.
Model is trained once at startup and persisted to disk.
"""
import os
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

from src.detector import AnomalyDetector
from src.reporter import Reporter
from src.collector import collect_snapshot

app = FastAPI(title="anomaly-agent API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.joblib"
ALERTS_PATH = "alerts.log"

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


def save_model():
    import joblib
    joblib.dump({"model": detector.model, "baseline": baseline}, MODEL_PATH)


def load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        saved = joblib.load(MODEL_PATH)
        detector.model = saved["model"]
        detector.is_fitted = True
        global baseline
        baseline = saved["baseline"]
        return True
    except Exception as e:
        print(f"[startup] Failed to load model: {e}")
        return False


def train_on_synthetic_baseline():
    global baseline
    if load_model():
        print("[startup] Loaded model from disk.")
        return
    samples = []
    for i in range(60):
        cpu = 10.0 + (i % 5) * 0.5
        mem = 40.0 + (i % 7) * 0.4
        samples.append(make_synthetic_snapshot(cpu=cpu, mem=mem))
    detector.fit(samples)
    keys = AnomalyDetector.FEATURES
    n = len(samples)
    baseline = {k: sum(s[k] for s in samples) / n for k in keys}
    save_model()
    print(f"[startup] Trained on {n} synthetic samples and saved to disk.")


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
    return {"status": "ok", "trained": bool(detector.is_fitted)}


@app.get("/collect")
def collect():
    """Return a real snapshot of the server's current state."""
    return collect_snapshot()


@app.post("/calibrate")
def calibrate(count: int = 30):
    """Collect `count` real snapshots and retrain on them."""
    global baseline
    samples = [collect_snapshot() for _ in range(count)]
    detector.fit(samples)
    keys = AnomalyDetector.FEATURES
    n = len(samples)
    baseline = {k: sum(s[k] for s in samples) / n for k in keys}
    save_model()
    return {"status": "calibrated", "samples": n, "baseline": baseline}


def log_alert(alert_text: str):
    with open(ALERTS_PATH, "a") as f:
        f.write(alert_text + "\n\n")


@app.get("/alerts")
def get_alerts(limit: int = 20):
    if not os.path.exists(ALERTS_PATH):
        return {"alerts": []}
    with open(ALERTS_PATH) as f:
        content = f.read().strip()
    blocks = [b for b in content.split("\n\n") if b]
    return {"alerts": blocks[-limit:]}


@app.post("/detect")
def detect(snapshot: Snapshot):
    s = snapshot.dict()
    is_anomaly, score = detector.predict(s)
    is_anomaly = bool(is_anomaly)
    score = float(score)
    alert = None
    if is_anomaly:
        alert = reporter.format_alert(s, score, baseline)
        if alert:
            log_alert(alert)
    return {"is_anomaly": is_anomaly, "score": score, "alert": alert}


@app.get("/demo/{kind}")
def demo(kind: str):
    if kind == "normal":
        s = make_synthetic_snapshot(cpu=10.5, mem=40.5)
    elif kind == "anomaly":
        s = make_synthetic_snapshot(cpu=95.0, mem=90.0)
    else:
        return {"error": "kind must be 'normal' or 'anomaly'"}
    is_anomaly, score = detector.predict(s)
    is_anomaly = bool(is_anomaly)
    score = float(score)
    alert = None
    if is_anomaly:
        alert = reporter.format_alert(s, score, baseline)
        if alert:
            log_alert(alert)
    return {"is_anomaly": is_anomaly, "score": score, "alert": alert}


@app.get("/stream")
async def stream():
    """Server-Sent Events: live snapshot + score every 5 seconds."""
    async def event_generator():
        while True:
            try:
                s = collect_snapshot()
                is_anomaly, score = detector.predict(s)
                payload = {
                    "snapshot": s,
                    "is_anomaly": bool(is_anomaly),
                    "score": float(score),
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
