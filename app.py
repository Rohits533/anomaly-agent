"""
FastAPI backend for anomaly-agent.
Model is trained once at startup. Detection is instant.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    global baseline
    samples = []
    for i in range(60):
        cpu = 10.0 + (i % 5) * 0.5
        mem = 40.0 + (i % 7) * 0.4
        samples.append(make_synthetic_snapshot(cpu=cpu, mem=mem))
    detector.fit(samples)
    keys = AnomalyDetector.FEATURES
    n = len(samples)
    baseline = {k: sum(s[k] for s in samples) / n for k in keys}
    print(f"[startup] Trained on {n} synthetic samples. Model ready.")


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


@app.post("/detect")
def detect(snapshot: Snapshot):
    s = snapshot.dict()
    is_anomaly, score = detector.predict(s)
    is_anomaly = bool(is_anomaly)
    score = float(score)
    alert = reporter.format_alert(s, score, baseline) if is_anomaly else None
    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "alert": alert,
    }


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
    alert = reporter.format_alert(s, score, baseline) if is_anomaly else None
    return {"is_anomaly": is_anomaly, "score": score, "alert": alert}
