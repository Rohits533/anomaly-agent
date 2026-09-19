"""Tests for the anomaly detector."""

import pytest
from src.detector import AnomalyDetector


def make_snapshot(cpu=10.0, mem=40.0, disk=50.0, sent=1.0, recv=2.0):
    return {
        "timestamp": "2026-01-01T00:00:00",
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_percent": disk,
        "net_sent_mb": sent,
        "net_recv_mb": recv,
    }


def test_fit_requires_enough_data():
    detector = AnomalyDetector()
    with pytest.raises(ValueError):
        detector.fit([make_snapshot()] * 5)


def test_predict_before_fit_raises():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.predict(make_snapshot())


def test_normal_snapshot_is_not_anomaly():
    detector = AnomalyDetector()
    normal = [make_snapshot(cpu=10, mem=40) for _ in range(50)]
    detector.fit(normal)
    is_anomaly, score = detector.predict(make_snapshot(cpu=11, mem=41))
    assert is_anomaly is False


def test_extreme_snapshot_is_anomaly():
    detector = AnomalyDetector()
    normal = [make_snapshot(cpu=10, mem=40) for _ in range(50)]
    detector.fit(normal)
    is_anomaly, score = detector.predict(make_snapshot(cpu=99, mem=99))
    assert is_anomaly is True
