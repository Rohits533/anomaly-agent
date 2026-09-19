"""Tests for the alert reporter."""

from src.reporter import Reporter


def make_snapshot(cpu=10.0, mem=40.0, disk=50.0, sent=1.0, recv=2.0):
    return {
        "timestamp": "2026-01-01T00:00:00",
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_percent": disk,
        "net_sent_mb": sent,
        "net_recv_mb": recv,
    }


def test_normal_score_returns_no_alert():
    reporter = Reporter(threshold=0.0)
    alert = reporter.format_alert(make_snapshot(), score=0.5, baseline=make_snapshot())
    assert alert is None


def test_anomalous_score_returns_alert():
    reporter = Reporter(threshold=0.0)
    snapshot = make_snapshot(cpu=99, mem=95)
    baseline = make_snapshot(cpu=10, mem=40)
    alert = reporter.format_alert(snapshot, score=-0.3, baseline=baseline)
    assert alert is not None
    assert "ALERT" in alert
    assert "CPU usage" in alert or "memory usage" in alert


def test_normal_status_line():
    reporter = Reporter()
    line = reporter.format_normal(score=0.42)
    assert line.startswith("[OK]")
