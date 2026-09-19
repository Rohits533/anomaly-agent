"""Tests for the agent baseline computation."""

from src.agent import compute_baseline


def make_snapshot(cpu, mem):
    return {
        "timestamp": "2026-01-01T00:00:00",
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_percent": 50.0,
        "net_sent_mb": 1.0,
        "net_recv_mb": 2.0,
    }


def test_baseline_is_mean_of_each_feature():
    samples = [
        make_snapshot(cpu=10, mem=40),
        make_snapshot(cpu=20, mem=60),
    ]
    baseline = compute_baseline(samples)
    assert baseline["cpu_percent"] == 15.0
    assert baseline["memory_percent"] == 50.0
    assert baseline["disk_percent"] == 50.0


def test_baseline_has_all_features():
    samples = [make_snapshot(cpu=10, mem=40) for _ in range(3)]
    baseline = compute_baseline(samples)
    expected = {
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "net_sent_mb",
        "net_recv_mb",
    }
    assert set(baseline.keys()) == expected
