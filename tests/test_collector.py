"""Tests for the metrics collector."""

from src.collector import collect_snapshot


def test_snapshot_has_expected_keys():
    snapshot = collect_snapshot()
    expected = {
        "timestamp",
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "net_sent_mb",
        "net_recv_mb",
    }
    assert expected.issubset(snapshot.keys())


def test_values_are_numeric():
    snapshot = collect_snapshot()
    assert isinstance(snapshot["cpu_percent"], (int, float))
    assert isinstance(snapshot["memory_percent"], (int, float))
