"""
Anomaly agent.
Ties collector, detector, and reporter into a running loop.

Usage:
    python -m src.agent --warmup 30 --interval 5
"""

import argparse
import time

from src.collector import collect_snapshot
from src.detector import AnomalyDetector
from src.reporter import Reporter


def compute_baseline(snapshots):
    """Return the mean of each feature across the warmup window."""
    keys = AnomalyDetector.FEATURES
    n = len(snapshots)
    return {k: sum(s[k] for s in snapshots) / n for k in keys}


def run(warmup, interval, contamination):
    detector = AnomalyDetector(contamination=contamination)
    reporter = Reporter(threshold=0.0)

    print(f"[agent] Collecting {warmup} baseline samples...")
    baseline_samples = []
    for i in range(warmup):
        baseline_samples.append(collect_snapshot())
        print(f"  sample {i + 1}/{warmup}", end="\r")
        time.sleep(interval)
    print()

    print("[agent] Training detector on baseline...")
    detector.fit(baseline_samples)
    baseline = compute_baseline(baseline_samples)
    print("[agent] Ready. Watching for anomalies.\n")

    try:
        while True:
            snapshot = collect_snapshot()
            is_anomaly, score = detector.predict(snapshot)

            if is_anomaly:
                alert = reporter.format_alert(snapshot, score, baseline)
                if alert:
                    print(alert)
                    print()
            else:
                print(reporter.format_normal(score))

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[agent] Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-trust anomaly detection agent."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of samples to collect before training (default: 30).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Seconds between samples (default: 5).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomaly proportion (default: 0.05).",
    )
    args = parser.parse_args()
    run(args.warmup, args.interval, args.contamination)


if __name__ == "__main__":
    main()
