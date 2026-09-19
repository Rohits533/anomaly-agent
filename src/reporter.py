"""
Alert reporter.
Formats anomaly detections into human-readable alerts with root-cause hints.
"""

from datetime import datetime


FEATURE_LABELS = {
    "cpu_percent": "CPU usage",
    "memory_percent": "memory usage",
    "disk_percent": "disk usage",
    "net_sent_mb": "network sent",
    "net_recv_mb": "network received",
}


class Reporter:
    """Formats anomaly alerts with root-cause hints."""

    def __init__(self, threshold=0.0):
        """
        threshold: score below this is considered anomalous.
        Lower = more severe.
        """
        self.threshold = threshold

    def _find_deviating_features(self, snapshot, baseline):
        """
        Compare a snapshot to a baseline and return the features
        that deviate the most, sorted by deviation magnitude.
        """
        deviations = []
        for feature in FEATURE_LABELS:
            if feature not in baseline or baseline[feature] == 0:
                continue
            deviation = abs(snapshot[feature] - baseline[feature]) / abs(baseline[feature])
            deviations.append((feature, deviation))
        deviations.sort(key=lambda x: x[1], reverse=True)
        return deviations[:2]

    def format_alert(self, snapshot, score, baseline):
        """
        Build a human-readable alert string.
        Returns None if the score is not anomalous.
        """
        if score >= self.threshold:
            return None

        top_features = self._find_deviating_features(snapshot, baseline)

        lines = [
            f"[ALERT] {datetime.utcnow().isoformat()}Z",
            f"  Score: {score:.4f} (threshold: {self.threshold:.4f})",
            "  Likely cause:",
        ]

        for feature, deviation in top_features:
            label = FEATURE_LABELS[feature]
            value = snapshot[feature]
            base = baseline[feature]
            percent = deviation * 100
            lines.append(
                f"    - {label}: {value:.2f} (baseline {base:.2f}, {percent:+.1f}%)"
            )

        return "\n".join(lines)

    def format_normal(self, score):
        """Return a short status line for normal readings."""
        return f"[OK] score={score:.4f}"
