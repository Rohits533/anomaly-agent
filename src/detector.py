"""
Anomaly detector.
Learns normal system behavior and flags deviations using Isolation Forest.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Unsupervised anomaly detector for system metrics."""

    FEATURES = [
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "net_sent_mb",
        "net_recv_mb",
    ]

    def __init__(self, contamination=0.05):
        """
        contamination: expected proportion of outliers in the data.
        Higher = more sensitive. Lower = more conservative.
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
        )
        self.is_fitted = False

    def _to_matrix(self, snapshots):
        """Convert list of snapshot dicts into a 2D numpy array."""
        return np.array(
            [[s[f] for f in self.FEATURES] for s in snapshots]
        )

    def fit(self, snapshots):
        """Train the model on a batch of normal snapshots."""
        if len(snapshots) < 10:
            raise ValueError("Need at least 10 snapshots to train.")
        X = self._to_matrix(snapshots)
        self.model.fit(X)
        self.is_fitted = True

    def predict(self, snapshot):
        """
        Return (is_anomaly, score) for a single snapshot.
        score: negative = more anomalous, positive = more normal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X = self._to_matrix([snapshot])
        label = self.model.predict(X)[0]        # 1 = normal, -1 = anomaly
        score = self.model.decision_function(X)[0]
        return (label == -1, float(score))
