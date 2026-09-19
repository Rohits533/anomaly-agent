"""
System metrics collector.
Reads CPU, memory, disk, and network stats from the local machine.
"""

import psutil
import time
from datetime import datetime


def collect_snapshot():
    """Return a single snapshot of system metrics as a dict."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "net_sent_mb": psutil.net_io_counters().bytes_sent / (1024 * 1024),
        "net_recv_mb": psutil.net_io_counters().bytes_recv / (1024 * 1024),
    }


if __name__ == "__main__":
    while True:
        print(collect_snapshot())
        time.sleep(5)
