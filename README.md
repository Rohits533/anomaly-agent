# anomaly-agent

A lightweight, zero-trust anomaly detection engine for distributed systems.

## What it does

Learns the normal behavior of a system (CPU, memory, disk, network) and
alerts on deviations before they cause an outage. Uses an Isolation Forest
model for unsupervised detection and produces alerts with root-cause hints.

## How it works

1. **Collector** — reads system metrics every few seconds.
2. **Detector** — trains an Isolation Forest on a warmup window, then
   scores each new sample.
3. **Reporter** — formats anomalies as alerts with the top deviating features.

## Usage

```bash
pip install -r requirements.txt
python -m src.agent --warmup 30 --interval 5
