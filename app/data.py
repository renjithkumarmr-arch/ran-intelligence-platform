from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT

KPI_COLUMNS = [
    "availability_pct", "dl_throughput_mbps", "ul_throughput_mbps",
    "latency_ms", "packet_loss_pct", "prb_utilization_pct",
    "rrc_success_pct", "handover_success_pct", "sinr_db"
]

def generate_telemetry(hours: int = 24, interval_minutes: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="Asia/Kolkata").floor("5min")
    times = pd.date_range(end=end, periods=hours * 60 // interval_minutes, freq=f"{interval_minutes}min")
    sites = [
        ("CHN-001", "Chennai Central", 13.0827, 80.2707),
        ("CHN-002", "Guindy", 13.0067, 80.2206),
        ("CHN-003", "OMR", 12.9010, 80.2279),
        ("BLR-001", "Bengaluru East", 12.9716, 77.5946),
        ("BLR-002", "Whitefield", 12.9698, 77.7500),
        ("HYD-001", "Hitech City", 17.4435, 78.3772),
    ]
    rows = []
    for sidx, (site_id, site_name, lat, lon) in enumerate(sites):
        for sector in (1, 2, 3):
            phase = sidx * .5 + sector
            for i, ts in enumerate(times):
                hour = ts.hour + ts.minute / 60
                busy = max(0, np.sin((hour - 8) / 24 * 2 * np.pi))
                prb = np.clip(45 + 38 * busy + rng.normal(0, 7), 5, 100)
                sinr = np.clip(22 - .12 * prb + rng.normal(0, 2.2), -5, 35)
                incident = ((i + sidx * 17 + sector * 9) % 173) in range(0, 8)
                if incident:
                    prb = np.clip(prb + 15, 0, 100); sinr -= 8
                dl = np.clip(230 - 1.7 * prb + 4.5 * sinr + rng.normal(0, 15), 2, 500)
                ul = np.clip(65 - .42 * prb + 1.1 * sinr + rng.normal(0, 5), 1, 150)
                latency = np.clip(7 + .19 * prb - .18 * sinr + rng.normal(0, 2.5), 2, 90)
                loss = np.clip(.08 + .012 * prb - .025 * sinr + rng.normal(0, .18), 0, 8)
                rrc = np.clip(99.8 - .025 * prb - .10 * loss + rng.normal(0, .25), 85, 100)
                ho = np.clip(99.2 - .018 * prb - .15 * loss + rng.normal(0, .35), 85, 100)
                avail = np.clip(99.99 - incident * rng.uniform(.5, 2.5) + rng.normal(0, .02), 94, 100)
                users = int(np.clip(80 + 4.2 * prb + rng.normal(0, 35), 10, 700))
                rows.append({
                    "timestamp": ts, "site_id": site_id, "site_name": site_name,
                    "cell_id": f"{site_id}-S{sector}", "sector": sector,
                    "technology": "5G NR", "band": "n78", "latitude": lat + sector*.002,
                    "longitude": lon + sector*.002, "connected_users": users,
                    "availability_pct": avail, "dl_throughput_mbps": dl,
                    "ul_throughput_mbps": ul, "latency_ms": latency,
                    "packet_loss_pct": loss, "prb_utilization_pct": prb,
                    "rrc_success_pct": rrc, "handover_success_pct": ho, "sinr_db": sinr,
                })
    return pd.DataFrame(rows)

def load_or_generate() -> pd.DataFrame:
    path = ROOT / "data" / "ran_telemetry.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        return df
    df = generate_telemetry()
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df
