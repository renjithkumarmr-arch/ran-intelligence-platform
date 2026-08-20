from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import ROOT

KPI_COLUMNS = [
    "availability_pct",
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "prb_utilization_pct",
    "rrc_success_pct",
    "handover_success_pct",
    "sinr_db",
]

# Demo site inventory. Replace this with OSS/EMS/SMO/GIS inventory in production.
SITE_INVENTORY = [
    {
        "site_id": "CHN-001",
        "site_name": "Chennai Central",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "technology": "5G NR",
        "band": "n78",
        "site_type": "Macro",
    },
    {
        "site_id": "CHN-002",
        "site_name": "Guindy",
        "latitude": 13.0067,
        "longitude": 80.2206,
        "technology": "5G NR",
        "band": "n78",
        "site_type": "Macro",
    },
    {
        "site_id": "CHN-003",
        "site_name": "OMR",
        "latitude": 12.9010,
        "longitude": 80.2279,
        "technology": "5G NR",
        "band": "n78",
        "site_type": "Macro",
    },
    {
        "site_id": "BLR-001",
        "site_name": "Bengaluru East",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "technology": "5G NR",
        "band": "n78",
        "site_type": "Macro",
    },
    {
        "site_id": "BLR-002",
        "site_name": "Whitefield",
        "latitude": 12.9698,
        "longitude": 77.7500,
        "technology": "5G NR",
        "band": "n78",
        "site_type": "Macro",
    },
    {
        "site_id": "HYD-001",
        "site_name": "Hitech City",
        "latitude": 17.4435,
        "longitude": 78.3772,
        "technology": "5G NR",
        "band": "n78",
        "site_type": "Macro",
    },
]

# Three-sector macro template. Range is illustrative, not RF-predicted coverage.
SECTOR_TEMPLATE = {
    1: {"azimuth_deg": 0.0, "beamwidth_deg": 65.0, "coverage_radius_km": 1.6},
    2: {"azimuth_deg": 120.0, "beamwidth_deg": 65.0, "coverage_radius_km": 1.6},
    3: {"azimuth_deg": 240.0, "beamwidth_deg": 65.0, "coverage_radius_km": 1.6},
}


def build_cell_inventory() -> pd.DataFrame:
    """Create one inventory row per cell/sector."""
    rows: list[dict] = []
    for site in SITE_INVENTORY:
        for sector, radio in SECTOR_TEMPLATE.items():
            rows.append(
                {
                    **site,
                    "cell_id": f"{site['site_id']}-S{sector}",
                    "sector": sector,
                    "azimuth_deg": radio["azimuth_deg"],
                    "beamwidth_deg": radio["beamwidth_deg"],
                    "coverage_radius_km": radio["coverage_radius_km"],
                    "antenna_height_m": 30.0,
                    "mechanical_tilt_deg": 2.0,
                    "electrical_tilt_deg": 4.0,
                    "pci": 100 + sector,
                    "arfcn": 640000,
                }
            )
    return pd.DataFrame(rows)


def generate_telemetry(
    hours: int = 24,
    interval_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic demonstration telemetry tied to the cell inventory."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="Asia/Kolkata").floor(f"{interval_minutes}min")
    periods = hours * 60 // interval_minutes
    times = pd.date_range(end=end, periods=periods, freq=f"{interval_minutes}min")
    inventory = build_cell_inventory()
    rows: list[dict] = []

    for cell_index, cell in inventory.iterrows():
        sector = int(cell["sector"])
        site_factor = cell_index // 3

        for time_index, timestamp in enumerate(times):
            hour = timestamp.hour + timestamp.minute / 60
            busy = max(0.0, np.sin((hour - 8) / 24 * 2 * np.pi))
            prb = np.clip(45 + 38 * busy + rng.normal(0, 7), 5, 100)
            sinr = np.clip(22 - 0.12 * prb + rng.normal(0, 2.2), -5, 35)

            incident = ((time_index + site_factor * 17 + sector * 9) % 173) in range(0, 8)
            if incident:
                prb = np.clip(prb + 15, 0, 100)
                sinr -= 8

            dl = np.clip(230 - 1.7 * prb + 4.5 * sinr + rng.normal(0, 15), 2, 500)
            ul = np.clip(65 - 0.42 * prb + 1.1 * sinr + rng.normal(0, 5), 1, 150)
            latency = np.clip(7 + 0.19 * prb - 0.18 * sinr + rng.normal(0, 2.5), 2, 90)
            loss = np.clip(0.08 + 0.012 * prb - 0.025 * sinr + rng.normal(0, 0.18), 0, 8)
            rrc = np.clip(99.8 - 0.025 * prb - 0.10 * loss + rng.normal(0, 0.25), 85, 100)
            handover = np.clip(99.2 - 0.018 * prb - 0.15 * loss + rng.normal(0, 0.35), 85, 100)
            availability = np.clip(99.99 - incident * rng.uniform(0.5, 2.5) + rng.normal(0, 0.02), 94, 100)
            users = int(np.clip(80 + 4.2 * prb + rng.normal(0, 35), 10, 700))

            rows.append(
                {
                    "timestamp": timestamp,
                    "site_id": cell["site_id"],
                    "site_name": cell["site_name"],
                    "cell_id": cell["cell_id"],
                    "sector": sector,
                    "technology": cell["technology"],
                    "band": cell["band"],
                    "site_type": cell["site_type"],
                    "latitude": cell["latitude"],
                    "longitude": cell["longitude"],
                    "azimuth_deg": cell["azimuth_deg"],
                    "beamwidth_deg": cell["beamwidth_deg"],
                    "coverage_radius_km": cell["coverage_radius_km"],
                    "antenna_height_m": cell["antenna_height_m"],
                    "mechanical_tilt_deg": cell["mechanical_tilt_deg"],
                    "electrical_tilt_deg": cell["electrical_tilt_deg"],
                    "pci": cell["pci"],
                    "arfcn": cell["arfcn"],
                    "connected_users": users,
                    "availability_pct": availability,
                    "dl_throughput_mbps": dl,
                    "ul_throughput_mbps": ul,
                    "latency_ms": latency,
                    "packet_loss_pct": loss,
                    "prb_utilization_pct": prb,
                    "rrc_success_pct": rrc,
                    "handover_success_pct": handover,
                    "sinr_db": sinr,
                }
            )

    return pd.DataFrame(rows)


def _add_inventory_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill architect-map fields into an older telemetry CSV."""
    inventory = build_cell_inventory()
    inventory_fields = [
        "cell_id",
        "site_name",
        "site_type",
        "latitude",
        "longitude",
        "azimuth_deg",
        "beamwidth_deg",
        "coverage_radius_km",
        "antenna_height_m",
        "mechanical_tilt_deg",
        "electrical_tilt_deg",
        "pci",
        "arfcn",
    ]

    fields_to_add = [
        column for column in inventory_fields if column != "cell_id" and column not in df.columns
    ]
    if not fields_to_add:
        return df

    return df.merge(
        inventory[["cell_id", *fields_to_add]],
        on="cell_id",
        how="left",
        validate="many_to_one",
    )


def load_or_generate() -> pd.DataFrame:
    path = ROOT / "data" / "ran_telemetry.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        return _add_inventory_fields(df)

    df = generate_telemetry()
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df
