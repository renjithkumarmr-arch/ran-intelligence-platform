from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG
from .data import KPI_COLUMNS


def _metric_score(
    series: pd.Series,
    warning: float,
    critical: float,
    direction: str,
) -> np.ndarray:
    """Convert a KPI to a 35-100 health component score."""
    values = series.astype(float)

    if direction == "high":
        span = max(warning - critical, 1e-6)
        return np.where(
            values >= warning,
            100,
            np.where(values <= critical, 35, 35 + 65 * (values - critical) / span),
        )

    span = max(critical - warning, 1e-6)
    return np.where(
        values <= warning,
        100,
        np.where(values >= critical, 35, 100 - 65 * (values - warning) / span),
    )


def _safe_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    """Read a numeric field safely from a telemetry row."""
    value = row.get(column, default)
    return default if pd.isna(value) else float(value)


def _rca_result(
    cause: str,
    confidence: int,
    evidence: list[str],
    recommendation: str,
    severity: str,
) -> dict:
    """Return one consistently structured RCA candidate."""
    return {
        "root_cause": cause,
        "rca_confidence_pct": int(np.clip(confidence, 0, 100)),
        "rca_evidence": "; ".join(evidence),
        "recommended_action": recommendation,
        "rca_severity": severity,
    }


def level2_root_cause(row: pd.Series) -> dict:
    """
    Transparent Level 2 RCA based on multi-KPI correlation.

    This gives a probable cause, not a confirmed cause. A production RCA
    should also correlate alarms, configuration changes, traces, topology,
    hardware status, transport counters and historical incidents.
    """
    availability = _safe_float(row, "availability_pct", 100.0)
    dl = _safe_float(row, "dl_throughput_mbps")
    ul = _safe_float(row, "ul_throughput_mbps")
    latency = _safe_float(row, "latency_ms")
    loss = _safe_float(row, "packet_loss_pct")
    prb = _safe_float(row, "prb_utilization_pct")
    rrc = _safe_float(row, "rrc_success_pct", 100.0)
    handover = _safe_float(row, "handover_success_pct", 100.0)
    sinr = _safe_float(row, "sinr_db")
    users = _safe_float(row, "connected_users")
    health = _safe_float(row, "health_score", 100.0)

    candidates: list[dict] = []

    # 1. Site/cell availability degradation
    if availability < 99.0:
        confidence = 70
        evidence = [f"Availability {availability:.2f}%"]
        if dl < 10:
            confidence += 10
            evidence.append(f"DL throughput {dl:.1f} Mbps")
        if users < 30:
            confidence += 10
            evidence.append(f"Only {users:.0f} connected users")
        if availability < 97.0:
            confidence += 5
            evidence.append("Availability is in the critical range")

        candidates.append(
            _rca_result(
                "Site or cell availability degradation",
                confidence,
                evidence,
                "Check node and cell alarms, power status, transport reachability, "
                "restart history and administrative state.",
                "Critical" if availability < 97.0 else "High",
            )
        )

    # 2. Radio capacity congestion
    if prb >= 85 and dl < 100:
        confidence = 60
        evidence = [f"PRB utilization {prb:.1f}%", f"DL throughput {dl:.1f} Mbps"]
        if users >= 300:
            confidence += 15
            evidence.append(f"{users:.0f} connected users")
        if prb >= 92:
            confidence += 10
            evidence.append("PRB utilization is critically high")
        if dl < 60:
            confidence += 10
            evidence.append("DL throughput is severely degraded")
        if sinr >= 12:
            confidence += 5
            evidence.append(f"SINR {sinr:.1f} dB indicates acceptable RF quality")

        candidates.append(
            _rca_result(
                "Radio capacity congestion",
                confidence,
                evidence,
                "Review busy-hour demand, user distribution, load balancing, "
                "carrier capacity, spectrum availability and sector expansion.",
                "Critical" if prb >= 92 else "High",
            )
        )

    # 3. RF quality, interference or coverage degradation
    if sinr < 12:
        confidence = 55
        evidence = [f"SINR {sinr:.1f} dB"]
        if dl < 80:
            confidence += 10
            evidence.append(f"DL throughput {dl:.1f} Mbps")
        if handover < 97:
            confidence += 10
            evidence.append(f"Handover success {handover:.2f}%")
        if loss >= 1:
            confidence += 5
            evidence.append(f"Packet loss {loss:.2f}%")
        if prb < 75:
            confidence += 10
            evidence.append(f"PRB utilization {prb:.1f}% is not congested")
        if sinr < 7:
            confidence += 10
            evidence.append("SINR is in the critical RF range")

        candidates.append(
            _rca_result(
                "RF quality, interference or coverage degradation",
                confidence,
                evidence,
                "Review coverage and interference plots, antenna tilt, overshooting "
                "cells, PCI conflicts, neighbour relations and sector hardware alarms.",
                "Critical" if sinr < 7 else "High",
            )
        )

    # 4. Mobility degradation
    if handover < 97:
        confidence = 55
        evidence = [f"Handover success {handover:.2f}%"]
        if sinr < 12:
            confidence += 10
            evidence.append(f"SINR {sinr:.1f} dB")
        if handover < 94:
            confidence += 15
            evidence.append("Handover success is critically low")
        if rrc >= 98:
            confidence += 5
            evidence.append(f"RRC success {rrc:.2f}% remains healthy")
        if prb >= 85:
            confidence += 5
            evidence.append(f"PRB utilization {prb:.1f}% may affect mobility")

        candidates.append(
            _rca_result(
                "Mobility or handover degradation",
                confidence,
                evidence,
                "Audit neighbour definitions, Xn/X2 connectivity, handover thresholds, "
                "time-to-trigger, hysteresis, PCI conflicts and coverage overlap.",
                "Critical" if handover < 94 else "High",
            )
        )

    # 5. Radio access or RRC establishment failure
    if rrc < 98:
        confidence = 55
        evidence = [f"RRC success {rrc:.2f}%"]
        if rrc < 95:
            confidence += 15
            evidence.append("RRC success is critically low")
        if prb >= 85:
            confidence += 10
            evidence.append(f"PRB utilization {prb:.1f}% may cause admission rejection")
        if sinr < 12:
            confidence += 10
            evidence.append(f"SINR {sinr:.1f} dB may affect access establishment")
        if availability >= 99.5:
            confidence += 5
            evidence.append(f"Availability {availability:.2f}% confirms the cell is online")

        candidates.append(
            _rca_result(
                "Radio access or RRC establishment failure",
                confidence,
                evidence,
                "Check RRC failure counters, admission control, random access, "
                "signalling capacity, RF conditions and AMF/core connectivity.",
                "Critical" if rrc < 95 else "High",
            )
        )

    # 6. Combined transport/backhaul degradation
    if latency > 20 and loss >= 1:
        confidence = 65
        evidence = [f"Latency {latency:.1f} ms", f"Packet loss {loss:.2f}%"]
        if sinr >= 12:
            confidence += 10
            evidence.append(f"SINR {sinr:.1f} dB indicates acceptable RF quality")
        if prb < 75:
            confidence += 10
            evidence.append(f"PRB utilization {prb:.1f}% indicates no radio congestion")
        if latency > 35:
            confidence += 5
            evidence.append("Latency is in the critical range")
        if loss >= 2:
            confidence += 5
            evidence.append("Packet loss is in the critical range")

        candidates.append(
            _rca_result(
                "Transport or backhaul degradation",
                confidence,
                evidence,
                "Check interface utilization, errors, queue drops, QoS policy, routing "
                "changes, microwave quality and end-to-end transport delay.",
                "Critical" if latency > 35 or loss >= 2 else "High",
            )
        )

    # 7. Latency-only transport degradation
    if latency > 20 and loss < 1:
        confidence = 50
        evidence = [f"Latency {latency:.1f} ms", f"Packet loss remains {loss:.2f}%"]
        if sinr >= 12:
            confidence += 10
            evidence.append(f"SINR {sinr:.1f} dB is acceptable")
        if prb < 75:
            confidence += 10
            evidence.append(f"PRB utilization {prb:.1f}% is normal")
        if latency > 35:
            confidence += 10
            evidence.append("Latency is in the critical range")

        candidates.append(
            _rca_result(
                "Transport latency degradation",
                confidence,
                evidence,
                "Check transport path delay, congestion, QoS queues, routing changes, "
                "synchronization and processing latency.",
                "Critical" if latency > 35 else "Medium",
            )
        )

    # 8. Packet-loss-only transport degradation
    if loss >= 1 and latency <= 20:
        confidence = 50
        evidence = [f"Packet loss {loss:.2f}%", f"Latency remains {latency:.1f} ms"]
        if sinr >= 12:
            confidence += 10
            evidence.append(f"SINR {sinr:.1f} dB is acceptable")
        if prb < 75:
            confidence += 10
            evidence.append(f"PRB utilization {prb:.1f}% is normal")
        if loss >= 2:
            confidence += 10
            evidence.append("Packet loss is in the critical range")

        candidates.append(
            _rca_result(
                "Transport packet-loss degradation",
                confidence,
                evidence,
                "Check interface errors, queue drops, MTU mismatch, routing stability, "
                "microwave link quality and discard counters.",
                "Critical" if loss >= 2 else "Medium",
            )
        )

    # 9. Downlink degradation without clear congestion or RF cause
    if dl < 80 and prb < 85 and sinr >= 12:
        confidence = 45
        evidence = [
            f"DL throughput {dl:.1f} Mbps",
            f"PRB utilization {prb:.1f}%",
            f"SINR {sinr:.1f} dB",
        ]
        if latency > 20:
            confidence += 10
            evidence.append(f"Latency {latency:.1f} ms")
        if loss >= 1:
            confidence += 10
            evidence.append(f"Packet loss {loss:.2f}%")

        candidates.append(
            _rca_result(
                "Downlink performance degradation",
                confidence,
                evidence,
                "Check scheduler behaviour, bearer QoS, transport capacity, MIMO layers, "
                "modulation distribution and UE capability mix.",
                "Medium",
            )
        )

    # 10. Uplink degradation
    if ul < 25 and availability >= 99:
        confidence = 45
        evidence = [f"UL throughput {ul:.1f} Mbps"]
        if sinr < 12:
            confidence += 10
            evidence.append(f"SINR {sinr:.1f} dB")
        if prb >= 85:
            confidence += 10
            evidence.append(f"PRB utilization {prb:.1f}%")

        candidates.append(
            _rca_result(
                "Uplink performance degradation",
                confidence,
                evidence,
                "Review uplink interference, power control, PUSCH quality, scheduler "
                "allocation, UE power limitation and transport capacity.",
                "Medium",
            )
        )

    if not candidates:
        if health >= 80:
            return _rca_result(
                "No dominant issue detected",
                90,
                [f"Health score {health:.1f}", "KPIs are within configured thresholds"],
                "Continue monitoring. No immediate corrective action is required.",
                "Normal",
            )

        return _rca_result(
            "Degradation requires additional correlation",
            35,
            [f"Health score {health:.1f}", "No Level 2 rule produced a strong match"],
            "Correlate OSS alarms, historical trends, configuration changes, neighbour "
            "data and transport counters.",
            "Medium",
        )

    candidates.sort(key=lambda item: item["rca_confidence_pct"], reverse=True)
    primary = candidates[0].copy()

    if len(candidates) > 1:
        secondary = candidates[1]
        if secondary["rca_confidence_pct"] >= 60:
            primary["rca_evidence"] += (
                f"; Contributing factor: {secondary['root_cause']} "
                f"({secondary['rca_confidence_pct']}%)"
            )

    return primary


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add health, anomaly, SLA risk and Level 2 RCA fields."""
    out = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True).copy()

    total_score = np.zeros(len(out))
    for metric, weight in CONFIG["weights"].items():
        threshold = CONFIG["thresholds"][metric]
        total_score += weight * _metric_score(
            out[metric],
            threshold["warning"],
            threshold["critical"],
            threshold["direction"],
        )

    out["health_score"] = np.clip(total_score, 0, 100).round(1)
    out["status"] = np.select(
        [out["health_score"] < 60, out["health_score"] < 80],
        ["Critical", "Degraded"],
        default="Healthy",
    )

    out["anomaly"] = False
    for _, indexes in out.groupby("cell_id").groups.items():
        group_data = out.loc[indexes, KPI_COLUMNS]
        anomaly_model = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    IsolationForest(
                        contamination=0.035,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        out.loc[indexes, "anomaly"] = anomaly_model.fit_predict(group_data) == -1

    out["breach_now"] = (out["health_score"] < 60).astype(int)
    out["future_breach"] = (
        out.groupby("cell_id")["breach_now"].shift(-3).fillna(0).astype(int)
    )

    model_features = KPI_COLUMNS + ["connected_users", "health_score"]
    split_index = max(int(len(out) * 0.8), 1)
    training_data = out.iloc[:split_index]

    if training_data["future_breach"].nunique() > 1:
        risk_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        risk_model.fit(training_data[model_features], training_data["future_breach"])
        out["breach_risk_pct"] = (
            risk_model.predict_proba(out[model_features])[:, 1] * 100
        ).round(1)
    else:
        out["breach_risk_pct"] = np.clip(
            (70 - out["health_score"]) * 2, 0, 100
        ).round(1)

    rca_results = out.apply(level2_root_cause, axis=1, result_type="expand")
    for column in rca_results.columns:
        out[column] = rca_results[column]

    return out
