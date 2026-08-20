from __future__ import annotations

import re
from typing import Any

import pandas as pd

SITE_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d{3}\b", re.IGNORECASE)
CELL_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d{3}-S\d+\b", re.IGNORECASE)


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest observation for every cell."""
    if df.empty:
        return df.copy()
    ordered = df.sort_values("timestamp")
    return ordered.groupby("cell_id", as_index=False).tail(1).copy()


def _number(row: pd.Series, field: str, digits: int = 1) -> str:
    value = row.get(field)
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _cell_response(row: pd.Series) -> dict[str, Any]:
    return {
        "type": "cell",
        "title": f"Cell Analysis: {row['cell_id']}",
        "severity": str(row.get("rca_severity", row.get("status", "Normal"))),
        "summary": (
            f"{row['cell_id']} is {str(row['status']).lower()} with a health score of "
            f"{_number(row, 'health_score')}/100 and a 15-minute SLA risk of "
            f"{_number(row, 'breach_risk_pct')}%."
        ),
        "facts": [
            f"Site: {row['site_id']} | Sector: {int(row['sector'])}",
            f"PRB utilization: {_number(row, 'prb_utilization_pct')}%",
            f"SINR: {_number(row, 'sinr_db')} dB",
            f"DL throughput: {_number(row, 'dl_throughput_mbps')} Mbps",
            f"Latency: {_number(row, 'latency_ms')} ms",
            f"Packet loss: {_number(row, 'packet_loss_pct', 2)}%",
            f"RRC success: {_number(row, 'rrc_success_pct', 2)}%",
            f"Handover success: {_number(row, 'handover_success_pct', 2)}%",
            f"Connected users: {int(row['connected_users'])}",
        ],
        "cause": str(row.get("root_cause", "No dominant issue detected")),
        "confidence": f"{_number(row, 'rca_confidence_pct', 0)}%",
        "evidence": str(row.get("rca_evidence", "No supporting evidence available")),
        "action": str(row.get("recommended_action", "Continue monitoring.")),
        "timestamp": str(row["timestamp"]),
    }


def _site_response(site_rows: pd.DataFrame, site_id: str) -> dict[str, Any]:
    site_rows = site_rows.sort_values(
        ["health_score", "breach_risk_pct"], ascending=[True, False]
    )
    worst = site_rows.iloc[0]
    status_counts = site_rows["status"].value_counts()
    causes = site_rows["root_cause"].value_counts()
    primary_cause = causes.index[0] if not causes.empty else "No dominant issue detected"

    return {
        "type": "site",
        "title": f"Site Analysis: {site_id}",
        "severity": str(worst.get("rca_severity", worst.get("status", "Normal"))),
        "summary": (
            f"{site_id} has {len(site_rows)} cells with an average health score of "
            f"{site_rows['health_score'].mean():.1f}/100. The maximum 15-minute SLA "
            f"risk is {site_rows['breach_risk_pct'].max():.1f}%."
        ),
        "facts": [
            f"Healthy: {int(status_counts.get('Healthy', 0))}",
            f"Degraded: {int(status_counts.get('Degraded', 0))}",
            f"Critical: {int(status_counts.get('Critical', 0))}",
            f"Connected users: {int(site_rows['connected_users'].sum())}",
            f"Average PRB: {site_rows['prb_utilization_pct'].mean():.1f}%",
            f"Average SINR: {site_rows['sinr_db'].mean():.1f} dB",
            f"Most affected cell: {worst['cell_id']}",
        ],
        "cause": str(primary_cause),
        "confidence": f"{float(worst.get('rca_confidence_pct', 0)):.0f}%",
        "evidence": (
            f"Worst cell {worst['cell_id']}: {worst.get('rca_evidence', 'No evidence available')}"
        ),
        "action": str(worst.get("recommended_action", "Continue monitoring.")),
        "timestamp": str(worst["timestamp"]),
    }


def _list_response(title: str, rows: pd.DataFrame, reason: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "type": "list",
            "title": title,
            "severity": "Normal",
            "summary": "No matching cells were found in the latest network snapshot.",
            "facts": [],
            "cause": "No active issue",
            "confidence": "N/A",
            "evidence": reason,
            "action": "Continue monitoring.",
            "timestamp": "Latest snapshot",
        }

    facts = []
    for _, row in rows.head(10).iterrows():
        facts.append(
            f"{row['cell_id']}: health {_number(row, 'health_score')}, "
            f"risk {_number(row, 'breach_risk_pct')}%, PRB "
            f"{_number(row, 'prb_utilization_pct')}%, cause {row['root_cause']}"
        )

    worst = rows.iloc[0]
    return {
        "type": "list",
        "title": title,
        "severity": str(worst.get("rca_severity", worst.get("status", "Normal"))),
        "summary": f"Found {len(rows)} matching cells. Showing up to 10.",
        "facts": facts,
        "cause": str(worst.get("root_cause", "Multiple conditions")),
        "confidence": f"{_number(worst, 'rca_confidence_pct', 0)}%",
        "evidence": reason,
        "action": str(worst.get("recommended_action", "Review the listed cells.")),
        "timestamp": str(worst["timestamp"]),
    }


def answer_query(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Answer supported local RAN operational questions without an external LLM."""
    text = (query or "").strip()
    latest = _latest(df)

    if not text:
        return {
            "type": "help",
            "title": "Enter a site, cell or supported question",
            "severity": "Normal",
            "summary": "Try CHN-001-S2, analyse CHN-001, highest risk cell, show congested cells, or show critical cells.",
            "facts": [],
            "cause": "Not applicable",
            "confidence": "N/A",
            "evidence": "The assistant uses the latest enriched dashboard data.",
            "action": "Enter a supported question.",
            "timestamp": "Latest snapshot",
        }

    upper = text.upper()
    cell_match = CELL_PATTERN.search(upper)
    if cell_match:
        cell_id = cell_match.group(0).upper()
        row = latest[latest["cell_id"].str.upper() == cell_id]
        if not row.empty:
            return _cell_response(row.iloc[0])
        return _not_found(cell_id, latest)

    site_match = SITE_PATTERN.search(upper)
    if site_match:
        site_id = site_match.group(0).upper()
        rows = latest[latest["site_id"].str.upper() == site_id]
        if not rows.empty:
            return _site_response(rows, site_id)
        return _not_found(site_id, latest)

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    if any(phrase in normalized for phrase in ["highest risk", "top risk", "most risk"]):
        rows = latest.sort_values("breach_risk_pct", ascending=False).head(10)
        return _list_response(
            "Top Cells by 15-Minute SLA Risk",
            rows,
            "Cells are ranked by the latest predicted SLA-breach risk.",
        )

    if any(word in normalized for word in ["congested", "congestion", "high prb"]):
        rows = latest[
            (latest["prb_utilization_pct"] >= 85)
            | latest["root_cause"].str.contains("congestion", case=False, na=False)
        ].sort_values(["prb_utilization_pct", "breach_risk_pct"], ascending=False)
        return _list_response(
            "Congested Cells",
            rows,
            "Matched PRB utilization of at least 85% or a congestion RCA classification.",
        )

    if any(word in normalized for word in ["critical", "worst", "degraded"]):
        if "degraded" in normalized:
            rows = latest[latest["status"] == "Degraded"].sort_values("health_score")
            title = "Degraded Cells"
        else:
            rows = latest[latest["status"] == "Critical"].sort_values("health_score")
            title = "Critical Cells"
        return _list_response(title, rows, "Matched the latest health-status classification.")

    if any(word in normalized for word in ["transport", "backhaul", "latency", "packet loss"]):
        rows = latest[
            latest["root_cause"].str.contains("transport|backhaul", case=False, na=False)
            | (latest["latency_ms"] > 20)
            | (latest["packet_loss_pct"] >= 1)
        ].sort_values(["breach_risk_pct", "latency_ms"], ascending=False)
        return _list_response(
            "Cells with Possible Transport Degradation",
            rows,
            "Matched transport RCA, latency above 20 ms, or packet loss of at least 1%.",
        )

    return {
        "type": "help",
        "title": "Query not recognised",
        "severity": "Medium",
        "summary": "This local assistant supports deterministic RAN operations queries. GenAI can be added later for unrestricted language.",
        "facts": [
            "Enter a Cell ID, for example CHN-001-S2",
            "Enter a Site ID, for example CHN-001",
            "Ask: highest risk cell",
            "Ask: show congested cells",
            "Ask: show critical cells",
            "Ask: show transport issues",
        ],
        "cause": "Unsupported query format",
        "confidence": "N/A",
        "evidence": "No supported site, cell, risk, congestion, status or transport intent was found.",
        "action": "Use one of the suggested questions.",
        "timestamp": "Latest snapshot",
    }


def _not_found(identifier: str, latest: pd.DataFrame) -> dict[str, Any]:
    sites = ", ".join(sorted(latest["site_id"].unique())[:8])
    return {
        "type": "not_found",
        "title": f"No data found for {identifier}",
        "severity": "Medium",
        "summary": "The identifier is not present in the latest telemetry snapshot.",
        "facts": [f"Available sites include: {sites}"],
        "cause": "Inventory lookup did not match",
        "confidence": "N/A",
        "evidence": "The assistant searched the latest enriched dataframe.",
        "action": "Check the site/cell ID and try again.",
        "timestamp": "Latest snapshot",
    }
