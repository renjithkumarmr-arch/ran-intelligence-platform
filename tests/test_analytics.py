from app.analytics import enrich, level2_root_cause
from app.data import generate_telemetry


def test_enrichment_columns_and_ranges():
    result = enrich(generate_telemetry(hours=1))
    expected = {
        "health_score",
        "status",
        "anomaly",
        "breach_risk_pct",
        "root_cause",
        "rca_confidence_pct",
        "rca_evidence",
        "recommended_action",
        "rca_severity",
    }
    assert expected.issubset(result.columns)
    assert result["health_score"].between(0, 100).all()
    assert result["breach_risk_pct"].between(0, 100).all()
    assert result["rca_confidence_pct"].between(0, 100).all()


def test_capacity_congestion_rule():
    row = {
        "availability_pct": 99.99,
        "dl_throughput_mbps": 45,
        "ul_throughput_mbps": 30,
        "latency_ms": 12,
        "packet_loss_pct": 0.2,
        "prb_utilization_pct": 95,
        "rrc_success_pct": 99,
        "handover_success_pct": 98,
        "sinr_db": 18,
        "connected_users": 450,
        "health_score": 58,
    }
    result = level2_root_cause(row)
    assert result["root_cause"] == "Radio capacity congestion"
    assert result["rca_confidence_pct"] >= 80


def test_transport_rule():
    row = {
        "availability_pct": 99.99,
        "dl_throughput_mbps": 90,
        "ul_throughput_mbps": 35,
        "latency_ms": 42,
        "packet_loss_pct": 2.2,
        "prb_utilization_pct": 55,
        "rrc_success_pct": 99,
        "handover_success_pct": 98,
        "sinr_db": 18,
        "connected_users": 150,
        "health_score": 55,
    }
    result = level2_root_cause(row)
    assert result["root_cause"] == "Transport or backhaul degradation"
    assert result["rca_confidence_pct"] >= 80
