from app.data import generate_telemetry
from app.analytics import enrich

def test_enrichment():
    df=enrich(generate_telemetry(hours=1))
    assert {"health_score","status","anomaly","breach_risk_pct","root_cause"}.issubset(df.columns)
    assert df.health_score.between(0,100).all()
    assert df.breach_risk_pct.between(0,100).all()
