from app.analytics import enrich
from app.assistant import answer_query
from app.data import generate_telemetry


def sample_df():
    return enrich(generate_telemetry(hours=1))


def test_cell_lookup():
    response = answer_query(sample_df(), "Why is CHN-001-S2 degraded?")
    assert response["type"] == "cell"
    assert "CHN-001-S2" in response["title"]
    assert response["cause"]
    assert response["action"]


def test_site_lookup():
    response = answer_query(sample_df(), "Analyse CHN-001")
    assert response["type"] == "site"
    assert "CHN-001" in response["title"]
    assert any("Most affected cell" in fact for fact in response["facts"])


def test_operational_queries():
    df = sample_df()
    assert answer_query(df, "highest risk cell")["type"] == "list"
    assert answer_query(df, "show congested cells")["type"] == "list"
    assert answer_query(df, "show critical cells")["type"] == "list"
