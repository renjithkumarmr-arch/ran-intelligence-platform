from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from .analytics import enrich
from .config import CONFIG
from .data import load_or_generate

RAW = load_or_generate()
DF = enrich(RAW)

STATUS_COLORS = {
    "Healthy": "#22c55e",
    "Degraded": "#f59e0b",
    "Critical": "#ef4444",
}

SEVERITY_COLORS = {
    "Normal": "#22c55e",
    "Medium": "#eab308",
    "High": "#f97316",
    "Critical": "#ef4444",
}

app = Dash(
    __name__,
    title=CONFIG["app"]["title"],
    suppress_callback_exceptions=True,
)
server = app.server


def kpi_card(title: str, value_id: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.P(title, className="card-title"),
            html.H2(id=value_id),
            html.Small(subtitle),
        ],
        className="kpi-card",
    )


sites = sorted(DF["site_id"].unique())

app.layout = html.Div(
    [
        dcc.Interval(
            id="refresh",
            interval=CONFIG["app"]["refresh_seconds"] * 1000,
            n_intervals=0,
        ),
        html.Header(
            [
                html.Div(
                    [
                        html.H1("AI-Powered 5G RAN Service Assurance Platform"),
                        html.P(
                            "Multi-KPI health, anomaly detection, SLA risk and Level 2 RCA"
                        ),
                    ]
                ),
                html.Div("LIVE DEMO DATA", className="live"),
            ]
        ),
        html.Div(
            [
                html.Label("Site"),
                dcc.Dropdown(
                    [{"label": "All Sites", "value": "ALL"}]
                    + [{"label": site, "value": site} for site in sites],
                    "ALL",
                    id="site-filter",
                    clearable=False,
                ),
                html.Label("Time window"),
                dcc.Dropdown(
                    [
                        {"label": "6 hours", "value": 6},
                        {"label": "12 hours", "value": 12},
                        {"label": "24 hours", "value": 24},
                    ],
                    24,
                    id="hours-filter",
                    clearable=False,
                ),
                html.Label("KPI"),
                dcc.Dropdown(
                    {
                        "health_score": "Health Score",
                        "dl_throughput_mbps": "DL Throughput",
                        "latency_ms": "Latency",
                        "prb_utilization_pct": "PRB Utilization",
                        "sinr_db": "SINR",
                        "handover_success_pct": "Handover Success",
                        "rrc_success_pct": "RRC Success",
                    },
                    "health_score",
                    id="kpi-filter",
                    clearable=False,
                ),
            ],
            className="filters",
        ),
        html.Main(
            [
                html.Div(
                    [
                        kpi_card("Network Health", "health", "Weighted multi-KPI score"),
                        kpi_card("Critical Cells", "critical", "Latest interval"),
                        kpi_card("Anomalies", "anomalies", "Selected window"),
                        kpi_card("SLA Risk > 60%", "risk", "Next 15 minutes"),
                        kpi_card("Active Users", "users", "Latest interval"),
                    ],
                    className="card-grid",
                ),
                html.Div(
                    [
                        html.Div([dcc.Graph(id="trend")], className="panel wide"),
                        html.Div([dcc.Graph(id="status-donut")], className="panel"),
                    ],
                    className="grid-2",
                ),
                html.Div(
                    [
                        html.Div([dcc.Graph(id="site-map")], className="panel"),
                        html.Div([dcc.Graph(id="heatmap")], className="panel"),
                    ],
                    className="grid-2 equal",
                ),
                html.Div(
                    [
                        html.Div([dcc.Graph(id="rca-chart")], className="panel"),
                        html.Div([dcc.Graph(id="risk-chart")], className="panel"),
                    ],
                    className="grid-2 equal",
                ),
                html.Div(
                    [
                        html.H3("Cell Operations and Level 2 Probable Root Cause"),
                        html.P(
                            "RCA is based on transparent multi-KPI correlation. "
                            "Use OSS alarms and detailed counters to confirm the cause.",
                            className="table-note",
                        ),
                        dash_table.DataTable(
                            id="cell-table",
                            page_size=12,
                            sort_action="native",
                            filter_action="native",
                            fixed_rows={"headers": True},
                            tooltip_duration=None,
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "680px",
                                "overflowY": "auto",
                            },
                            style_cell={
                                "backgroundColor": "#0f172a",
                                "color": "#dbeafe",
                                "border": "1px solid #334155",
                                "padding": "9px",
                                "fontFamily": "Inter, Segoe UI, sans-serif",
                                "fontSize": "13px",
                                "textAlign": "left",
                                "minWidth": "120px",
                                "width": "150px",
                                "maxWidth": "320px",
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_cell_conditional=[
                                {
                                    "if": {"column_id": "root_cause"},
                                    "minWidth": "220px",
                                    "width": "250px",
                                    "maxWidth": "300px",
                                },
                                {
                                    "if": {"column_id": "rca_evidence"},
                                    "minWidth": "340px",
                                    "width": "400px",
                                    "maxWidth": "500px",
                                },
                                {
                                    "if": {"column_id": "recommended_action"},
                                    "minWidth": "400px",
                                    "width": "450px",
                                    "maxWidth": "550px",
                                },
                            ],
                            style_header={
                                "backgroundColor": "#1e293b",
                                "color": "#f8fafc",
                                "fontWeight": "700",
                                "border": "1px solid #475569",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": "{status} = Critical"},
                                    "backgroundColor": "#4c0519",
                                },
                                {
                                    "if": {"filter_query": "{status} = Degraded"},
                                    "backgroundColor": "#451a03",
                                },
                                {
                                    "if": {
                                        "filter_query": "{rca_severity} = Critical",
                                        "column_id": "rca_severity",
                                    },
                                    "backgroundColor": "#991b1b",
                                    "color": "white",
                                    "fontWeight": "700",
                                },
                                {
                                    "if": {
                                        "filter_query": "{rca_severity} = High",
                                        "column_id": "rca_severity",
                                    },
                                    "backgroundColor": "#9a3412",
                                    "color": "white",
                                    "fontWeight": "700",
                                },
                            ],
                        ),
                    ],
                    className="panel operations-panel",
                ),
            ]
        ),
    ],
    className="page",
)


@app.callback(
    [
        Output("health", "children"),
        Output("critical", "children"),
        Output("anomalies", "children"),
        Output("risk", "children"),
        Output("users", "children"),
        Output("trend", "figure"),
        Output("status-donut", "figure"),
        Output("site-map", "figure"),
        Output("heatmap", "figure"),
        Output("rca-chart", "figure"),
        Output("risk-chart", "figure"),
        Output("cell-table", "data"),
        Output("cell-table", "columns"),
        Output("cell-table", "tooltip_data"),
    ],
    [
        Input("site-filter", "value"),
        Input("hours-filter", "value"),
        Input("kpi-filter", "value"),
        Input("refresh", "n_intervals"),
    ],
)
def update_dashboard(site: str, hours: int, kpi: str, _: int):
    cutoff = DF["timestamp"].max() - pd.Timedelta(hours=int(hours))
    selected = DF[DF["timestamp"] >= cutoff].copy()

    if site != "ALL":
        selected = selected[selected["site_id"] == site].copy()

    latest_timestamp = selected["timestamp"].max()
    latest = selected[selected["timestamp"] == latest_timestamp].copy()

    health_value = f"{latest['health_score'].mean():.1f}/100"
    critical_value = str(int((latest["status"] == "Critical").sum()))
    anomaly_value = str(int(selected["anomaly"].sum()))
    risk_value = str(int((latest["breach_risk_pct"] > 60).sum()))
    user_value = f"{int(latest['connected_users'].sum()):,}"

    labels = {
        "health_score": "Health Score",
        "dl_throughput_mbps": "DL Throughput (Mbps)",
        "latency_ms": "Latency (ms)",
        "prb_utilization_pct": "PRB Utilization (%)",
        "sinr_db": "SINR (dB)",
        "handover_success_pct": "Handover Success (%)",
        "rrc_success_pct": "RRC Success (%)",
    }

    trend_data = selected.groupby("timestamp", as_index=False)[kpi].mean()
    trend = px.line(
        trend_data,
        x="timestamp",
        y=kpi,
        title=f"Network Trend: {labels[kpi]}",
        template="plotly_dark",
    )

    anomaly_rows = selected[selected["anomaly"]]
    trend.add_trace(
        go.Scatter(
            x=anomaly_rows["timestamp"],
            y=anomaly_rows[kpi],
            mode="markers",
            name="ML anomaly",
            marker={"symbol": "x", "size": 9, "color": "#ef4444"},
        )
    )

    status_counts = latest["status"].value_counts().reset_index()
    status_counts.columns = ["status", "cells"]
    status_donut = px.pie(
        status_counts,
        names="status",
        values="cells",
        hole=0.65,
        title="Current Cell Status",
        color="status",
        color_discrete_map=STATUS_COLORS,
        template="plotly_dark",
    )

    site_summary = latest.groupby(
        ["site_id", "site_name", "latitude", "longitude"],
        as_index=False,
    ).agg(
        health_score=("health_score", "mean"),
        connected_users=("connected_users", "sum"),
        breach_risk_pct=("breach_risk_pct", "max"),
    )

    site_map = px.scatter_map(
        site_summary,
        lat="latitude",
        lon="longitude",
        color="health_score",
        size="connected_users",
        hover_name="site_name",
        hover_data=["site_id", "breach_risk_pct"],
        zoom=4.1,
        height=420,
        color_continuous_scale="RdYlGn",
        range_color=[40, 100],
        map_style="carto-darkmatter",
        title="Site Health and Load",
    )

    health_pivot = latest.pivot_table(
        index="site_id",
        columns="sector",
        values="health_score",
        aggfunc="mean",
    )
    heatmap = px.imshow(
        health_pivot,
        aspect="auto",
        text_auto=".0f",
        zmin=40,
        zmax=100,
        color_continuous_scale="RdYlGn",
        title="Sector Health Heatmap",
        labels={"x": "Sector", "y": "Site", "color": "Health"},
        template="plotly_dark",
    )

    rca_counts = latest["root_cause"].value_counts().head(8).sort_values()
    rca_chart = px.bar(
        x=rca_counts.values,
        y=rca_counts.index,
        orientation="h",
        title="Current Probable Root Causes",
        labels={"x": "Cells", "y": "Probable root cause"},
        template="plotly_dark",
        color=rca_counts.values,
        color_continuous_scale="OrRd",
    )
    rca_chart.update_coloraxes(showscale=False)

    top_risk = latest.nlargest(10, "breach_risk_pct").sort_values("breach_risk_pct")
    risk_chart = px.bar(
        top_risk,
        x="breach_risk_pct",
        y="cell_id",
        orientation="h",
        color="rca_severity",
        color_discrete_map=SEVERITY_COLORS,
        title="Top Cells by 15-Minute SLA Risk",
        labels={"breach_risk_pct": "Risk (%)", "cell_id": "Cell"},
        template="plotly_dark",
        hover_data=["root_cause", "health_score"],
    )
    risk_chart.update_xaxes(range=[0, 100])

    table_columns = [
        "site_id",
        "cell_id",
        "status",
        "health_score",
        "breach_risk_pct",
        "rca_severity",
        "root_cause",
        "rca_confidence_pct",
        "rca_evidence",
        "recommended_action",
        "connected_users",
        "prb_utilization_pct",
        "sinr_db",
        "rrc_success_pct",
        "handover_success_pct",
        "dl_throughput_mbps",
        "ul_throughput_mbps",
        "latency_ms",
        "packet_loss_pct",
        "availability_pct",
    ]

    table_frame = (
        latest[table_columns]
        .round(2)
        .sort_values(
            ["health_score", "breach_risk_pct"],
            ascending=[True, False],
        )
    )

    table_data = table_frame.to_dict("records")
    columns = [
        {"name": column.replace("_", " ").title(), "id": column}
        for column in table_columns
    ]
    tooltip_data = [
        {
            column: {"value": str(value), "type": "markdown"}
            for column, value in row.items()
            if column in {"root_cause", "rca_evidence", "recommended_action"}
        }
        for row in table_data
    ]

    for figure in (
        trend,
        status_donut,
        site_map,
        heatmap,
        rca_chart,
        risk_chart,
    ):
        figure.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#dbeafe",
            margin={"l": 35, "r": 20, "t": 55, "b": 35},
        )

    return (
        health_value,
        critical_value,
        anomaly_value,
        risk_value,
        user_value,
        trend,
        status_donut,
        site_map,
        heatmap,
        rca_chart,
        risk_chart,
        table_data,
        columns,
        tooltip_data,
    )
