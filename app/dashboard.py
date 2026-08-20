from __future__ import annotations

import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from .analytics import enrich
from .assistant import answer_query
from .config import CONFIG
from .data import load_or_generate

RAW = load_or_generate()
DF = enrich(RAW)

STATUS_COLORS = {"Healthy": "#22c55e", "Degraded": "#f59e0b", "Critical": "#ef4444"}
SEVERITY_COLORS = {"Normal": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"}

app = Dash(
    __name__,
    title=CONFIG["app"]["title"],
    suppress_callback_exceptions=True,
    assets_folder=str(__import__("pathlib").Path(__file__).resolve().parent / "assets"),
)
server = app.server


def destination_point(latitude: float, longitude: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Calculate a destination point on Earth from bearing and distance."""
    earth_radius_km = 6371.0088
    angular_distance = distance_km / earth_radius_km
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def sector_polygon(
    latitude: float,
    longitude: float,
    azimuth_deg: float,
    beamwidth_deg: float,
    radius_km: float,
    points: int = 20,
) -> tuple[list[float], list[float]]:
    """Return a fan-shaped sector polygon."""
    start = azimuth_deg - beamwidth_deg / 2
    bearings = [start + beamwidth_deg * i / points for i in range(points + 1)]
    coordinates = [
        destination_point(latitude, longitude, bearing, radius_km)
        for bearing in bearings
    ]
    latitudes = [latitude, *[item[0] for item in coordinates], latitude]
    longitudes = [longitude, *[item[1] for item in coordinates], longitude]
    return latitudes, longitudes


def rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def build_architect_map(latest: pd.DataFrame) -> go.Figure:
    """Build an architect-level cell-sector map with directional coverage fans."""
    figure = go.Figure()

    # Coverage fans, one trace per cell.
    for _, row in latest.sort_values(["site_id", "sector"]).iterrows():
        color = STATUS_COLORS.get(row["status"], "#94a3b8")
        latitudes, longitudes = sector_polygon(
            float(row["latitude"]),
            float(row["longitude"]),
            float(row["azimuth_deg"]),
            float(row["beamwidth_deg"]),
            float(row["coverage_radius_km"]),
        )
        hover = (
            f"<b>{row['cell_id']}</b><br>"
            f"Site: {row['site_name']}<br>"
            f"Status: {row['status']}<br>"
            f"Health: {row['health_score']:.1f}/100<br>"
            f"SLA risk: {row['breach_risk_pct']:.1f}%<br>"
            f"RCA: {row['root_cause']}<br>"
            f"Confidence: {row['rca_confidence_pct']:.0f}%<br>"
            f"Azimuth: {row['azimuth_deg']:.0f} degrees<br>"
            f"Beamwidth: {row['beamwidth_deg']:.0f} degrees<br>"
            f"Range: {row['coverage_radius_km']:.1f} km<br>"
            f"Band: {row['band']} | PCI: {row['pci']}<br>"
            f"PRB: {row['prb_utilization_pct']:.1f}% | SINR: {row['sinr_db']:.1f} dB<br>"
            f"Users: {row['connected_users']:.0f}<extra></extra>"
        )
        figure.add_trace(
            go.Scattermap(
                lat=latitudes,
                lon=longitudes,
                mode="lines",
                fill="toself",
                fillcolor=rgba(color, 0.34),
                line={"color": color, "width": 1.6},
                name=row["cell_id"],
                legendgroup=row["status"],
                showlegend=False,
                hovertemplate=hover,
            )
        )

    # Site markers on top of the sector fans.
    site_summary = latest.groupby(
        ["site_id", "site_name", "latitude", "longitude"], as_index=False
    ).agg(
        health_score=("health_score", "mean"),
        connected_users=("connected_users", "sum"),
        breach_risk_pct=("breach_risk_pct", "max"),
    )
    site_summary["status"] = pd.cut(
        site_summary["health_score"],
        bins=[-1, 60, 80, 101],
        labels=["Critical", "Degraded", "Healthy"],
        right=False,
    ).astype(str)

    for status in ["Healthy", "Degraded", "Critical"]:
        group = site_summary[site_summary["status"] == status]
        if group.empty:
            continue
        figure.add_trace(
            go.Scattermap(
                lat=group["latitude"],
                lon=group["longitude"],
                mode="markers+text",
                text=group["site_id"],
                textposition="top center",
                marker={
                    "size": 14,
                    "color": STATUS_COLORS[status],
                    "opacity": 0.95,
                },
                customdata=group[["site_name", "health_score", "connected_users", "breach_risk_pct"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Site: %{customdata[0]}<br>"
                    "Health: %{customdata[1]:.1f}/100<br>"
                    "Users: %{customdata[2]:.0f}<br>"
                    "Maximum SLA risk: %{customdata[3]:.1f}%<extra></extra>"
                ),
                name=status,
            )
        )

    center = {
        "lat": float(latest["latitude"].mean()),
        "lon": float(latest["longitude"].mean()),
    }
    figure.update_layout(
        title="Directional Cell Coverage and Live Service Health",
        map={"style": "carto-darkmatter", "center": center, "zoom": 4.2},
        height=650,
        margin={"l": 8, "r": 8, "t": 55, "b": 8},
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#dbeafe",
        legend={"orientation": "h", "y": 1.01, "x": 0.01},
        uirevision="architect-map",
    )
    return figure


def kpi_card(title: str, value_id: str, subtitle: str) -> html.Div:
    return html.Div(
        [html.P(title, className="card-title"), html.H2(id=value_id), html.Small(subtitle)],
        className="kpi-card",
    )


sites = sorted(DF["site_id"].unique())


def render_assistant_response(result: dict) -> html.Div:
    severity = str(result.get("severity", "Normal"))
    facts = result.get("facts", [])
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(result.get("title", "RAN Analysis"), className="chat-result-title"),
                            html.Div(result.get("summary", ""), className="chat-result-summary"),
                        ]
                    ),
                    html.Span(severity, className=f"chat-badge severity-{severity.lower()}"),
                ],
                className="chat-result-head",
            ),
            html.Div(
                [
                    html.Div([html.Span("Probable cause"), html.Strong(result.get("cause", "N/A"))], className="chat-metric cause"),
                    html.Div([html.Span("Confidence"), html.Strong(result.get("confidence", "N/A"))], className="chat-metric confidence"),
                ],
                className="chat-metrics",
            ),
            html.Div(
                [html.Div([html.Span("Observed KPIs", className="chat-section-label"), html.Ul([html.Li(fact) for fact in facts])])]
                if facts else [],
                className="chat-observations",
            ),
            html.Div(
                [html.Span("Why this was identified", className="chat-section-label"), html.P(result.get("evidence", "N/A"))],
                className="chat-evidence",
            ),
            html.Div(
                [html.Span("Recommended next action", className="chat-section-label"), html.P(result.get("action", "N/A"))],
                className="chat-action",
            ),
            html.Div(f"Latest data: {result.get('timestamp', 'Latest snapshot')}", className="chat-time"),
        ],
        className="chat-result",
    )


def assistant_panel() -> html.Div:
    suggestions = ["CHN-001-S2", "Analyse CHN-001", "Highest risk cell", "Show congested cells", "Show critical cells"]
    return html.Div(
        [
            html.Button(
                [html.Span("AI", className="launcher-icon"), html.Span("RAN Assistant", className="launcher-text"), html.Span("●", className="launcher-live")],
                id="assistant-launcher",
                n_clicks=0,
                className="assistant-launcher",
                title="Open RAN Operations Assistant",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.Div("RA", className="assistant-avatar"), html.Div([html.H3("RAN Operations Assistant"), html.P("Live network analysis")])], className="assistant-identity"),
                            html.Button("×", id="assistant-close", n_clicks=0, className="assistant-close", title="Close"),
                        ],
                        className="assistant-window-header",
                    ),
                    html.Div(
                        [
                            html.Div([html.Strong("Hello, Renjith"), html.P("Enter a Site ID or Cell ID. I will explain the current issue, evidence and recommended action.")], className="assistant-intro"),
                            html.Div([html.Button(text, id={"type": "assistant-suggestion", "index": i}, n_clicks=0, className="chat-chip") for i, text in enumerate(suggestions)], className="chat-chips"),
                            dcc.Loading(
                                html.Div(id="assistant-answer", children=html.Div([html.Div("Try a quick analysis", className="empty-title"), html.Div("CHN-001-S2 or Analyse CHN-001", className="empty-subtitle")], className="assistant-empty")),
                                type="circle",
                                color="#38bdf8",
                            ),
                        ],
                        className="assistant-scroll",
                    ),
                    html.Div(
                        [
                            dcc.Input(id="assistant-input", type="text", placeholder="Ask about a site or cell...", debounce=False, className="assistant-input"),
                            html.Button("➤", id="assistant-send", n_clicks=0, className="assistant-send", title="Analyse"),
                        ],
                        className="assistant-composer",
                    ),
                    html.Div([html.Span("●", className="privacy-dot"), html.Span("Local analysis. No data leaves this application."), html.Button("Clear", id="assistant-clear", n_clicks=0, className="assistant-clear")], className="assistant-footer"),
                    dcc.Store(id="assistant-suggestions-store", data=suggestions),
                ],
                id="assistant-window",
                className="assistant-window assistant-hidden",
            ),
        ],
        className="assistant-floating-root",
    )


app.layout = html.Div(
    [
        dcc.Interval(id="refresh", interval=CONFIG["app"]["refresh_seconds"] * 1000, n_intervals=0),
        html.Header(
            [
                html.Div(
                    [
                        html.H1("AI-Powered 5G RAN Service Assurance Platform"),
                        html.P("Directional cell coverage, service health, SLA risk and Level 2 RCA"),
                    ]
                ),
                html.Div("ARCHITECT VIEW", className="live"),
            ]
        ),
        html.Div(
            [
                html.Label("Site"),
                dcc.Dropdown(
                    [{"label": "All Sites", "value": "ALL"}] + [{"label": site, "value": site} for site in sites],
                    "ALL",
                    id="site-filter",
                    clearable=False,
                ),
                html.Label("Time window"),
                dcc.Dropdown(
                    [{"label": "6 hours", "value": 6}, {"label": "12 hours", "value": 12}, {"label": "24 hours", "value": 24}],
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
                        html.Div(
                            [
                                dcc.Graph(id="architect-map"),
                                html.P(
                                    "Coverage fans use inventory azimuth, beamwidth and an illustrative radius. "
                                    "They are directional sector visualizations, not RF propagation predictions.",
                                    className="table-note",
                                ),
                            ],
                            className="panel architect-map-panel",
                        )
                    ],
                    className="single-grid",
                ),
                html.Div(
                    [html.Div([dcc.Graph(id="trend")], className="panel"), html.Div([dcc.Graph(id="status-donut")], className="panel")],
                    className="grid-2",
                ),
                html.Div(
                    [html.Div([dcc.Graph(id="heatmap")], className="panel"), html.Div([dcc.Graph(id="rca-chart")], className="panel")],
                    className="grid-2 equal",
                ),
                html.Div(
                    [
                        html.H3("Cell Inventory, RF Orientation and Level 2 RCA"),
                        dash_table.DataTable(
                            id="cell-table",
                            page_size=12,
                            sort_action="native",
                            filter_action="native",
                            fixed_rows={"headers": True},
                            style_table={"overflowX": "auto", "maxHeight": "680px", "overflowY": "auto"},
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
                                "maxWidth": "420px",
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_header={"backgroundColor": "#1e293b", "color": "#f8fafc", "fontWeight": "700"},
                            style_data_conditional=[
                                {"if": {"filter_query": "{status} = Critical"}, "backgroundColor": "#4c0519"},
                                {"if": {"filter_query": "{status} = Degraded"}, "backgroundColor": "#451a03"},
                            ],
                        ),
                    ],
                    className="panel operations-panel",
                ),
                assistant_panel(),
            ]
        ),
    ],
    className="page",
)


@app.callback(
    [
        Output("health", "children"), Output("critical", "children"), Output("anomalies", "children"),
        Output("risk", "children"), Output("users", "children"), Output("architect-map", "figure"),
        Output("trend", "figure"), Output("status-donut", "figure"), Output("heatmap", "figure"),
        Output("rca-chart", "figure"), Output("cell-table", "data"), Output("cell-table", "columns"),
    ],
    [Input("site-filter", "value"), Input("hours-filter", "value"), Input("kpi-filter", "value"), Input("refresh", "n_intervals")],
)
def update_dashboard(site: str, hours: int, kpi: str, _: int):
    cutoff = DF["timestamp"].max() - pd.Timedelta(hours=int(hours))
    selected = DF[DF["timestamp"] >= cutoff].copy()
    if site != "ALL":
        selected = selected[selected["site_id"] == site].copy()

    latest = selected[selected["timestamp"] == selected["timestamp"].max()].copy()
    health_value = f"{latest['health_score'].mean():.1f}/100"
    critical_value = str(int((latest["status"] == "Critical").sum()))
    anomaly_value = str(int(selected["anomaly"].sum()))
    risk_value = str(int((latest["breach_risk_pct"] > 60).sum()))
    user_value = f"{int(latest['connected_users'].sum()):,}"

    architect_map = build_architect_map(latest)

    labels = {
        "health_score": "Health Score", "dl_throughput_mbps": "DL Throughput (Mbps)",
        "latency_ms": "Latency (ms)", "prb_utilization_pct": "PRB Utilization (%)",
        "sinr_db": "SINR (dB)", "handover_success_pct": "Handover Success (%)",
        "rrc_success_pct": "RRC Success (%)",
    }
    trend_data = selected.groupby("timestamp", as_index=False)[kpi].mean()
    trend = px.line(trend_data, x="timestamp", y=kpi, title=f"Network Trend: {labels[kpi]}", template="plotly_dark")
    anomaly_rows = selected[selected["anomaly"]]
    trend.add_trace(go.Scatter(x=anomaly_rows["timestamp"], y=anomaly_rows[kpi], mode="markers", name="ML anomaly", marker={"symbol": "x", "size": 9, "color": "#ef4444"}))

    counts = latest["status"].value_counts().reset_index()
    counts.columns = ["status", "cells"]
    donut = px.pie(counts, names="status", values="cells", hole=0.65, title="Current Cell Status", color="status", color_discrete_map=STATUS_COLORS, template="plotly_dark")

    pivot = latest.pivot_table(index="site_id", columns="sector", values="health_score", aggfunc="mean")
    heatmap = px.imshow(pivot, aspect="auto", text_auto=".0f", zmin=40, zmax=100, color_continuous_scale="RdYlGn", title="Sector Health Heatmap", labels={"x": "Sector", "y": "Site", "color": "Health"}, template="plotly_dark")

    rca_counts = latest["root_cause"].value_counts().head(8).sort_values()
    rca_chart = px.bar(x=rca_counts.values, y=rca_counts.index, orientation="h", title="Current Probable Root Causes", labels={"x": "Cells", "y": "Probable root cause"}, template="plotly_dark", color=rca_counts.values, color_continuous_scale="OrRd")
    rca_chart.update_coloraxes(showscale=False)

    columns_to_show = [
        "site_id", "cell_id", "sector", "status", "health_score", "breach_risk_pct",
        "azimuth_deg", "beamwidth_deg", "coverage_radius_km", "band", "pci", "arfcn",
        "antenna_height_m", "mechanical_tilt_deg", "electrical_tilt_deg", "root_cause",
        "rca_confidence_pct", "rca_evidence", "recommended_action", "connected_users",
        "prb_utilization_pct", "sinr_db", "rrc_success_pct", "handover_success_pct",
        "dl_throughput_mbps", "latency_ms", "packet_loss_pct", "availability_pct",
    ]
    table_frame = latest[columns_to_show].round(2).sort_values(["health_score", "breach_risk_pct"], ascending=[True, False])
    table_data = table_frame.to_dict("records")
    table_columns = [{"name": column.replace("_", " ").title(), "id": column} for column in columns_to_show]

    for figure in (trend, donut, heatmap, rca_chart):
        figure.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#dbeafe", margin={"l": 35, "r": 20, "t": 55, "b": 35})

    return health_value, critical_value, anomaly_value, risk_value, user_value, architect_map, trend, donut, heatmap, rca_chart, table_data, table_columns


@app.callback(
    Output("assistant-window", "className"),
    Input("assistant-launcher", "n_clicks"),
    Input("assistant-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_assistant_window(open_clicks: int, close_clicks: int):
    from dash import ctx
    if ctx.triggered_id == "assistant-launcher":
        return "assistant-window assistant-visible"
    return "assistant-window assistant-hidden"


@app.callback(
    Output("assistant-input", "value"),
    Input({"type": "assistant-suggestion", "index": __import__("dash").ALL}, "n_clicks"),
    State("assistant-suggestions-store", "data"),
    prevent_initial_call=True,
)
def use_suggestion(clicks, suggestions):
    from dash import ctx
    if not ctx.triggered_id or not any(clicks):
        return no_update
    return suggestions[int(ctx.triggered_id["index"])]


@app.callback(
    Output("assistant-answer", "children"),
    Input("assistant-send", "n_clicks"),
    Input("assistant-clear", "n_clicks"),
    Input("assistant-input", "n_submit"),
    State("assistant-input", "value"),
    prevent_initial_call=True,
)
def run_assistant(send_clicks, clear_clicks, submit_count, query):
    from dash import ctx
    if ctx.triggered_id == "assistant-clear":
        return html.Div([html.Div("Try a quick analysis", className="empty-title"), html.Div("CHN-001-S2 or Analyse CHN-001", className="empty-subtitle")], className="assistant-empty")
    return render_assistant_response(answer_query(DF, query or ""))

