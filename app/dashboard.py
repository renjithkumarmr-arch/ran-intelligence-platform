from __future__ import annotations
from dash import Dash, Input, Output, dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from .config import CONFIG
from .data import load_or_generate
from .analytics import enrich

RAW = load_or_generate()
DF = enrich(RAW)
COLORS = {"Healthy":"#22c55e", "Degraded":"#f59e0b", "Critical":"#ef4444"}
app = Dash(__name__, title=CONFIG["app"]["title"], suppress_callback_exceptions=True)
server = app.server

def card(title, value_id, subtitle):
    return html.Div([html.P(title, className="card-title"), html.H2(id=value_id), html.Small(subtitle)], className="kpi-card")

latest_ts = DF.timestamp.max()
sites = sorted(DF.site_id.unique())
app.layout = html.Div([
    dcc.Interval(id="refresh", interval=CONFIG["app"]["refresh_seconds"]*1000, n_intervals=0),
    html.Header([html.Div([html.H1("RAN Intelligence Platform"), html.P("5G NR service assurance, anomaly detection and SLA risk")]),
                 html.Div("LIVE DEMO DATA", className="live")]),
    html.Div([
      html.Label("Site"), dcc.Dropdown([{"label":"All Sites","value":"ALL"}]+[{"label":s,"value":s} for s in sites], "ALL", id="site-filter", clearable=False),
      html.Label("Time window"), dcc.Dropdown([6,12,24], 24, id="hours-filter", clearable=False),
      html.Label("KPI"), dcc.Dropdown({"health_score":"Health Score","dl_throughput_mbps":"DL Throughput","latency_ms":"Latency","prb_utilization_pct":"PRB Utilization","sinr_db":"SINR"}, "health_score", id="kpi-filter", clearable=False)
    ], className="filters"),
    html.Main([
      html.Div([card("Network Health", "health", "Weighted multi-KPI score"), card("Critical Cells", "critical", "Latest interval"), card("Anomalies", "anomalies", "Selected window"), card("SLA Risk > 60%", "risk", "Next 15 minutes"), card("Active Users", "users", "Latest interval")], className="card-grid"),
      html.Div([html.Div([dcc.Graph(id="trend")], className="panel wide"), html.Div([dcc.Graph(id="status-donut")], className="panel")], className="grid-2"),
      html.Div([html.Div([dcc.Graph(id="site-map")], className="panel"), html.Div([dcc.Graph(id="heatmap")], className="panel")], className="grid-2"),
      html.Div([html.H3("Cell Operations and Probable Root Cause"), dash_table.DataTable(id="cell-table", page_size=12, sort_action="native", filter_action="native", style_table={"overflowX":"auto"}, style_cell={"backgroundColor":"#0f172a","color":"#dbeafe","border":"1px solid #334155","padding":"9px","fontFamily":"Inter, sans-serif","textAlign":"left"}, style_header={"backgroundColor":"#1e293b","fontWeight":"700"}, style_data_conditional=[{"if":{"filter_query":"{status} = Critical"},"backgroundColor":"#4c0519"},{"if":{"filter_query":"{status} = Degraded"},"backgroundColor":"#451a03"}])], className="panel")
    ])
], className="page")

@app.callback([Output(x,"children") for x in ["health","critical","anomalies","risk","users"]] + [Output("trend","figure"),Output("status-donut","figure"),Output("site-map","figure"),Output("heatmap","figure"),Output("cell-table","data"),Output("cell-table","columns")], [Input("site-filter","value"),Input("hours-filter","value"),Input("kpi-filter","value"),Input("refresh","n_intervals")])
def update(site, hours, kpi, _):
    cutoff = DF.timestamp.max() - pd.Timedelta(hours=int(hours))
    d = DF[DF.timestamp >= cutoff]
    if site != "ALL": d = d[d.site_id == site]
    latest = d[d.timestamp == d.timestamp.max()].copy()
    health = f"{latest.health_score.mean():.1f}/100"
    critical = str((latest.status == "Critical").sum())
    anomalies = str(int(d.anomaly.sum()))
    risk = str(int((latest.breach_risk_pct > 60).sum()))
    users = f"{int(latest.connected_users.sum()):,}"
    labels={"health_score":"Health Score","dl_throughput_mbps":"DL Throughput (Mbps)","latency_ms":"Latency (ms)","prb_utilization_pct":"PRB Utilization (%)","sinr_db":"SINR (dB)"}
    agg=d.groupby("timestamp",as_index=False)[kpi].mean()
    trend=px.line(agg,x="timestamp",y=kpi,title=f"Network Trend: {labels[kpi]}",template="plotly_dark")
    anomalies_df=d[d.anomaly]
    trend.add_trace(go.Scatter(x=anomalies_df.timestamp,y=anomalies_df[kpi],mode="markers",name="ML anomaly",marker=dict(symbol="x",size=9,color="#ef4444")))
    counts=latest.status.value_counts().reset_index(); counts.columns=["status","cells"]
    donut=px.pie(counts,names="status",values="cells",hole=.65,title="Current Cell Status",color="status",color_discrete_map=COLORS,template="plotly_dark")
    site_agg=latest.groupby(["site_id","site_name","latitude","longitude"],as_index=False).agg(health_score=("health_score","mean"),connected_users=("connected_users","sum"),breach_risk_pct=("breach_risk_pct","max"))
    m=px.scatter_map(site_agg,lat="latitude",lon="longitude",color="health_score",size="connected_users",hover_name="site_name",hover_data=["site_id","breach_risk_pct"],zoom=4.1,height=420,color_continuous_scale="RdYlGn",range_color=[40,100],map_style="carto-darkmatter",title="Site Health and Load")
    pivot=latest.pivot_table(index="site_id",columns="sector",values="health_score",aggfunc="mean")
    heat=px.imshow(pivot,aspect="auto",text_auto=".0f",zmin=40,zmax=100,color_continuous_scale="RdYlGn",title="Sector Health Heatmap",labels={"x":"Sector","y":"Site","color":"Health"},template="plotly_dark")
    cols=["site_id","cell_id","status","health_score","breach_risk_pct","connected_users","prb_utilization_pct","sinr_db","dl_throughput_mbps","latency_ms","root_cause"]
    table=latest[cols].round(1).sort_values(["health_score","breach_risk_pct"]).to_dict("records")
    columns=[{"name":c.replace("_"," ").title(),"id":c} for c in cols]
    for fig in (trend,donut,m,heat): fig.update_layout(paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",font_color="#dbeafe",margin=dict(l=35,r=20,t=55,b=35))
    return health,critical,anomalies,risk,users,trend,donut,m,heat,table,columns
