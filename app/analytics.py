from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .config import CONFIG
from .data import KPI_COLUMNS

def _metric_score(series: pd.Series, warning: float, critical: float, direction: str) -> pd.Series:
    x = series.astype(float)
    if direction == "high":
        span = max(warning - critical, 1e-6)
        return np.where(x >= warning, 100, np.where(x <= critical, 35, 35 + 65*(x-critical)/span))
    span = max(critical - warning, 1e-6)
    return np.where(x <= warning, 100, np.where(x >= critical, 35, 100 - 65*(x-warning)/span))

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["cell_id", "timestamp"]).copy()
    total = np.zeros(len(out))
    for metric, weight in CONFIG["weights"].items():
        t = CONFIG["thresholds"][metric]
        total += weight * _metric_score(out[metric], t["warning"], t["critical"], t["direction"])
    out["health_score"] = np.clip(total, 0, 100).round(1)
    out["status"] = np.select([out.health_score < 60, out.health_score < 80], ["Critical", "Degraded"], default="Healthy")
    out["anomaly"] = False
    for _, idx in out.groupby("cell_id").groups.items():
        part = out.loc[idx, KPI_COLUMNS]
        model = Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()),
                          ("model", IsolationForest(contamination=.035, random_state=42))])
        out.loc[idx, "anomaly"] = model.fit_predict(part) == -1
    out["breach_now"] = (out.health_score < 60).astype(int)
    out["future_breach"] = out.groupby("cell_id")["breach_now"].shift(-3).fillna(0).astype(int)
    features = KPI_COLUMNS + ["connected_users", "health_score"]
    split = max(int(len(out)*.8), 1)
    train = out.iloc[:split]
    if train.future_breach.nunique() > 1:
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(train[features], train.future_breach)
        out["breach_risk_pct"] = (clf.predict_proba(out[features])[:,1]*100).round(1)
    else:
        out["breach_risk_pct"] = np.clip((70-out.health_score)*2, 0, 100).round(1)
    out["root_cause"] = out.apply(root_cause, axis=1)
    return out

def root_cause(row: pd.Series) -> str:
    candidates = []
    for metric, cfg in CONFIG["thresholds"].items():
        value = row[metric]
        badness = (cfg["warning"]-value) if cfg["direction"] == "high" else (value-cfg["warning"])
        if badness > 0: candidates.append((badness, metric))
    if not candidates: return "No dominant KPI"
    labels = {"prb_utilization_pct":"Capacity congestion", "sinr_db":"RF quality / interference",
              "handover_success_pct":"Mobility failure", "rrc_success_pct":"Access failure",
              "packet_loss_pct":"Transport packet loss", "latency_ms":"Transport latency",
              "availability_pct":"Availability degradation", "dl_throughput_mbps":"Downlink capacity",
              "ul_throughput_mbps":"Uplink capacity"}
    return labels[max(candidates)[1]]
