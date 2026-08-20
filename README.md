AI-Powered 5G RAN Service Assurance Platform

🚀 Overview

An end-to-end 5G RAN Operations Analytics and Service Assurance Platform built using Python, Dash, Plotly, Machine Learning, and Telecom Domain Intelligence.

The platform transforms raw RAN telemetry into actionable operational insights by providing:

Network Health Scoring
Machine Learning Anomaly Detection
SLA Breach Prediction
Multi-KPI Root Cause Analysis (RCA)
Capacity & Congestion Analytics
RF Health Visualization
Architect-Level Geographic Cell Coverage Mapping
Interactive RAN Operations Assistant

The objective is to help Network Operations Centers (NOCs) move from reactive monitoring to proactive service assurance.

📸 Dashboard Preview
Executive Operations Dashboard

✅ Multi-KPI Health Monitoring

✅ SLA Risk Prediction

✅ Root Cause Analysis

✅ Site & Sector Visibility

✅ Architect-Level Cell Coverage View

✅ Interactive Operations Assistant


🎯 Business Problems Solved

Traditional NOC teams often ask:

Is the network healthy right now?
Which sites or cells are degraded?
Which cells are likely to breach SLA?
Are issues caused by Radio, RF, Mobility, Transport, or Capacity?
Which subscribers are most likely impacted?
What should operations teams investigate first?

This platform answers these questions automatically.

🏗 Solution Architecture
                         +----------------+
                         | RAN Telemetry  |
                         +--------+-------+
                                  |
                                  v
                     +------------------------+
                     | Data Processing Layer  |
                     +-----------+------------+
                                 |
                                 v
                    +-------------------------+
                    | KPI Normalization       |
                    | Health Score Engine     |
                    +-----------+-------------+
                                |
        +-----------------------+----------------------+
        |                                              |
        v                                              v
+-------------------+                     +----------------------+
| Anomaly Detection |                     | SLA Prediction       |
| Isolation Forest  |                     | Random Forest        |
+---------+---------+                     +----------+-----------+
          |                                         |
          +----------------+------------------------+
                           |
                           v
               +-------------------------+
               | Level 2 RCA Engine      |
               +------------+------------+
                            |
                            v
          +-------------------------------------------+
          | Dash / Plotly Visualization Layer         |
          +-------------------------------------------+
                            |
          +-------------------------------------------+
          | RAN Operations Assistant                  |
          +-------------------------------------------+

📊 Key Features
1. Health Score Engine

Each cell is assigned a Health Score between:

0 – 100


Based on:

Availability
Downlink Throughput
Uplink Throughput
Latency
Packet Loss
PRB Utilization
SINR
RRC Success Rate
Handover Success Rate
Classification
Health Score	Status80–100	Healthy
60–79	Degraded
Below 60	Critical
2. Machine Learning Anomaly Detection

Uses:

Isolation Forest


To identify unusual behavior before service impact occurs.

Examples:

Latency spikes
Throughput degradation
RF anomalies
Congestion patterns
Traffic irregularities
3. SLA Breach Prediction

Uses:

Random Forest Classifier


To predict:

Will this cell likely breach SLA within the next 15 minutes?

Output:

0% - 100% Risk Score


This enables proactive operational response.

4. Level 2 Root Cause Analysis

Instead of simple threshold monitoring, the platform correlates multiple KPIs.

Capacity Congestion

Correlates:

High PRB Utilization
+
Low Throughput
+
High User Volume

RF Quality / Interference

Correlates:

Low SINR
+
Throughput Degradation
+
Mobility Issues

Mobility Degradation

Correlates:

Low Handover Success
+
Coverage Issues
+
Neighbour Relation Problems

Transport / Backhaul Issues

Correlates:

High Latency
+
Packet Loss
+
Healthy RF Conditions

Access Failure

Correlates:

Low RRC Success
+
RF Issues
+
Congestion


Each RCA provides:

Probable Root Cause
RCA Confidence
Supporting Evidence
Recommended Action
🗺 Architect-Level RF Visualization

Traditional dashboards show sites.

This platform visualizes:

Site
+
Sector
+
Azimuth
+
Beamwidth
+
Coverage Radius


Each sector is represented as a directional coverage fan.

Displayed information:

Site ID
Cell ID
Sector
Band
PCI
Health Score
SLA Risk
Root Cause
Users
PRB Utilization
SINR
Visual Status

🟢 Healthy

🟠 Degraded

🔴 Critical

🤖 RAN Operations Assistant

Version 2.5 introduces a floating operations assistant.

The assistant performs live analysis using platform telemetry and RCA intelligence.

Supported Queries
CHN-001-S2

Why is CHN-001-S2 degraded?

Analyse CHN-001

Highest risk cell

Show congested cells

Show critical cells

Show transport issues

Example Response
Cell: CHN-001-S2

Health Score: 68

SLA Risk: 72%

Probable Cause:
Radio Capacity Congestion

Confidence:
90%

Evidence:
PRB Utilization = 94%
DL Throughput = 52 Mbps
Connected Users = 487
SINR = 17 dB

Recommendation:
Review carrier utilization and busy-hour demand.


No external AI service is required.

All analysis is performed locally using enriched telemetry data.

🛠 Technology Stack
Frontend
Dash
Plotly
Backend
Python
Data Engineering
Pandas
NumPy
Machine Learning
Scikit-Learn
Isolation Forest
Random Forest
Visualization
Plotly Maps
Heatmaps
KPI Trends
Interactive Dashboards
📁 Repository Structure
ran-intelligence-platform/
│
├── app/
│   ├── analytics.py
│   ├── assistant.py
│   ├── dashboard.py
│   ├── data.py
│   └── assets/
│
├── config/
│   └── config.yaml
│
├── data/
│
├── tests/
│
├── run.py
├── requirements.txt
├── Dockerfile
└── README.md

⚡ Quick Start
Create Virtual Environment
python -m venv .venv

Activate Environment
Windows
.\.venv\Scripts\Activate.ps1

Linux/macOS
source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Start Application
python run.py


Open:

http://localhost:8050

🎓 Interview Talking Points
Problem Solved

Traditional NOCs are reactive. This platform enables:

Predict
↓
Detect
↓
Explain
↓
Recommend


before SLA degradation impacts users.

Key Innovations
Multi-KPI Health Scoring
ML-based Anomaly Detection
SLA Risk Prediction
Correlation-Based RCA
Architect-Level Sector Visualization
Interactive Operations Assistant
Value Delivered
Proactive Operations
Faster Troubleshooting
Reduced Mean Time To Resolution (MTTR)
Better Network Visibility
Operational Decision Support
🚀 Future Roadmap
Version 3.0
Azure OpenAI Integration
Natural Language Queries
Executive Summary Generation
Incident Report Generation
Microsoft Teams Integration
Version 4.0
O-RAN Analytics
xApp/rApp Support
Multi-Vendor Correlation
Digital Twin Integration
Version 5.0
Autonomous RCA
Closed-Loop Automation
Self-Healing Recommendations
👨‍💻 Author

Renjith Kumar M R

Senior Network Architect | Telecom & Networking Specialist

Areas of Interest:

5G & Open RAN
Network Analytics
AI for Telecom Operations
Service Assurance
Cloud-Native Networking
Generative AI for Operations
