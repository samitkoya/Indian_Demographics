# 🇮🇳 UIDAI Aadhaar Data Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![pandas](https://img.shields.io/badge/pandas-1.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Unlocking Societal Trends in Aadhaar Enrollment and Update Data**

*Analyzing 4.94 million records across 36 states*

</div>

---

## 📋 Overview

This project performs comprehensive analysis of **UIDAI Aadhaar data** to identify patterns, trends, and actionable insights for resource optimization and policy planning.

| Dataset | Records | Description |
|---------|---------|-------------|
| Biometric Updates | 1.86M | Fingerprint & iris updates |
| Demographic Updates | 2.07M | Address, name, DoB changes |
| New Enrollments | 1.01M | Fresh Aadhaar registrations |

**Data Period:** March 2025 – December 2025

---

## 📁 Project Structure

```
Indian_Demographics/
├── 📂 scripts/                  # Python analysis scripts
│   ├── data_cleaning.py         # Main analysis (21 visualizations)
│   ├── analysis.py              # Core visualization functions
│   └── advanced_analytics.py    # ML-based analytics (7 visualizations)
│
├── 📂 data/                     # 12 CSV data files (~200MB)
├── 📂 images/                   # 28 PNG visualizations + dashboard
├── 📂 docs/                     # Detailed documentation
│
├── README.md                    # Project overview (this file)
├── walkthrough.md               # Complete analysis guide
├── report.txt                   # Key findings summary
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
cd scripts

# Generate 21 main visualizations + report
python data_cleaning.py

# Generate 7 advanced analytics visualizations  
python advanced_analytics.py
```

### 3. View Results

- **Visualizations:** Check the `images/` folder
- **Key Findings:** Read `report.txt`
- **Full Walkthrough:** See [walkthrough.md](walkthrough.md)

---

## 📊 Key Findings

| Metric | Value |
|--------|-------|
| **Top State** | Uttar Pradesh (dominates all categories) |
| **Peak Day** | Tuesday (highest activity) |
| **Child Enrollments** | 65% of new registrations |
| **Bio vs Demo Ratio** | 1.42:1 |
| **Anomalies Detected** | 11 days |

### Highlights

- 🗺️ **Geographic:** UP leads all services; 19 states show activity across all categories
- 👶 **Age Patterns:** 65% of enrollments are children 0-5 years
- 📅 **Temporal:** Weekdays show 50% higher activity than weekends
- 📈 **Predictive:** 4 distinct state clusters identified for targeted planning

---

## 📈 Visualizations Generated

### Main Analysis (21 Charts)

| Category | Count | Description |
|----------|-------|-------------|
| State Distribution | 3 | Top 20 states by volume |
| Age Groups | 3 | Youth vs adult breakdown |
| Temporal Trends | 3 | Daily, weekly, monthly patterns |
| Anomaly Detection | 3 | Statistical outliers |
| Hotspots | 3 | Top 25 districts |
| Other | 6 | Comparisons, dashboard |

### Advanced Analytics (7 Charts)

| Chart | Description |
|-------|-------------|
| State Clustering | K-Means grouping |
| Correlation Analysis | Service relationships |
| Demand Forecast | 30-day prediction |
| Capacity Planning | Resource allocation |
| Regional Disparity | Geographic equity |
| Executive Summary | All-in-one dashboard |

---

## 💡 Recommendations

1. **Resource Optimization** – Deploy additional centers in high-demand districts
2. **Youth Campaigns** – Partner with schools for enrollment drives
3. **Weekend Services** – Consider extended hours in busy areas
4. **Anomaly Monitoring** – Implement real-time volume alerts
5. **Digital Promotion** – Push mAadhaar for demographic updates

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [walkthrough.md](walkthrough.md) | Complete analysis with visuals |
| [docs/data_flow.md](docs/data_flow.md) | Data pipeline explanation |
| [docs/scripts.md](docs/scripts.md) | Python scripts reference |
| [docs/requirements.md](docs/requirements.md) | Installation guide |
| [docs/outputs.md](docs/outputs.md) | Output files catalog |

---

## 🛠️ Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

See [requirements.txt](requirements.txt) for exact versions.

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

---

<div align="center">
<i>Built for UIDAI Hackathon 2026</i>
</div>
