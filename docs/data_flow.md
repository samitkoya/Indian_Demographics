# Data Flow Documentation

## Overview

This document explains how data flows through the UIDAI Aadhaar Analysis project, from raw CSV files to final PNG visualizations and insights.

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │ CSV Files   │───▶│   Python    │───▶│  Analysis   │───▶│    PNG      │  │
│   │  (data/)    │    │  Scripts    │    │  Results    │    │  (images/)  │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│          │                  │                  │                  │          │
│          ▼                  ▼                  ▼                  ▼          │
│   - Biometric CSV    - Load & Clean    - Statistics       - 28 Charts      │
│   - Demographic CSV  - Preprocess      - Aggregations     - Dashboard      │
│   - Enrollment CSV   - Analyze         - Insights         - Report.txt     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📥 Input: CSV Data Files

### Location
All CSV files are stored in the `data/` folder.

### File Types

| File Pattern | Description | Records |
|--------------|-------------|---------|
| `api_data_aadhar_biometric_*.csv` | Biometric update records | ~1.86M |
| `api_data_aadhar_demographic_*.csv` | Demographic update records | ~2.07M |
| `api_data_aadhar_enrolment_*.csv` | New enrollment records | ~1.01M |

### CSV Structure

#### Biometric Updates

| Column | Type | Description |
|--------|------|-------------|
| `date` | String | Date in DD-MM-YYYY format |
| `state` | String | State name |
| `district` | String | District name |
| `pincode` | Integer | 6-digit postal code |
| `bio_age_5_17` | Integer | Updates by age group 5-17 years |
| `bio_age_17_` | Integer | Updates by age group 17+ years |

#### Demographic Updates

| Column | Type | Description |
|--------|------|-------------|
| `date` | String | Date in DD-MM-YYYY format |
| `state` | String | State name |
| `district` | String | District name |
| `pincode` | Integer | 6-digit postal code |
| `demo_age_5_17` | Integer | Updates by age group 5-17 years |
| `demo_age_17_` | Integer | Updates by age group 17+ years |

#### New Enrollments

| Column | Type | Description |
|--------|------|-------------|
| `date` | String | Date in DD-MM-YYYY format |
| `state` | String | State name |
| `district` | String | District name |
| `pincode` | Integer | 6-digit postal code |
| `age_0_5` | Integer | Enrollments by age group 0-5 years |
| `age_5_17` | Integer | Enrollments by age group 5-17 years |
| `age_18_greater` | Integer | Enrollments by age group 18+ years |

---

## ⚙️ Processing: Python Scripts

### Step 1: Data Loading

The `load_all_csv_files()` function:
1. Finds all CSV files matching the pattern
2. Reads each file using `pandas.read_csv()`
3. Concatenates all files into a single DataFrame
4. Returns the combined dataset

```python
# Example: Loading biometric data
bio_df = load_all_csv_files('api_data_aadhar_biometric_*.csv')
```

### Step 2: Data Preprocessing

The `preprocess_data()` function performs:

| Operation | Description |
|-----------|-------------|
| Date Conversion | Converts string dates to datetime objects |
| State Cleaning | Strips whitespace from state names |
| Time Features | Extracts month, day of week, week number |
| Column Renaming | Standardizes column names across datasets |

```python
# Processing flow
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['state'] = df['state'].str.strip()
df['month'] = df['date'].dt.month
df['day_name'] = df['date'].dt.day_name()
```

### Step 3: Data Analysis

Multiple analysis functions process the data:

| Function | Input | Output |
|----------|-------|--------|
| `analyze_state_distribution()` | DataFrame | State totals + PNG chart |
| `analyze_age_group_distribution()` | DataFrame | Age totals + PNG chart |
| `analyze_temporal_trends()` | DataFrame | Time series + PNG chart |
| `detect_anomalies()` | DataFrame | Anomaly list + PNG chart |
| `analyze_geographic_hotspots()` | DataFrame | Hotspot list + PNG chart |

---

## 📤 Output: Visualizations

### Location
All PNG files are saved in the `images/` folder.

### Output Categories

```
images/
├── State Distribution Charts (3 files)
│   ├── 01_biometric_state_distribution.png
│   ├── 02_demographic_state_distribution.png
│   └── 03_enrollment_state_distribution.png
│
├── Age Group Charts (3 files)
│   ├── 04_biometric_age_distribution.png
│   ├── 05_demographic_age_distribution.png
│   └── 06_enrollment_age_distribution.png
│
├── Temporal Trend Charts (3 files)
│   ├── 07_biometric_temporal_trends.png
│   ├── 08_demographic_temporal_trends.png
│   └── 09_enrollment_temporal_trends.png
│
├── Anomaly Detection Charts (3 files)
│   ├── 10_biometric_anomalies.png
│   ├── 11_demographic_anomalies.png
│   └── 12_enrollment_anomalies.png
│
├── Comparison & Analysis (5 files)
│   ├── 13_biometric_vs_demographic.png
│   ├── 14-16_hotspots.png (3 files)
│   └── 17_youth_analysis.png
│
├── Pincode Analysis (3 files)
│   ├── 18_pincode_biometric.png
│   ├── 19_pincode_demographic.png
│   └── 20_pincode_enrollment.png
│
├── Dashboard (1 file)
│   └── 21_comprehensive_dashboard.png
│
└── Advanced Analytics (7 files)
    ├── adv_01_state_clustering.png
    ├── adv_02_correlation_analysis.png
    ├── adv_03_demand_forecast.png
    ├── adv_04_capacity_planning.png
    ├── adv_05_regional_disparity.png
    ├── adv_06_predictive_indicators.png
    └── adv_07_executive_summary.png
```

---

## 🔄 Complete Data Flow Example

Here's a complete example showing how biometric data flows through the pipeline:

### 1. Raw Data (CSV)

```csv
date,state,district,pincode,bio_age_5_17,bio_age_17_
01-03-2025,Uttar Pradesh,Lucknow,226001,150,320
01-03-2025,Uttar Pradesh,Lucknow,226002,145,315
```

### 2. Loaded DataFrame

```python
# After loading
bio_df = pd.DataFrame({
    'date': ['01-03-2025', '01-03-2025'],
    'state': ['Uttar Pradesh', 'Uttar Pradesh'],
    'district': ['Lucknow', 'Lucknow'],
    'pincode': [226001, 226002],
    'bio_age_5_17': [150, 145],
    'bio_age_17_': [320, 315]
})
```

### 3. Preprocessed DataFrame

```python
# After preprocessing
bio_df = pd.DataFrame({
    'date': [datetime(2025, 3, 1), datetime(2025, 3, 1)],
    'state': ['Uttar Pradesh', 'Uttar Pradesh'],
    'district': ['Lucknow', 'Lucknow'],
    'pincode': [226001, 226002],
    'youth_5_17': [150, 145],        # Renamed
    'adult_17_plus': [320, 315],     # Renamed
    'month': [3, 3],                  # Added
    'day_name': ['Saturday', 'Saturday'],  # Added
    'week': [9, 9]                    # Added
})
```

### 4. Aggregated Results

```python
# State-wise totals
state_totals = bio_df.groupby('state')[['youth_5_17', 'adult_17_plus']].sum()
# Output:
#                    youth_5_17  adult_17_plus  total
# Uttar Pradesh          295           635       930
```

### 5. Final Visualization (PNG)

The aggregated data is then plotted using matplotlib/seaborn and saved as a PNG file in the `images/` folder.

---

## 📊 Report Generation

The `generate_insights_report()` function creates a text report containing:

1. **Dataset Overview** - Record counts and date ranges
2. **Geographic Findings** - Top states and districts
3. **Age Group Patterns** - Distribution percentages
4. **Temporal Insights** - Peak days and weekly patterns
5. **Anomalies** - Detected outliers
6. **Recommendations** - Actionable suggestions

Output: `report.txt` in the project root folder.

---

## 🔗 Script Dependencies

```
data_cleaning.py
    ↓
    ├── Loads CSV files
    ├── Preprocesses data
    ├── Runs analysis functions
    ├── Generates main visualizations (21 PNGs)
    └── Creates report.txt

advanced_analytics.py
    ↓
    ├── Loads preprocessed data
    ├── Runs advanced analytics
    ├── State clustering (K-Means)
    ├── Demand forecasting (Linear Regression)
    ├── Capacity planning
    └── Generates advanced visualizations (7 PNGs)
```

---

## ✅ Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1. Load | 12 CSV files | `load_all_csv_files()` | 3 DataFrames |
| 2. Clean | 3 DataFrames | `preprocess_data()` | Cleaned DataFrames |
| 3. Analyze | Cleaned DataFrames | Analysis functions | Statistics & Aggregations |
| 4. Visualize | Statistics | Matplotlib/Seaborn | 28 PNG files |
| 5. Report | Analysis results | `generate_insights_report()` | report.txt |
