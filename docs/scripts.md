# Python Scripts Documentation

## Overview

This document provides detailed documentation for each Python script in the project.

---

## 📄 Script Summary

| Script | Purpose | Output |
|--------|---------|--------|
| `data_cleaning.py` | Data loading, cleaning, and main analysis | 21 PNGs + report.txt |
| `analysis.py` | Core visualization functions | Supporting module |
| `advanced_analytics.py` | Advanced analytics and predictions | 7 PNGs |

---

## 📘 data_cleaning.py

### Purpose
The main analysis script that handles the complete workflow from data loading to visualization generation.

### How to Run

```bash
cd path/to/UIDAI_new/scripts
python data_cleaning.py
```

### Functions

#### `load_all_csv_files(pattern, prefix='')`

Loads and combines multiple CSV files matching a pattern.

**Parameters:**
- `pattern` (str): Glob pattern to match files (e.g., `'api_data_*.csv'`)
- `prefix` (str): Optional prefix for logging

**Returns:**
- `DataFrame`: Combined data from all matching files

**Example:**
```python
bio_df = load_all_csv_files('api_data_aadhar_biometric_*.csv')
```

---

#### `preprocess_data(df, dataset_type)`

Cleans and prepares the data for analysis.

**Parameters:**
- `df` (DataFrame): Raw data
- `dataset_type` (str): One of `'biometric'`, `'demographic'`, or `'enrolment'`

**Processing Steps:**
1. Converts date strings to datetime objects
2. Removes whitespace from state names
3. Extracts month, day of week, week number
4. Renames columns to standardized format

**Returns:**
- `DataFrame`: Cleaned and preprocessed data

**Example:**
```python
bio_df = preprocess_data(bio_df, 'biometric')
```

---

#### `analyze_state_distribution(df, title, filename)`

Creates a horizontal bar chart showing state-wise distribution.

**Parameters:**
- `df` (DataFrame): Preprocessed data
- `title` (str): Chart title
- `filename` (str): Output PNG filename

**Output:**
- PNG file saved to `images/` folder
- Returns state totals DataFrame

**Visualization:**
- Top 20 states by volume
- Bar labels showing values in millions

---

#### `analyze_age_group_distribution(df, title, filename, dataset_type)`

Creates pie chart and bar chart showing age group distribution.

**Parameters:**
- `df` (DataFrame): Preprocessed data
- `title` (str): Chart title
- `filename` (str): Output PNG filename
- `dataset_type` (str): Type of dataset

**Output:**
- PNG file with two subplots (pie chart + bar chart)
- Returns age totals Series

---

#### `analyze_temporal_trends(df, title, filename)`

Analyzes and visualizes time-based patterns.

**Parameters:**
- `df` (DataFrame): Preprocessed data
- `title` (str): Chart title
- `filename` (str): Output PNG filename

**Output:**
- 4-panel visualization:
  - Daily time series
  - Day of week distribution
  - Monthly distribution
  - Weekly trend line

---

#### `detect_anomalies(df, title, filename)`

Identifies statistical outliers using rolling mean method.

**Algorithm:**
1. Calculate 7-day rolling mean
2. Calculate 7-day rolling standard deviation
3. Flag days outside ±2 standard deviations

**Output:**
- Line chart with anomaly points highlighted
- Returns DataFrame of anomaly days

---

#### `compare_update_types(bio_df, demo_df, filename)`

Compares biometric and demographic update patterns by state.

**Output:**
- 2-panel visualization:
  - Grouped bar chart (absolute values)
  - Diverging bar chart (percentages)

---

#### `analyze_geographic_hotspots(df, title, filename)`

Identifies top districts by volume.

**Output:**
- Horizontal bar chart of top 25 districts
- Returns DataFrame with district totals

---

#### `analyze_youth_patterns(df_list, labels, filename)`

Analyzes youth (5-17 age group) patterns across all datasets.

**Parameters:**
- `df_list` (list): List of DataFrames
- `labels` (list): Labels for each dataset
- `filename` (str): Output PNG filename

---

#### `create_pincode_heatmap(df, title, filename)`

Analyzes volume by pincode regions (first 2 digits).

---

#### `generate_insights_report(bio_df, demo_df, enrol_df, insights)`

Generates a formatted text report with key findings.

**Returns:**
- Formatted string template for report

---

#### `main()`

Main execution function that runs all analyses in sequence.

**Steps:**
1. Load data (Step 1)
2. Preprocess data (Step 2)
3. Analyze state distribution (Step 3)
4. Analyze age groups (Step 4)
5. Analyze temporal trends (Step 5)
6. Detect anomalies (Step 6)
7. Compare update types (Step 7)
8. Identify hotspots (Step 8)
9. Analyze youth patterns (Step 9)
10. Analyze pincode regions (Step 10)
11. Create dashboard (Step 11)
12. Generate report (Step 12)

---

## 📗 analysis.py

### Purpose
A streamlined module containing core visualization functions. This can be imported and used independently.

### How to Run

```bash
cd path/to/UIDAI_new/scripts
python analysis.py
```

### Key Functions

| Function | Description |
|----------|-------------|
| `load_processed_data()` | Loads and preprocesses all three datasets |
| `analyze_state_distribution()` | State-wise distribution chart |
| `analyze_age_distribution()` | Age group distribution chart |
| `analyze_temporal_trends()` | Temporal analysis charts |
| `detect_anomalies()` | Anomaly detection visualization |

### Configuration

```python
# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')

# Color palette
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'accent': '#E53935',
    'success': '#43A047'
}
```

---

## 📕 advanced_analytics.py

### Purpose
Performs advanced statistical analysis including machine learning techniques.

### How to Run

```bash
cd path/to/UIDAI_new/scripts
python advanced_analytics.py
```

### Functions

#### `load_data()`

Loads all datasets for advanced analysis.

---

#### `state_clustering_analysis(bio_df, demo_df, enrol_df)`

Clusters states using K-Means algorithm.

**Algorithm:**
1. Calculate service mix ratios for each state
2. Standardize features using StandardScaler
3. Apply K-Means clustering (k=4)
4. Visualize clusters

**Output:**
- `adv_01_state_clustering.png`

**Insights:**
- Identifies states with similar service patterns
- Groups states into 4 distinct clusters

---

#### `correlation_analysis(bio_df, demo_df, enrol_df)`

Analyzes correlations between different services.

**Output:**
- `adv_02_correlation_analysis.png`
- Correlation heatmap
- Scatter plot with trend line

---

#### `demand_forecasting(bio_df)`

Predicts future demand using linear regression.

**Algorithm:**
1. Calculate daily totals
2. Fit linear regression model
3. Forecast next 30 days
4. Visualize with confidence intervals

**Output:**
- `adv_03_demand_forecast.png`

---

#### `capacity_planning_analysis(bio_df, demo_df, enrol_df)`

Analyzes capacity requirements for enrollment centers.

**Metrics:**
- Average daily load per district
- Peak load vs average ratio
- Coefficient of variation (demand variability)

**Output:**
- `adv_04_capacity_planning.png`

**Recommendations:**
- Identifies high-load districts
- Calculates recommended capacity buffer

---

#### `regional_disparity_analysis(bio_df, demo_df, enrol_df)`

Analyzes differences between geographic regions.

**Regions Defined:**
- North (Delhi, UP, Punjab, etc.)
- South (Tamil Nadu, Karnataka, etc.)
- East (Bihar, West Bengal, etc.)
- West (Maharashtra, Gujarat, etc.)
- Central (MP, Chhattisgarh)
- Northeast (Assam, etc.)

**Output:**
- `adv_05_regional_disparity.png`
- Disparity index calculation

---

#### `predictive_indicators(bio_df, demo_df, enrol_df)`

Generates predictive indicators for planning.

**Indicators:**
- Weekly load pattern index
- State growth momentum
- Service mix ratio

**Output:**
- `adv_06_predictive_indicators.png`

---

#### `create_executive_summary(bio_df, demo_df, enrol_df)`

Creates a comprehensive executive summary visualization.

**Output:**
- `adv_07_executive_summary.png`

**Contents:**
- Key numbers overview
- Service distribution pie chart
- Top states bar chart
- Daily trends line chart
- Weekly patterns
- Key insights summary

---

#### `main()`

Main execution function for advanced analytics.

**Execution Order:**
1. Load data
2. State clustering analysis
3. Correlation analysis
4. Demand forecasting
5. Capacity planning analysis
6. Regional disparity analysis
7. Predictive indicators
8. Executive summary

---

## 🔧 Common Utilities

### Color Palette

All scripts use a consistent color scheme:

```python
COLORS = {
    'primary': '#1E88E5',    # Blue
    'secondary': '#FFC107',  # Yellow/Amber
    'accent': '#E53935',     # Red
    'success': '#43A047',    # Green
    'info': '#00ACC1'        # Cyan
}
```

### Directory Configuration

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'images')
```

### Matplotlib Configuration

```python
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
```

---

## ⚠️ Error Handling

All scripts include:
- Warning suppression for cleaner output
- Error handling for missing files
- Graceful fallback for missing columns

```python
import warnings
warnings.filterwarnings('ignore')
```

---

## 📊 Output Summary

| Script | PNG Files | Other Outputs |
|--------|-----------|---------------|
| data_cleaning.py | 21 | report.txt |
| analysis.py | 12 | - |
| advanced_analytics.py | 7 | - |

**Total: 28 visualization files + 1 report**
