"""
UIDAI Aadhaar Advanced Analytics & Predictive Indicators
=========================================================
Advanced analysis including:
- Predictive modeling for demand forecasting
- Clustering analysis for regional patterns
- Correlation analysis
- Trend decomposition
- Capacity planning recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from glob import glob
import os
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Directory configuration for organized project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'images')

def load_data():
    """Load all datasets."""
    print("[LOAD] Loading datasets...")
    
    bio_files = glob(os.path.join(DATA_DIR, 'api_data_aadhar_biometric_*.csv'))
    demo_files = glob(os.path.join(DATA_DIR, 'api_data_aadhar_demographic_*.csv'))
    enrol_files = glob(os.path.join(DATA_DIR, 'api_data_aadhar_enrolment_*.csv'))
    
    bio_df = pd.concat([pd.read_csv(f) for f in bio_files], ignore_index=True)
    demo_df = pd.concat([pd.read_csv(f) for f in demo_files], ignore_index=True)
    enrol_df = pd.concat([pd.read_csv(f) for f in enrol_files], ignore_index=True)
    
    # Preprocess
    for df in [bio_df, demo_df, enrol_df]:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    print(f"  [OK] Loaded {len(bio_df):,} biometric, {len(demo_df):,} demographic, {len(enrol_df):,} enrollment records")
    
    return bio_df, demo_df, enrol_df


def state_clustering_analysis(bio_df, demo_df, enrol_df):
    """Cluster states based on their Aadhaar activity patterns."""
    print("\n[CLUSTER] Performing state clustering analysis...")
    
    # Aggregate by state
    bio_cols = [c for c in bio_df.columns if 'bio_' in c]
    demo_cols = [c for c in demo_df.columns if 'demo_' in c]
    enrol_cols = [c for c in enrol_df.columns if 'age_' in c]
    
    bio_state = bio_df.groupby('state')[bio_cols].sum()
    demo_state = demo_df.groupby('state')[demo_cols].sum()
    enrol_state = enrol_df.groupby('state')[enrol_cols].sum()
    
    # Combine features
    combined = pd.concat([
        bio_state.sum(axis=1).rename('bio_total'),
        demo_state.sum(axis=1).rename('demo_total'),
        enrol_state.sum(axis=1).rename('enrol_total')
    ], axis=1).fillna(0)
    
    combined['total'] = combined.sum(axis=1)
    combined['bio_ratio'] = combined['bio_total'] / combined['total']
    combined['demo_ratio'] = combined['demo_total'] / combined['total']
    combined['enrol_ratio'] = combined['enrol_total'] / combined['total']
    
    # Standardize features
    scaler = StandardScaler()
    features = ['bio_ratio', 'demo_ratio', 'enrol_ratio', 'total']
    X = scaler.fit_transform(combined[features])
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    combined['cluster'] = kmeans.fit_predict(X)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Scatter plot
    scatter = axes[0].scatter(combined['bio_ratio'], combined['demo_ratio'], 
                              c=combined['cluster'], cmap='viridis',
                              s=combined['total']/combined['total'].max()*500 + 50,
                              alpha=0.7, edgecolors='black')
    
    # Label top states
    top_states = combined.nlargest(10, 'total').index
    for state in top_states:
        axes[0].annotate(state[:15], (combined.loc[state, 'bio_ratio'], 
                                       combined.loc[state, 'demo_ratio']),
                        fontsize=8, ha='center')
    
    axes[0].set_xlabel('Biometric Update Ratio', fontsize=12)
    axes[0].set_ylabel('Demographic Update Ratio', fontsize=12)
    axes[0].set_title('State Clustering by Update Preferences\n(Size = Total Volume)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Cluster characteristics
    cluster_summary = combined.groupby('cluster')[['bio_ratio', 'demo_ratio', 'enrol_ratio', 'total']].mean()
    cluster_summary.plot(kind='bar', ax=axes[1])
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Average Value', fontsize=12)
    axes[1].set_title('Cluster Characteristics', fontsize=14, fontweight='bold')
    axes[1].legend(title='Metric')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_01_state_clustering.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [OK] State clustering complete - 4 clusters identified")
    return combined


def correlation_analysis(bio_df, demo_df, enrol_df):
    """Analyze correlations between different services and regions."""
    print("\n[CORR] Performing correlation analysis...")
    
    # Daily aggregations
    bio_daily = bio_df.groupby('date')[[c for c in bio_df.columns if 'bio_' in c]].sum().sum(axis=1)
    demo_daily = demo_df.groupby('date')[[c for c in demo_df.columns if 'demo_' in c]].sum().sum(axis=1)
    enrol_daily = enrol_df.groupby('date')[[c for c in enrol_df.columns if 'age_' in c]].sum().sum(axis=1)
    
    # Combine
    daily_data = pd.DataFrame({
        'Biometric': bio_daily,
        'Demographic': demo_daily,
        'Enrollment': enrol_daily
    }).dropna()
    
    # Correlation matrix
    corr_matrix = daily_data.corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0], square=True, fmt='.3f')
    axes[0].set_title('Correlation Between Services\n(Daily Volume)', fontsize=14, fontweight='bold')
    
    # Pairplot style scatter
    axes[1].scatter(daily_data['Biometric'], daily_data['Demographic'], alpha=0.5)
    z = np.polyfit(daily_data['Biometric'], daily_data['Demographic'], 1)
    p = np.poly1d(z)
    axes[1].plot(daily_data['Biometric'].sort_values(), 
                 p(daily_data['Biometric'].sort_values()), 
                 "r--", alpha=0.8, label=f'Trend line (r={corr_matrix.loc["Biometric", "Demographic"]:.3f})')
    axes[1].set_xlabel('Biometric Volume', fontsize=12)
    axes[1].set_ylabel('Demographic Volume', fontsize=12)
    axes[1].set_title('Biometric vs Demographic Correlation', fontsize=14, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_02_correlation_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Correlation analysis complete")
    print(f"    Bio-Demo correlation: {corr_matrix.loc['Biometric', 'Demographic']:.3f}")
    
    return corr_matrix


def demand_forecasting(bio_df):
    """Simple demand forecasting using trend analysis."""
    print("\n[FORECAST] Performing demand forecasting...")
    
    bio_cols = [c for c in bio_df.columns if 'bio_' in c]
    daily = bio_df.groupby('date')[bio_cols].sum().sum(axis=1).reset_index()
    daily.columns = ['date', 'volume']
    daily = daily.sort_values('date')
    daily['day_num'] = (daily['date'] - daily['date'].min()).dt.days
    
    # Linear regression for trend
    X = daily['day_num'].values.reshape(-1, 1)
    y = daily['volume'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    daily['predicted'] = model.predict(X)
    daily['residual'] = daily['volume'] - daily['predicted']
    
    # Future forecast (next 30 days)
    last_day = daily['day_num'].max()
    future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
    future_pred = model.predict(future_days)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Actual vs Predicted
    axes[0, 0].plot(daily['date'], daily['volume'], label='Actual', alpha=0.7)
    axes[0, 0].plot(daily['date'], daily['predicted'], label='Trend', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Volume')
    axes[0, 0].set_title('Demand Trend Analysis', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Residuals
    axes[0, 1].scatter(daily['date'], daily['residual'], alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].set_title('Residuals (Actual - Trend)', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Histogram of residuals
    axes[1, 0].hist(daily['residual'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    
    # Future forecast
    last_date = daily['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    axes[1, 1].plot(daily['date'], daily['volume'], label='Historical', alpha=0.7)
    axes[1, 1].plot(future_dates, future_pred, label='Forecast', color='green', linewidth=2, linestyle='--')
    axes[1, 1].axvline(x=last_date, color='red', linestyle=':', label='Forecast Start')
    axes[1, 1].fill_between(future_dates, future_pred * 0.9, future_pred * 1.1, alpha=0.2, color='green')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Volume')
    axes[1, 1].set_title('30-Day Demand Forecast', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_03_demand_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    trend_direction = 'increasing' if model.coef_[0] > 0 else 'decreasing'
    print(f"  [OK] Demand forecasting complete")
    print(f"    Trend direction: {trend_direction}")
    print(f"    Daily trend change: {model.coef_[0]:.2f} per day")
    
    return model


def capacity_planning_analysis(bio_df, demo_df, enrol_df):
    """Analyze capacity requirements for enrollment centers."""
    print("\n[CAPACITY] Performing capacity planning analysis...")
    
    # District-level load analysis
    bio_cols = [c for c in bio_df.columns if 'bio_' in c]
    demo_cols = [c for c in demo_df.columns if 'demo_' in c]
    enrol_cols = [c for c in enrol_df.columns if 'age_' in c]
    
    # Peak day analysis
    bio_df['total'] = bio_df[bio_cols].sum(axis=1)
    demo_df['total'] = demo_df[demo_cols].sum(axis=1)
    enrol_df['total'] = enrol_df[enrol_cols].sum(axis=1)
    
    # District daily averages and peaks
    bio_district = bio_df.groupby(['state', 'district']).agg({
        'total': ['mean', 'max', 'std']
    }).droplevel(0, axis=1)
    bio_district.columns = ['avg_daily', 'peak_daily', 'std_daily']
    bio_district['peak_ratio'] = bio_district['peak_daily'] / bio_district['avg_daily']
    bio_district = bio_district.reset_index()
    
    # Identify high-load districts
    high_load = bio_district.nlargest(20, 'avg_daily')
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Average vs Peak
    axes[0, 0].scatter(bio_district['avg_daily'], bio_district['peak_daily'], alpha=0.5, s=20)
    max_val = max(bio_district['avg_daily'].max(), bio_district['peak_daily'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
    axes[0, 0].plot([0, max_val], [0, max_val*2], 'g--', label='2:1 ratio', alpha=0.5)
    axes[0, 0].set_xlabel('Average Daily Load')
    axes[0, 0].set_ylabel('Peak Daily Load')
    axes[0, 0].set_title('Average vs Peak Load by District\n(Capacity Planning)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    # Top 20 high-load districts
    labels = [f"{row['district'][:15]}" for _, row in high_load.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    axes[0, 1].bar(x - width/2, high_load['avg_daily'], width, label='Average')
    axes[0, 1].bar(x + width/2, high_load['peak_daily'], width, label='Peak')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Daily Load')
    axes[0, 1].set_title('Top 20 High-Load Districts', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    
    # Peak ratio distribution
    axes[1, 0].hist(bio_district['peak_ratio'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=bio_district['peak_ratio'].median(), color='red', linestyle='--', 
                       label=f'Median: {bio_district["peak_ratio"].median():.2f}')
    axes[1, 0].set_xlabel('Peak / Average Ratio')
    axes[1, 0].set_ylabel('Number of Districts')
    axes[1, 0].set_title('Peak to Average Ratio Distribution\n(Capacity Buffer Needed)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    
    # Variability analysis (coefficient of variation)
    bio_district['cv'] = bio_district['std_daily'] / bio_district['avg_daily']
    high_cv = bio_district.nlargest(20, 'cv')
    axes[1, 1].barh(range(20), high_cv['cv'].values)
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([f"{row['district'][:15]}" for _, row in high_cv.iterrows()])
    axes[1, 1].set_xlabel('Coefficient of Variation')
    axes[1, 1].set_title('Most Variable Districts\n(Unpredictable Demand)', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_04_capacity_planning.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Capacity planning analysis complete")
    print(f"    Average peak ratio: {bio_district['peak_ratio'].mean():.2f}x")
    print(f"    Recommended capacity buffer: {(bio_district['peak_ratio'].quantile(0.95) - 1)*100:.0f}%")
    
    return bio_district


def regional_disparity_analysis(bio_df, demo_df, enrol_df):
    """Analyze regional disparities in Aadhaar services."""
    print("\n[REGION] Analyzing regional disparities...")
    
    # Define regions (simplified)
    region_mapping = {
        'North': ['Delhi', 'Haryana', 'Punjab', 'Rajasthan', 'Uttar Pradesh', 
                  'Uttarakhand', 'Himachal Pradesh', 'Jammu and Kashmir', 'Ladakh', 'Chandigarh'],
        'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 
                  'Puducherry', 'Lakshadweep', 'Andaman and Nicobar Islands'],
        'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Sikkim'],
        'West': ['Gujarat', 'Maharashtra', 'Goa', 'Dadra and Nagar Haveli and Daman and Diu'],
        'Central': ['Madhya Pradesh', 'Chhattisgarh'],
        'Northeast': ['Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 
                      'Mizoram', 'Nagaland', 'Tripura']
    }
    
    # Reverse mapping
    state_to_region = {}
    for region, states in region_mapping.items():
        for state in states:
            state_to_region[state] = region
    
    # Map regions
    bio_df['region'] = bio_df['state'].map(state_to_region).fillna('Other')
    demo_df['region'] = demo_df['state'].map(state_to_region).fillna('Other')
    enrol_df['region'] = enrol_df['state'].map(state_to_region).fillna('Other')
    
    # Aggregate by region
    bio_cols = [c for c in bio_df.columns if 'bio_' in c]
    demo_cols = [c for c in demo_df.columns if 'demo_' in c]
    enrol_cols = [c for c in enrol_df.columns if 'age_' in c]
    
    bio_region = bio_df.groupby('region')[bio_cols].sum().sum(axis=1)
    demo_region = demo_df.groupby('region')[demo_cols].sum().sum(axis=1)
    enrol_region = enrol_df.groupby('region')[enrol_cols].sum().sum(axis=1)
    
    regional_data = pd.DataFrame({
        'Biometric': bio_region,
        'Demographic': demo_region,
        'Enrollment': enrol_region
    }).fillna(0)
    regional_data['Total'] = regional_data.sum(axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Stacked bar
    regional_data[['Biometric', 'Demographic', 'Enrollment']].plot(kind='bar', stacked=True, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('Total Volume')
    axes[0, 0].set_title('Regional Distribution by Service Type', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Pie chart
    axes[0, 1].pie(regional_data['Total'], labels=regional_data.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Regional Share of Total Activity', fontsize=14, fontweight='bold')
    
    # Normalized comparison
    regional_norm = regional_data[['Biometric', 'Demographic', 'Enrollment']].div(regional_data['Total'], axis=0) * 100
    regional_norm.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Region')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title('Service Mix by Region (Normalized)', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title='Service')
    
    # Disparity index (Gini-like)
    disparity = regional_data['Total'] / regional_data['Total'].sum()
    ideal = 1 / len(disparity)
    disparity_index = abs(disparity - ideal).sum() / 2
    
    axes[1, 1].bar(regional_data.index, disparity * 100, color='steelblue')
    axes[1, 1].axhline(y=ideal * 100, color='red', linestyle='--', label=f'Equal distribution ({ideal*100:.1f}%)')
    axes[1, 1].set_xlabel('Region')
    axes[1, 1].set_ylabel('Share (%)')
    axes[1, 1].set_title(f'Regional Disparity Analysis\n(Disparity Index: {disparity_index:.3f})', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_05_regional_disparity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Regional disparity analysis complete")
    print(f"    Disparity index: {disparity_index:.3f} (0 = equal, 1 = max disparity)")
    
    return regional_data


def predictive_indicators(bio_df, demo_df, enrol_df):
    """Generate predictive indicators for system planning."""
    print("\n[PREDICT] Generating predictive indicators...")
    
    bio_cols = [c for c in bio_df.columns if 'bio_' in c]
    demo_cols = [c for c in demo_df.columns if 'demo_' in c]
    enrol_cols = [c for c in enrol_df.columns if 'age_' in c]
    
    # Weekly patterns
    bio_df['day_of_week'] = bio_df['date'].dt.dayofweek
    demo_df['day_of_week'] = demo_df['date'].dt.dayofweek
    enrol_df['day_of_week'] = enrol_df['date'].dt.dayofweek
    
    # Weekly pattern index
    bio_weekly = bio_df.groupby('day_of_week')[bio_cols].sum().sum(axis=1)
    bio_weekly_norm = (bio_weekly - bio_weekly.min()) / (bio_weekly.max() - bio_weekly.min())
    
    # State growth momentum
    bio_df['week'] = bio_df['date'].dt.isocalendar().week
    weekly_state = bio_df.groupby(['state', 'week'])[bio_cols].sum().sum(axis=1).unstack()
    growth = (weekly_state.iloc[:, -1] - weekly_state.iloc[:, 0]) / weekly_state.iloc[:, 0] * 100
    growth = growth.dropna().sort_values(ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Weekly pattern predictor
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in bio_weekly_norm.values]
    axes[0, 0].bar(days, bio_weekly_norm.values, color=colors)
    axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Day of Week')
    axes[0, 0].set_ylabel('Load Index (0-1)')
    axes[0, 0].set_title('Weekly Load Pattern Indicator\n(Green = High, Red = Low)', fontsize=14, fontweight='bold')
    
    # Growth momentum
    top_growth = growth.head(15)
    bottom_growth = growth.tail(5)
    combined_growth = pd.concat([top_growth, bottom_growth])
    colors = ['green' if v > 0 else 'red' for v in combined_growth.values]
    axes[0, 1].barh(range(len(combined_growth)), combined_growth.values, color=colors)
    axes[0, 1].set_yticks(range(len(combined_growth)))
    axes[0, 1].set_yticklabels(combined_growth.index)
    axes[0, 1].axvline(x=0, color='black', linestyle='-')
    axes[0, 1].set_xlabel('Growth Rate (%)')
    axes[0, 1].set_title('State Growth Momentum\n(Week-over-Week)', fontsize=14, fontweight='bold')
    
    # Service mix predictor
    total_bio = bio_df[bio_cols].sum().sum()
    total_demo = demo_df[demo_cols].sum().sum()
    total_enrol = enrol_df[enrol_cols].sum().sum()
    
    service_mix = pd.Series({
        'Updates\n(Bio+Demo)': (total_bio + total_demo) / (total_bio + total_demo + total_enrol) * 100,
        'New\nEnrollments': total_enrol / (total_bio + total_demo + total_enrol) * 100
    })
    
    axes[1, 0].pie(service_mix, labels=service_mix.index, autopct='%1.1f%%', 
                   colors=['#1E88E5', '#43A047'], explode=[0.02, 0.02],
                   startangle=90)
    axes[1, 0].set_title('Service Mix Indicator\n(Updates vs Fresh Enrollments)', fontsize=14, fontweight='bold')
    
    # Key metrics summary
    axes[1, 1].axis('off')
    
    # Calculate key indicators
    avg_daily = (bio_df.groupby('date')[bio_cols].sum().sum(axis=1).mean() +
                 demo_df.groupby('date')[demo_cols].sum().sum(axis=1).mean() +
                 enrol_df.groupby('date')[enrol_cols].sum().sum(axis=1).mean())
    
    peak_day_idx = bio_weekly.idxmax()
    low_day_idx = bio_weekly.idxmin()
    
    metrics_text = f"""
    ============================================================
    |         PREDICTIVE INDICATORS DASHBOARD                  |
    ============================================================
    |                                                          |
    |  [DATA] DEMAND INDICATORS                                |
    |  ----------------------                                  |
    |  * Average Daily Load: {avg_daily:,.0f}                         
    |  * Peak Day: {days[peak_day_idx]} (Index: {bio_weekly_norm.iloc[peak_day_idx]:.2f})                      
    |  * Low Day: {days[low_day_idx]} (Index: {bio_weekly_norm.iloc[low_day_idx]:.2f})                         
    |                                                          |
    |  [TREND] GROWTH INDICATORS                               |
    |  ----------------------                                  |
    |  * Fastest Growing: {growth.idxmax()[:20]}
    |  * Growth Rate: {growth.max():.1f}%                              
    |  * Declining States: {(growth < 0).sum()}                             
    |                                                          |
    |  [MIX] SERVICE MIX                                       |
    |  ----------------------                                  |
    |  * Update Ratio: {service_mix.iloc[0]:.1f}%                          
    |  * Enrollment Ratio: {service_mix.iloc[1]:.1f}%                      
    |                                                          |
    |  [ACTION] RECOMMENDATIONS                                |
    |  ----------------------                                  |
    |  * High-load days: Deploy extra capacity                 |
    |  * Growing states: Plan infrastructure expansion         |
    |  * Low days: Schedule maintenance windows                |
    |                                                          |
    ============================================================
    """
    
    axes[1, 1].text(0.5, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=11, fontfamily='monospace', verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_06_predictive_indicators.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Predictive indicators generated")
    
    return {
        'weekly_pattern': bio_weekly_norm,
        'growth_momentum': growth,
        'service_mix': service_mix
    }


def create_executive_summary(bio_df, demo_df, enrol_df):
    """Create an executive summary visualization."""
    print("\n[SUMMARY] Creating executive summary...")
    
    bio_cols = [c for c in bio_df.columns if 'bio_' in c]
    demo_cols = [c for c in demo_df.columns if 'demo_' in c]
    enrol_cols = [c for c in enrol_df.columns if 'age_' in c]
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # Title
    fig.suptitle('UIDAI AADHAAR DATA - EXECUTIVE SUMMARY\nAdvanced Analytics & Insights', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # 1. Key Numbers (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    total = len(bio_df) + len(demo_df) + len(enrol_df)
    numbers = f"""
    [KEY NUMBERS]
    ================
    
    Total Records: {total:,}
    
    Biometric: {len(bio_df):,}
    Demographic: {len(demo_df):,}
    Enrollment: {len(enrol_df):,}
    
    States: {bio_df['state'].nunique()}
    Districts: {bio_df['district'].nunique()}
    """
    ax1.text(0.1, 0.9, numbers, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    # 2. Service Distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1:3])
    totals = [
        bio_df[bio_cols].sum().sum(),
        demo_df[demo_cols].sum().sum(),
        enrol_df[enrol_cols].sum().sum()
    ]
    colors = ['#1E88E5', '#FFC107', '#43A047']
    wedges, texts, autotexts = ax2.pie(totals, labels=['Biometric\nUpdates', 'Demographic\nUpdates', 'New\nEnrollments'],
                                        autopct='%1.1f%%', colors=colors, explode=[0.02]*3,
                                        startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Service Distribution', fontsize=14, fontweight='bold')
    
    # 3. Top States (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    state_totals = bio_df.groupby('state')[bio_cols].sum().sum(axis=1).nlargest(5)
    ax3.barh(state_totals.index, state_totals.values, color='steelblue')
    ax3.set_title('Top 5 States\n(Biometric)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # 4. Daily Trend (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    daily_bio = bio_df.groupby('date')[bio_cols].sum().sum(axis=1)
    daily_demo = demo_df.groupby('date')[demo_cols].sum().sum(axis=1)
    daily_enrol = enrol_df.groupby('date')[enrol_cols].sum().sum(axis=1)
    
    ax4.plot(daily_bio.index, daily_bio.values, label='Biometric', alpha=0.8)
    ax4.plot(daily_demo.index, daily_demo.values, label='Demographic', alpha=0.8)
    ax4.plot(daily_enrol.index, daily_enrol.values, label='Enrollment', alpha=0.8)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Daily Volume')
    ax4.set_title('Daily Volume Trends', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Day of Week Pattern (lower left)
    ax5 = fig.add_subplot(gs[2, :2])
    bio_df['dow'] = bio_df['date'].dt.dayofweek
    dow_pattern = bio_df.groupby('dow')[bio_cols].sum().sum(axis=1)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors = ['#E53935' if i >= 5 else '#1E88E5' for i in range(7)]
    ax5.bar(days, dow_pattern.values, color=colors)
    ax5.set_xlabel('Day of Week')
    ax5.set_ylabel('Volume')
    ax5.set_title('Weekly Pattern (Red = Weekend)', fontsize=14, fontweight='bold')
    
    # 6. Age Distribution (lower right)
    ax6 = fig.add_subplot(gs[2, 2:])
    age_data = {
        'Youth 5-17': bio_df[bio_cols[0]].sum() if len(bio_cols) > 0 else 0,
        'Adults 17+': bio_df[bio_cols[1]].sum() if len(bio_cols) > 1 else 0
    }
    ax6.bar(age_data.keys(), age_data.values(), color=['#E53935', '#1E88E5'])
    ax6.set_xlabel('Age Group')
    ax6.set_ylabel('Volume')
    ax6.set_title('Age Group Distribution (Biometric)', fontsize=14, fontweight='bold')
    
    # 7. Key Insights (bottom)
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    insights = """
    ===========================================================================================================================================
    
    KEY INSIGHTS & RECOMMENDATIONS:
    
    1. [GEO] GEOGRAPHIC CONCENTRATION: A few states dominate Aadhaar activity. Recommend capacity expansion in top states.
    
    2. [TIME] TEMPORAL PATTERNS: Clear weekday preference, with lower weekend activity. Consider extended weekend hours in high-demand areas.
    
    3. [YOUTH] YOUTH ENGAGEMENT: Significant youth population activity suggests successful school enrollment drives.
    
    4. [MIX] SERVICE MIX: Balance between updates and new enrollments indicates mature system with both new and existing users.
    
    5. [GROWTH] GROWTH TRAJECTORY: Positive growth trends in several states indicate need for proactive infrastructure planning.
    
    ===========================================================================================================================================
    """
    
    ax7.text(0.5, 0.5, insights, transform=ax7.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'adv_07_executive_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  [OK] Executive summary created")


def main():
    """Main function for advanced analytics."""
    print("=" * 80)
    print("       UIDAI AADHAAR - ADVANCED ANALYTICS")
    print("       Predictive Indicators & Pattern Analysis")
    print("=" * 80)
    print()
    
    # Load data
    bio_df, demo_df, enrol_df = load_data()
    
    # Run advanced analyses
    state_clustering_analysis(bio_df, demo_df, enrol_df)
    correlation_analysis(bio_df, demo_df, enrol_df)
    demand_forecasting(bio_df)
    capacity_planning_analysis(bio_df, demo_df, enrol_df)
    regional_disparity_analysis(bio_df, demo_df, enrol_df)
    predictive_indicators(bio_df, demo_df, enrol_df)
    create_executive_summary(bio_df, demo_df, enrol_df)
    
    print("\n" + "=" * 80)
    print("       ADVANCED ANALYTICS COMPLETE")
    print("=" * 80)
    print("\n[FILES] Generated visualizations:")
    print("  * adv_01_state_clustering.png")
    print("  * adv_02_correlation_analysis.png")
    print("  * adv_03_demand_forecast.png")
    print("  * adv_04_capacity_planning.png")
    print("  * adv_05_regional_disparity.png")
    print("  * adv_06_predictive_indicators.png")
    print("  * adv_07_executive_summary.png")
    print()


if __name__ == "__main__":
    main()
