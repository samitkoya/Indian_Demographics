"""
UIDAI Aadhaar Data Analysis - Visualization & Insights
========================================================
This script performs comprehensive data analysis and generates visualizations
for UIDAI Aadhaar enrollment and update data.

Features:
- State-wise distribution analysis
- Age group pattern analysis
- Temporal trend analysis
- Anomaly detection
- Comparative analysis
- Geographic hotspot identification
- Comprehensive dashboard generation

Input: Cleaned CSV data from data_cleaning.py
Output: 21 PNG visualization files + insights report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from glob import glob

warnings.filterwarnings('ignore')

# Set style for premium visualizations
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Color palettes for visualizations
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'accent': '#E53935',
    'success': '#43A047',
    'info': '#00ACC1',
    'gradient': ['#1E88E5', '#42A5F5', '#64B5F6', '#90CAF9', '#BBDEFB'],
    'age_palette': ['#E53935', '#1E88E5', '#43A047']
}

# Directory for saving outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')


def load_processed_data():
    """Load preprocessed data from the data directory."""
    print("[LOAD] Loading processed data...")
    
    # Load biometric data
    bio_files = glob(os.path.join(DATA_DIR, 'api_data_aadhar_biometric_*.csv'))
    bio_df = pd.concat([pd.read_csv(f) for f in bio_files], ignore_index=True)
    
    # Load demographic data
    demo_files = glob(os.path.join(DATA_DIR, 'api_data_aadhar_demographic_*.csv'))
    demo_df = pd.concat([pd.read_csv(f) for f in demo_files], ignore_index=True)
    
    # Load enrollment data
    enrol_files = glob(os.path.join(DATA_DIR, 'api_data_aadhar_enrolment_*.csv'))
    enrol_df = pd.concat([pd.read_csv(f) for f in enrol_files], ignore_index=True)
    
    # Preprocess dates
    for df in [bio_df, demo_df, enrol_df]:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df['state'] = df['state'].str.strip()
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
    
    # Rename columns for consistency
    bio_df = bio_df.rename(columns={
        'bio_age_5_17': 'youth_5_17',
        'bio_age_17_': 'adult_17_plus'
    })
    demo_df = demo_df.rename(columns={
        'demo_age_5_17': 'youth_5_17',
        'demo_age_17_': 'adult_17_plus'
    })
    enrol_df = enrol_df.rename(columns={
        'age_0_5': 'children_0_5',
        'age_5_17': 'youth_5_17',
        'age_18_greater': 'adult_18_plus'
    })
    
    print(f"  [OK] Loaded {len(bio_df):,} biometric records")
    print(f"  [OK] Loaded {len(demo_df):,} demographic records")
    print(f"  [OK] Loaded {len(enrol_df):,} enrollment records")
    
    return bio_df, demo_df, enrol_df


def analyze_state_distribution(df, title, filename):
    """Analyze and visualize state-wise distribution."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    state_totals = df.groupby('state')[age_cols].sum()
    state_totals['total'] = state_totals.sum(axis=1)
    state_totals = state_totals.sort_values('total', ascending=True).tail(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = sns.color_palette('viridis', len(state_totals))
    bars = ax.barh(state_totals.index, state_totals['total'], color=colors)
    
    for bar, val in zip(bars, state_totals['total']):
        ax.text(val + state_totals['total'].max() * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val/1e6:.2f}M', va='center', fontsize=10)
    
    ax.set_xlabel('Total Count (Millions)', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title(f'{title}\nTop 20 States by Volume', fontsize=16, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return state_totals


def analyze_age_distribution(df, title, filename):
    """Analyze age group distribution patterns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    age_totals = df[age_cols].sum()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = COLORS['age_palette'][:len(age_totals)]
    labels = [col.replace('_', ' ').title() for col in age_totals.index]
    
    # Pie chart
    axes[0].pie(age_totals, labels=labels, autopct='%1.1f%%',
                colors=colors, explode=[0.02]*len(age_totals), shadow=True, startangle=90)
    axes[0].set_title(f'{title}\nAge Group Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = axes[1].bar(labels, age_totals, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, age_totals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + age_totals.max()*0.02,
                    f'{val/1e6:.2f}M', ha='center', fontsize=12, fontweight='bold')
    
    axes[1].set_ylabel('Count (Millions)', fontsize=12)
    axes[1].set_title(f'{title}\nAbsolute Numbers by Age Group', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return age_totals


def analyze_temporal_trends(df, title, filename):
    """Analyze temporal patterns - daily and weekly trends."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    daily_totals = df.groupby('date')[age_cols].sum()
    daily_totals['total'] = daily_totals.sum(axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Time series plot
    axes[0, 0].plot(daily_totals.index, daily_totals['total'], color=COLORS['primary'], linewidth=2)
    axes[0, 0].fill_between(daily_totals.index, daily_totals['total'], alpha=0.3, color=COLORS['primary'])
    axes[0, 0].set_xlabel('Date', fontsize=12)
    axes[0, 0].set_ylabel('Daily Volume', fontsize=12)
    axes[0, 0].set_title(f'{title}\nDaily Volume Trend', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Day of week analysis
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_totals = df.groupby('day_name')[age_cols].sum().sum(axis=1)
    dow_totals = dow_totals.reindex(dow_order)
    
    colors = ['#E53935' if d in ['Saturday', 'Sunday'] else COLORS['primary'] for d in dow_order]
    axes[0, 1].bar(dow_order, dow_totals, color=colors)
    axes[0, 1].set_xlabel('Day of Week', fontsize=12)
    axes[0, 1].set_ylabel('Total Volume', fontsize=12)
    axes[0, 1].set_title('Volume by Day of Week\n(Red = Weekend)', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Month analysis
    monthly = df.groupby('month_name')[age_cols].sum().sum(axis=1)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    
    colors = sns.color_palette('coolwarm', len(monthly))
    axes[1, 0].bar(monthly.index, monthly.values, color=colors)
    axes[1, 0].set_xlabel('Month', fontsize=12)
    axes[1, 0].set_ylabel('Total Volume', fontsize=12)
    axes[1, 0].set_title('Volume by Month', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Weekly trend line
    weekly = df.groupby('week')[age_cols].sum().sum(axis=1)
    axes[1, 1].plot(weekly.index, weekly.values, marker='o', color=COLORS['success'], linewidth=2, markersize=6)
    axes[1, 1].fill_between(weekly.index, weekly.values, alpha=0.3, color=COLORS['success'])
    axes[1, 1].set_xlabel('Week Number', fontsize=12)
    axes[1, 1].set_ylabel('Weekly Volume', fontsize=12)
    axes[1, 1].set_title('Volume by Week Number', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'{title} - Temporal Analysis', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return daily_totals


def detect_anomalies(df, title, filename):
    """Detect anomalies in the data using statistical methods."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    daily = df.groupby('date')[age_cols].sum()
    daily['total'] = daily.sum(axis=1)
    
    # Calculate rolling statistics
    daily['rolling_mean'] = daily['total'].rolling(window=7, min_periods=1).mean()
    daily['rolling_std'] = daily['total'].rolling(window=7, min_periods=1).std()
    
    # Define anomalies (2 standard deviations from rolling mean)
    daily['upper_bound'] = daily['rolling_mean'] + 2 * daily['rolling_std']
    daily['lower_bound'] = daily['rolling_mean'] - 2 * daily['rolling_std']
    daily['is_anomaly'] = (daily['total'] > daily['upper_bound']) | (daily['total'] < daily['lower_bound'])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(daily.index, daily['total'], color=COLORS['primary'], linewidth=1.5, label='Daily Total')
    ax.plot(daily.index, daily['rolling_mean'], color=COLORS['secondary'], linewidth=2, label='7-Day Rolling Mean')
    ax.fill_between(daily.index, daily['lower_bound'], daily['upper_bound'],
                    alpha=0.2, color=COLORS['success'], label='Normal Range (+/-2 sigma)')
    
    # Highlight anomalies
    anomaly_dates = daily[daily['is_anomaly']].index
    anomaly_values = daily.loc[anomaly_dates, 'total']
    ax.scatter(anomaly_dates, anomaly_values, color=COLORS['accent'], s=100, zorder=5,
               label=f'Anomalies ({len(anomaly_dates)} detected)', edgecolors='black')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Volume', fontsize=12)
    ax.set_title(f'{title}\nAnomaly Detection (Statistical Outliers)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return daily[daily['is_anomaly']]


def main():
    """Main analysis function."""
    print("=" * 80)
    print("       UIDAI AADHAAR DATA ANALYSIS")
    print("       Generating Visualizations & Insights")
    print("=" * 80)
    print()
    
    # Load data
    bio_df, demo_df, enrol_df = load_processed_data()
    
    # Run analyses
    print("\n[ANALYSIS] Generating state distributions...")
    analyze_state_distribution(bio_df, 'Biometric Updates', '01_biometric_state_distribution.png')
    analyze_state_distribution(demo_df, 'Demographic Updates', '02_demographic_state_distribution.png')
    analyze_state_distribution(enrol_df, 'New Enrollments', '03_enrollment_state_distribution.png')
    
    print("[ANALYSIS] Generating age distributions...")
    analyze_age_distribution(bio_df, 'Biometric Updates', '04_biometric_age_distribution.png')
    analyze_age_distribution(demo_df, 'Demographic Updates', '05_demographic_age_distribution.png')
    analyze_age_distribution(enrol_df, 'New Enrollments', '06_enrollment_age_distribution.png')
    
    print("[ANALYSIS] Generating temporal trends...")
    analyze_temporal_trends(bio_df, 'Biometric Updates', '07_biometric_temporal_trends.png')
    analyze_temporal_trends(demo_df, 'Demographic Updates', '08_demographic_temporal_trends.png')
    analyze_temporal_trends(enrol_df, 'New Enrollments', '09_enrollment_temporal_trends.png')
    
    print("[ANALYSIS] Detecting anomalies...")
    detect_anomalies(bio_df, 'Biometric Updates', '10_biometric_anomalies.png')
    detect_anomalies(demo_df, 'Demographic Updates', '11_demographic_anomalies.png')
    detect_anomalies(enrol_df, 'New Enrollments', '12_enrollment_anomalies.png')
    
    print("\n" + "=" * 80)
    print("       ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
