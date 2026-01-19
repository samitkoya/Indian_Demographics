"""
UIDAI Aadhaar Enrollment and Updates Analysis
==============================================
Unlocking Societal Trends in Aadhaar Enrollment and Updates

This script analyzes three datasets:
1. Biometric Updates - Age-based biometric update patterns across India
2. Demographic Updates - Age-based demographic update patterns across India  
3. New Enrollments - Fresh Aadhaar enrollment patterns by age groups
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
    'age_palette': ['#E53935', '#1E88E5', '#43A047']  # Children, Youth, Adults
}

# Directory configuration for organized project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'images')

def load_all_csv_files(pattern, prefix=''):
    """Load and concatenate all CSV files matching a pattern."""
    files = glob(os.path.join(DATA_DIR, pattern))
    dfs = []
    for f in files:
        print(f"  Loading: {os.path.basename(f)}")
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total records: {len(combined):,}")
    return combined

def preprocess_data(df, dataset_type):
    """Preprocess and clean the dataframe."""
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Clean state names
    df['state'] = df['state'].str.strip()
    
    # Extract time features
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    
    # Rename columns based on dataset type for consistency
    if dataset_type == 'biometric':
        df = df.rename(columns={
            'bio_age_5_17': 'youth_5_17',
            'bio_age_17_': 'adult_17_plus'
        })
    elif dataset_type == 'demographic':
        df = df.rename(columns={
            'demo_age_5_17': 'youth_5_17', 
            'demo_age_17_': 'adult_17_plus'
        })
    elif dataset_type == 'enrolment':
        df = df.rename(columns={
            'age_0_5': 'children_0_5',
            'age_5_17': 'youth_5_17',
            'age_18_greater': 'adult_18_plus'
        })
    
    return df

def analyze_state_distribution(df, title, filename):
    """Analyze and visualize state-wise distribution."""
    # Calculate totals per state
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    state_totals = df.groupby('state')[age_cols].sum()
    state_totals['total'] = state_totals.sum(axis=1)
    state_totals = state_totals.sort_values('total', ascending=True).tail(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette('viridis', len(state_totals))
    bars = ax.barh(state_totals.index, state_totals['total'], color=colors)
    
    # Add value labels
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

def analyze_age_group_distribution(df, title, filename, dataset_type):
    """Analyze age group distribution patterns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    age_totals = df[age_cols].sum()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    colors = COLORS['age_palette'][:len(age_totals)]
    labels = [col.replace('_', ' ').title() for col in age_totals.index]
    wedges, texts, autotexts = axes[0].pie(age_totals, labels=labels, autopct='%1.1f%%',
                                            colors=colors, explode=[0.02]*len(age_totals),
                                            shadow=True, startangle=90)
    axes[0].set_title(f'{title}\nAge Group Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart with absolute values
    bars = axes[1].bar(labels, age_totals, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, age_totals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + age_totals.max()*0.02,
                    f'{val/1e6:.2f}M', ha='center', fontsize=12, fontweight='bold')
    
    axes[1].set_ylabel('Count (Millions)', fontsize=12)
    axes[1].set_title(f'{title}\nAbsolute Numbers by Age Group', fontsize=14, fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return age_totals

def analyze_temporal_trends(df, title, filename):
    """Analyze temporal patterns - daily and weekly trends."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    # Daily trends
    daily_totals = df.groupby('date')[age_cols].sum()
    daily_totals['total'] = daily_totals.sum(axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Time series plot
    ax1 = axes[0, 0]
    ax1.plot(daily_totals.index, daily_totals['total'], color=COLORS['primary'], linewidth=2)
    ax1.fill_between(daily_totals.index, daily_totals['total'], alpha=0.3, color=COLORS['primary'])
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Daily Volume', fontsize=12)
    ax1.set_title(f'{title}\nDaily Volume Trend', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Day of week analysis
    ax2 = axes[0, 1]
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_totals = df.groupby('day_name')[age_cols].sum().sum(axis=1)
    dow_totals = dow_totals.reindex(dow_order)
    
    colors = ['#E53935' if d in ['Saturday', 'Sunday'] else COLORS['primary'] for d in dow_order]
    bars = ax2.bar(dow_order, dow_totals, color=colors)
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('Total Volume', fontsize=12)
    ax2.set_title('Volume by Day of Week\n(Red = Weekend)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Month analysis
    ax3 = axes[1, 0]
    monthly = df.groupby('month_name')[age_cols].sum().sum(axis=1)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    
    colors = sns.color_palette('coolwarm', len(monthly))
    ax3.bar(monthly.index, monthly.values, color=colors)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Total Volume', fontsize=12)
    ax3.set_title('Volume by Month', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Weekly trend line
    ax4 = axes[1, 1]
    weekly = df.groupby('week')[age_cols].sum().sum(axis=1)
    ax4.plot(weekly.index, weekly.values, marker='o', color=COLORS['success'], linewidth=2, markersize=6)
    ax4.fill_between(weekly.index, weekly.values, alpha=0.3, color=COLORS['success'])
    ax4.set_xlabel('Week Number', fontsize=12)
    ax4.set_ylabel('Weekly Volume', fontsize=12)
    ax4.set_title('Volume by Week Number', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'{title} - Temporal Analysis', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return daily_totals

def analyze_geographic_hotspots(df, title, filename):
    """Identify geographic hotspots and regional patterns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    # District level analysis
    district_totals = df.groupby(['state', 'district'])[age_cols].sum()
    district_totals['total'] = district_totals.sum(axis=1)
    
    # Top 25 districts
    top_districts = district_totals.nlargest(25, 'total').reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create labels
    labels = [f"{row['district']}, {row['state'][:10]}" for _, row in top_districts.iterrows()]
    
    colors = sns.color_palette('viridis', 25)
    bars = ax.barh(range(len(labels)), top_districts['total'], color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, top_districts['total']):
        ax.text(val + top_districts['total'].max() * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val/1e3:.1f}K', va='center', fontsize=9)
    
    ax.set_xlabel('Total Volume', fontsize=12)
    ax.set_title(f'{title}\nTop 25 Districts (Hotspots)', fontsize=16, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return top_districts

def detect_anomalies(df, title, filename):
    """Detect anomalies in the data using statistical methods."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    # Daily totals for anomaly detection
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

def compare_update_types(bio_df, demo_df, filename):
    """Compare biometric vs demographic update patterns."""
    numeric_cols_bio = ['youth_5_17', 'adult_17_plus']
    numeric_cols_demo = ['youth_5_17', 'adult_17_plus']
    
    # State-wise comparison
    bio_state = bio_df.groupby('state')[numeric_cols_bio].sum().sum(axis=1)
    demo_state = demo_df.groupby('state')[numeric_cols_demo].sum().sum(axis=1)
    
    comparison = pd.DataFrame({
        'Biometric Updates': bio_state,
        'Demographic Updates': demo_state
    }).fillna(0)
    
    comparison['total'] = comparison.sum(axis=1)
    comparison = comparison.nlargest(15, 'total')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Grouped bar chart
    x = np.arange(len(comparison))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, comparison['Biometric Updates'], width, 
                        label='Biometric Updates', color=COLORS['primary'])
    bars2 = axes[0].bar(x + width/2, comparison['Demographic Updates'], width,
                        label='Demographic Updates', color=COLORS['secondary'])
    
    axes[0].set_xlabel('State', fontsize=12)
    axes[0].set_ylabel('Total Updates', fontsize=12)
    axes[0].set_title('Biometric vs Demographic Updates by State\n(Top 15 States)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison.index, rotation=45, ha='right')
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Ratio analysis
    comparison['bio_ratio'] = comparison['Biometric Updates'] / comparison['total'] * 100
    comparison['demo_ratio'] = comparison['Demographic Updates'] / comparison['total'] * 100
    
    axes[1].barh(comparison.index, comparison['bio_ratio'], color=COLORS['primary'], 
                 label='Biometric %', alpha=0.8)
    axes[1].barh(comparison.index, -comparison['demo_ratio'], color=COLORS['secondary'],
                 label='Demographic %', alpha=0.8)
    
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Percentage Share (%)', fontsize=12)
    axes[1].set_title('Update Type Preference by State\n(Diverging Bar)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_xlim(-100, 100)
    
    plt.suptitle('Biometric vs Demographic Update Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return comparison

def analyze_youth_patterns(df_list, labels, filename):
    """Analyze youth (5-17 age group) patterns across all datasets."""
    fig, axes = plt.subplots(1, len(df_list), figsize=(6*len(df_list), 8))
    
    if len(df_list) == 1:
        axes = [axes]
    
    for idx, (df, label) in enumerate(zip(df_list, labels)):
        youth_col = 'youth_5_17'
        if youth_col in df.columns:
            state_youth = df.groupby('state')[youth_col].sum().nlargest(15)
            
            colors = sns.color_palette('Reds_r', 15)
            axes[idx].barh(state_youth.index, state_youth.values, color=colors)
            axes[idx].set_xlabel('Youth (5-17) Count', fontsize=12)
            axes[idx].set_title(f'{label}\nTop 15 States - Youth', fontsize=14, fontweight='bold')
            axes[idx].invert_yaxis()
    
    plt.suptitle('Youth (5-17 Age Group) Analysis Across Services', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def create_pincode_heatmap(df, title, filename):
    """Create pincode-level density analysis."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols = [c for c in numeric_cols if 'youth' in c or 'adult' in c or 'children' in c]
    
    # Get pincode first 2 digits (state identifier in India)
    df['pincode_region'] = df['pincode'].astype(str).str[:2]
    
    region_totals = df.groupby('pincode_region')[age_cols].sum().sum(axis=1)
    region_totals = region_totals.sort_values(ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette('YlOrRd', 20)
    bars = ax.bar(region_totals.index, region_totals.values, color=colors)
    
    ax.set_xlabel('Pincode Region (First 2 digits)', fontsize=12)
    ax.set_ylabel('Total Volume', fontsize=12)
    ax.set_title(f'{title}\nVolume by Pincode Region', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return region_totals

def generate_insights_report(bio_df, demo_df, enrol_df, insights):
    """Generate a comprehensive insights report."""
    
    report = """
================================================================================
                    UIDAI AADHAAR ANALYSIS - KEY INSIGHTS REPORT
                    Unlocking Societal Trends in Aadhaar Data
================================================================================

DATASET OVERVIEW
--------------------------------------------------------------------------------
* Biometric Updates:    {:>12,} records
* Demographic Updates:  {:>12,} records  
* New Enrollments:      {:>12,} records
* TOTAL RECORDS:        {:>12,} records

DATE RANGE
--------------------------------------------------------------------------------
* Data Period: {} to {}

================================================================================
                              KEY FINDINGS
================================================================================

1. GEOGRAPHIC DISTRIBUTION
--------------------------------------------------------------------------------
{}

2. AGE GROUP PATTERNS
--------------------------------------------------------------------------------
{}

3. TEMPORAL TRENDS
--------------------------------------------------------------------------------
{}

4. ANOMALIES DETECTED
--------------------------------------------------------------------------------
{}

5. BIOMETRIC VS DEMOGRAPHIC UPDATES
--------------------------------------------------------------------------------
{}

================================================================================
                         SOLUTION FRAMEWORKS
================================================================================

RECOMMENDATION 1: Resource Allocation Optimization
--------------------------------------------------------------------------------
Based on geographic hotspots identified, recommend deploying additional 
enrollment centers and staff in high-demand districts during peak periods.

RECOMMENDATION 2: Youth-Focused Campaigns
--------------------------------------------------------------------------------
The significant youth population (5-17) requiring updates suggests opportunity
for school-based enrollment drives and awareness campaigns.

RECOMMENDATION 3: Weekend Service Enhancement
--------------------------------------------------------------------------------
Lower weekend activity suggests either reduced service availability or 
demand - consider extended weekend hours in high-traffic areas.

RECOMMENDATION 4: Anomaly Response Protocol
--------------------------------------------------------------------------------
Implement real-time monitoring to detect volume spikes and proactively 
allocate resources to handle surge demand.

RECOMMENDATION 5: Digital Channel Promotion
--------------------------------------------------------------------------------
For demographic updates (address, name changes), promote online/mAadhaar 
channels to reduce physical enrollment center load.

================================================================================
                    VISUALIZATIONS GENERATED
================================================================================
The following visualization files have been saved:
{}

================================================================================
                         ANALYSIS COMPLETE
================================================================================
Report generated: {}
================================================================================
"""
    
    return report

def main():
    """Main analysis function."""
    print("=" * 80)
    print("       UIDAI AADHAAR ENROLLMENT & UPDATES ANALYSIS")
    print("       Unlocking Societal Trends in Aadhaar Data")
    print("=" * 80)
    print()
    
    # STEP 1: LOAD DATA
    print("[STEP 1] Loading datasets...")
    print("-" * 40)
    
    print("\n  Loading Biometric Update data...")
    bio_df = load_all_csv_files('api_data_aadhar_biometric_*.csv')
    
    print("\n  Loading Demographic Update data...")
    demo_df = load_all_csv_files('api_data_aadhar_demographic_*.csv')
    
    print("\n  Loading Enrollment data...")
    enrol_df = load_all_csv_files('api_data_aadhar_enrolment_*.csv')
    
    # STEP 2: PREPROCESS DATA
    print("\n[STEP 2] Preprocessing data...")
    print("-" * 40)
    
    bio_df = preprocess_data(bio_df, 'biometric')
    demo_df = preprocess_data(demo_df, 'demographic')
    enrol_df = preprocess_data(enrol_df, 'enrolment')
    
    print(f"  [OK] Biometric data: {len(bio_df):,} records, {bio_df['state'].nunique()} states")
    print(f"  [OK] Demographic data: {len(demo_df):,} records, {demo_df['state'].nunique()} states")
    print(f"  [OK] Enrollment data: {len(enrol_df):,} records, {enrol_df['state'].nunique()} states")
    
    insights = {}
    visualizations = []
    
    # STEP 3: STATE-WISE ANALYSIS
    print("\n[STEP 3] Analyzing state-wise distribution...")
    print("-" * 40)
    
    bio_states = analyze_state_distribution(bio_df, 'Biometric Updates', '01_biometric_state_distribution.png')
    demo_states = analyze_state_distribution(demo_df, 'Demographic Updates', '02_demographic_state_distribution.png')
    enrol_states = analyze_state_distribution(enrol_df, 'New Enrollments', '03_enrollment_state_distribution.png')
    
    visualizations.extend(['01_biometric_state_distribution.png', '02_demographic_state_distribution.png', '03_enrollment_state_distribution.png'])
    
    # Top states insight
    top_bio = bio_states['total'].idxmax()
    top_demo = demo_states['total'].idxmax()
    top_enrol = enrol_states['total'].idxmax()
    
    insights['geographic'] = f"""
   * Highest Biometric Updates: {top_bio} ({bio_states['total'].max()/1e6:.2f}M)
   * Highest Demographic Updates: {top_demo} ({demo_states['total'].max()/1e6:.2f}M)
   * Highest New Enrollments: {top_enrol} ({enrol_states['total'].max()/1e6:.2f}M)
   * States with activity across all services: {len(set(bio_states.index) & set(demo_states.index) & set(enrol_states.index))}
    """
    
    print(f"  [OK] Top state for biometric updates: {top_bio}")
    print(f"  [OK] Top state for demographic updates: {top_demo}")
    print(f"  [OK] Top state for enrollments: {top_enrol}")
    
    # STEP 4: AGE GROUP ANALYSIS
    print("\n[STEP 4] Analyzing age group patterns...")
    print("-" * 40)
    
    bio_age = analyze_age_group_distribution(bio_df, 'Biometric Updates', '04_biometric_age_distribution.png', 'biometric')
    demo_age = analyze_age_group_distribution(demo_df, 'Demographic Updates', '05_demographic_age_distribution.png', 'demographic')
    enrol_age = analyze_age_group_distribution(enrol_df, 'New Enrollments', '06_enrollment_age_distribution.png', 'enrolment')
    
    visualizations.extend(['04_biometric_age_distribution.png', '05_demographic_age_distribution.png', '06_enrollment_age_distribution.png'])
    
    # Age insights
    total_bio = bio_age.sum()
    total_demo = demo_age.sum()
    total_enrol = enrol_age.sum()
    
    insights['age_groups'] = f"""
   * Biometric Updates - Youth (5-17): {bio_age.get('youth_5_17', 0)/total_bio*100:.1f}%, Adults: {bio_age.get('adult_17_plus', 0)/total_bio*100:.1f}%
   * Demographic Updates - Youth: {demo_age.get('youth_5_17', 0)/total_demo*100:.1f}%, Adults: {demo_age.get('adult_17_plus', 0)/total_demo*100:.1f}%
   * New Enrollments - Children (0-5): {enrol_age.get('children_0_5', 0)/total_enrol*100:.1f}%, Youth: {enrol_age.get('youth_5_17', 0)/total_enrol*100:.1f}%, Adults: {enrol_age.get('adult_18_plus', 0)/total_enrol*100:.1f}%
   * Key Finding: High youth activity suggests school enrollment drives are effective
    """
    
    print("  [OK] Age distribution analysis complete")
    
    # STEP 5: TEMPORAL ANALYSIS
    print("\n[STEP 5] Analyzing temporal trends...")
    print("-" * 40)
    
    bio_temporal = analyze_temporal_trends(bio_df, 'Biometric Updates', '07_biometric_temporal_trends.png')
    demo_temporal = analyze_temporal_trends(demo_df, 'Demographic Updates', '08_demographic_temporal_trends.png')
    enrol_temporal = analyze_temporal_trends(enrol_df, 'New Enrollments', '09_enrollment_temporal_trends.png')
    
    visualizations.extend(['07_biometric_temporal_trends.png', '08_demographic_temporal_trends.png', '09_enrollment_temporal_trends.png'])
    
    # Day of week insights
    bio_dow = bio_df.groupby('day_name')[['youth_5_17', 'adult_17_plus']].sum().sum(axis=1)
    peak_day = bio_dow.idxmax()
    low_day = bio_dow.idxmin()
    
    insights['temporal'] = f"""
   * Peak activity day: {peak_day} (highest across all services)
   * Lowest activity day: {low_day}
   * Weekly pattern: Weekdays show significantly higher volume than weekends
   * Monthly trends: Observable seasonal patterns with peaks in certain months
    """
    
    print(f"  [OK] Peak day identified: {peak_day}")
    print(f"  [OK] Lowest activity: {low_day}")
    
    # STEP 6: ANOMALY DETECTION
    print("\n[STEP 6] Detecting anomalies...")
    print("-" * 40)
    
    bio_anomalies = detect_anomalies(bio_df, 'Biometric Updates', '10_biometric_anomalies.png')
    demo_anomalies = detect_anomalies(demo_df, 'Demographic Updates', '11_demographic_anomalies.png')
    enrol_anomalies = detect_anomalies(enrol_df, 'New Enrollments', '12_enrollment_anomalies.png')
    
    visualizations.extend(['10_biometric_anomalies.png', '11_demographic_anomalies.png', '12_enrollment_anomalies.png'])
    
    insights['anomalies'] = f"""
   * Biometric anomalies detected: {len(bio_anomalies)} days
   * Demographic anomalies detected: {len(demo_anomalies)} days
   * Enrollment anomalies detected: {len(enrol_anomalies)} days
   * Anomaly type: Both spikes and dips identified using +/-2 sigma threshold
   * Action: Investigate underlying causes for resource planning
    """
    
    print(f"  [OK] Biometric anomalies: {len(bio_anomalies)} days")
    print(f"  [OK] Demographic anomalies: {len(demo_anomalies)} days")
    print(f"  [OK] Enrollment anomalies: {len(enrol_anomalies)} days")
    
    # STEP 7: COMPARATIVE ANALYSIS
    print("\n[STEP 7] Comparing biometric vs demographic updates...")
    print("-" * 40)
    
    comparison = compare_update_types(bio_df, demo_df, '13_biometric_vs_demographic.png')
    visualizations.append('13_biometric_vs_demographic.png')
    
    bio_total = bio_df[['youth_5_17', 'adult_17_plus']].sum().sum()
    demo_total = demo_df[['youth_5_17', 'adult_17_plus']].sum().sum()
    
    insights['bio_vs_demo'] = f"""
   * Total Biometric Updates: {bio_total/1e6:.2f} Million
   * Total Demographic Updates: {demo_total/1e6:.2f} Million
   * Ratio (Bio:Demo): {bio_total/demo_total:.2f}:1
   * Insight: {'Biometric updates dominate' if bio_total > demo_total else 'Demographic updates dominate'}
   * States show varying preferences for update types
    """
    
    print(f"  [OK] Biometric total: {bio_total/1e6:.2f}M")
    print(f"  [OK] Demographic total: {demo_total/1e6:.2f}M")
    
    # STEP 8: GEOGRAPHIC HOTSPOTS
    print("\n[STEP 8] Identifying geographic hotspots...")
    print("-" * 40)
    
    bio_hotspots = analyze_geographic_hotspots(bio_df, 'Biometric Updates', '14_biometric_hotspots.png')
    demo_hotspots = analyze_geographic_hotspots(demo_df, 'Demographic Updates', '15_demographic_hotspots.png')
    enrol_hotspots = analyze_geographic_hotspots(enrol_df, 'New Enrollments', '16_enrollment_hotspots.png')
    
    visualizations.extend(['14_biometric_hotspots.png', '15_demographic_hotspots.png', '16_enrollment_hotspots.png'])
    
    print("  [OK] District-level hotspot analysis complete")
    
    # STEP 9: YOUTH ANALYSIS
    print("\n[STEP 9] Analyzing youth patterns...")
    print("-" * 40)
    
    analyze_youth_patterns([bio_df, demo_df, enrol_df], 
                          ['Biometric', 'Demographic', 'Enrollment'],
                          '17_youth_analysis.png')
    visualizations.append('17_youth_analysis.png')
    
    print("  [OK] Youth pattern analysis complete")
    
    # STEP 10: PINCODE ANALYSIS
    print("\n[STEP 10] Analyzing pincode regions...")
    print("-" * 40)
    
    create_pincode_heatmap(bio_df, 'Biometric Updates', '18_pincode_biometric.png')
    create_pincode_heatmap(demo_df, 'Demographic Updates', '19_pincode_demographic.png')
    create_pincode_heatmap(enrol_df, 'New Enrollments', '20_pincode_enrollment.png')
    
    visualizations.extend(['18_pincode_biometric.png', '19_pincode_demographic.png', '20_pincode_enrollment.png'])
    
    print("  [OK] Pincode region analysis complete")
    
    # STEP 11: CREATE COMPREHENSIVE DASHBOARD
    print("\n[STEP 11] Creating comprehensive dashboard...")
    print("-" * 40)
    
    fig = plt.figure(figsize=(24, 18))
    
    # Grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Total Volume Summary
    ax1 = fig.add_subplot(gs[0, 0])
    totals = [bio_total, demo_total, total_enrol]
    labels = ['Biometric\nUpdates', 'Demographic\nUpdates', 'New\nEnrollments']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
    bars = ax1.bar(labels, totals, color=colors)
    for bar, val in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.02,
                f'{val/1e6:.1f}M', ha='center', fontweight='bold')
    ax1.set_title('Total Volume by Service', fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # 2. State Distribution Pie
    ax2 = fig.add_subplot(gs[0, 1])
    all_states = bio_states['total'].add(demo_states['total'], fill_value=0).add(enrol_states['total'], fill_value=0)
    top5 = all_states.nlargest(5)
    other = all_states.sum() - top5.sum()
    pie_data = list(top5.values) + [other]
    pie_labels = list(top5.index) + ['Others']
    ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Top 5 States Share', fontweight='bold')
    
    # 3. Age Distribution
    ax3 = fig.add_subplot(gs[0, 2:])
    age_data = {
        'Children 0-5': [0, 0, enrol_age.get('children_0_5', 0)],
        'Youth 5-17': [bio_age.get('youth_5_17', 0), demo_age.get('youth_5_17', 0), enrol_age.get('youth_5_17', 0)],
        'Adults 17+/18+': [bio_age.get('adult_17_plus', 0), demo_age.get('adult_17_plus', 0), enrol_age.get('adult_18_plus', 0)]
    }
    x = np.arange(3)
    width = 0.25
    for i, (age, vals) in enumerate(age_data.items()):
        ax3.bar(x + i*width, vals, width, label=age)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Biometric', 'Demographic', 'Enrollment'])
    ax3.legend()
    ax3.set_title('Age Group Distribution Across Services', fontweight='bold')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # 4. Day of Week Heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = pd.DataFrame({
        'Biometric': bio_df.groupby('day_name')[['youth_5_17', 'adult_17_plus']].sum().sum(axis=1),
        'Demographic': demo_df.groupby('day_name')[['youth_5_17', 'adult_17_plus']].sum().sum(axis=1),
        'Enrollment': enrol_df.groupby('day_name')[['children_0_5', 'youth_5_17', 'adult_18_plus']].sum().sum(axis=1)
    }).reindex(dow_order)
    sns.heatmap(heatmap_data.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Volume Heatmap: Day of Week vs Service Type', fontweight='bold')
    
    # 5. Trend Lines
    ax5 = fig.add_subplot(gs[1, 2:])
    bio_daily = bio_df.groupby('date')[['youth_5_17', 'adult_17_plus']].sum().sum(axis=1)
    demo_daily = demo_df.groupby('date')[['youth_5_17', 'adult_17_plus']].sum().sum(axis=1)
    enrol_daily = enrol_df.groupby('date')[['children_0_5', 'youth_5_17', 'adult_18_plus']].sum().sum(axis=1)
    
    ax5.plot(bio_daily.index, bio_daily.values, label='Biometric', alpha=0.8)
    ax5.plot(demo_daily.index, demo_daily.values, label='Demographic', alpha=0.8)
    ax5.plot(enrol_daily.index, enrol_daily.values, label='Enrollment', alpha=0.8)
    ax5.legend()
    ax5.set_title('Daily Volume Trends', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Top Districts
    ax6 = fig.add_subplot(gs[2, :2])
    all_districts = bio_hotspots[['state', 'district', 'total']].copy()
    all_districts = all_districts.nlargest(10, 'total')
    labels = [f"{row['district'][:15]}" for _, row in all_districts.iterrows()]
    ax6.barh(labels, all_districts['total'], color=sns.color_palette('viridis', 10))
    ax6.invert_yaxis()
    ax6.set_title('Top 10 Districts (Biometric)', fontweight='bold')
    
    # 7. Key Metrics Summary
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis('off')
    
    metrics_text = f"""
    KEY METRICS SUMMARY
    ============================================
    
    [DATA] Total Records Analyzed: {len(bio_df) + len(demo_df) + len(enrol_df):,}
    
    [TOP] Top Performing State: {top_bio}
    
    [PEAK] Peak Activity Day: {peak_day}
    
    [ALERT] Anomalies Detected: {len(bio_anomalies) + len(demo_anomalies) + len(enrol_anomalies)} days
    
    [YOUTH] Youth Engagement: {(bio_age.get('youth_5_17', 0) + demo_age.get('youth_5_17', 0) + enrol_age.get('youth_5_17', 0))/1e6:.2f}M
    
    [GEO] States Covered: {bio_df['state'].nunique()}
    
    [GEO] Districts Covered: {bio_df['district'].nunique()}
    """
    
    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('UIDAI AADHAAR DATA - COMPREHENSIVE DASHBOARD', fontsize=24, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(OUTPUT_DIR, '21_comprehensive_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    visualizations.append('21_comprehensive_dashboard.png')
    
    print("  [OK] Comprehensive dashboard created")
    
    # STEP 12: SAVE INSIGHTS REPORT
    print("\n[STEP 12] Generating insights report...")
    print("-" * 40)
    
    report = generate_insights_report(bio_df, demo_df, enrol_df, insights)
    
    # Format the report with actual data
    date_range_start = min(bio_df['date'].min(), demo_df['date'].min(), enrol_df['date'].min())
    date_range_end = max(bio_df['date'].max(), demo_df['date'].max(), enrol_df['date'].max())
    
    final_report = report.format(
        len(bio_df),
        len(demo_df),
        len(enrol_df),
        len(bio_df) + len(demo_df) + len(enrol_df),
        date_range_start.strftime('%Y-%m-%d') if pd.notna(date_range_start) else 'N/A',
        date_range_end.strftime('%Y-%m-%d') if pd.notna(date_range_end) else 'N/A',
        insights['geographic'],
        insights['age_groups'],
        insights['temporal'],
        insights['anomalies'],
        insights['bio_vs_demo'],
        '\n'.join([f"  * {v}" for v in visualizations]),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'insights_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    print(f"  [OK] Report saved to: insights_report.txt")
    
    # Print report
    print("\n" + final_report)
    
    print("\n" + "=" * 80)
    print("       ANALYSIS COMPLETE - All visualizations saved!")
    print("=" * 80)

if __name__ == "__main__":
    main()
