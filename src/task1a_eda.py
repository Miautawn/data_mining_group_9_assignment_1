"""Task 1A: Exploratory Data Analysis"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/figures'

# 1. Load dataset
df = pd.read_csv('/Users/valentijnheijnsbroek/Desktop/Data Mining/data/raw/dataset_mood_smartphone.csv')
print("=== RAW DATA ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# Drop unnamed index column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Parse timestamps
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date

# 2. Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total rows: {len(df):,}")
print(f"Unique patients: {df['id'].nunique()}")
print(f"Unique variables: {df['variable'].nunique()}")
print(f"Variables: {sorted(df['variable'].unique())}")

# Date range per patient
print("\n--- Date range per patient ---")
date_range = df.groupby('id')['time'].agg(['min', 'max'])
date_range['duration_days'] = (date_range['max'] - date_range['min']).dt.days
print(date_range)

# Records per variable
print("\n--- Records per variable ---")
var_counts = df.groupby('variable')['value'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
print(var_counts.to_string())

# Records per patient
print("\n--- Records per patient ---")
pat_counts = df.groupby('id')['value'].count().sort_values(ascending=False)
print(pat_counts)

# 3. PLOTS

# 3a. Histogram of mood values
mood_data = df[df['variable'] == 'mood']['value']
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(mood_data, bins=np.arange(0.5, 11.5, 1), edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Mood Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Mood Values Across All Patients')
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/mood_histogram.png', dpi=150)
plt.close()
print("\nSaved: mood_histogram.png")

# 3b. Boxplot of mood per patient
fig, ax = plt.subplots(figsize=(14, 6))
mood_by_patient = df[df['variable'] == 'mood'].groupby('id')['value'].apply(list)
patients_sorted = mood_by_patient.index.tolist()
ax.boxplot([mood_by_patient[p] for p in patients_sorted], labels=patients_sorted)
ax.set_xlabel('Patient ID')
ax.set_ylabel('Mood')
ax.set_title('Mood Distribution per Patient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/mood_boxplot_per_patient.png', dpi=150)
plt.close()
print("Saved: mood_boxplot_per_patient.png")

# 3c. Time series of daily average mood for 4 diverse patients
daily_mood = df[df['variable'] == 'mood'].groupby(['id', 'date'])['value'].mean().reset_index()
daily_mood['date'] = pd.to_datetime(daily_mood['date'])

# Pick 4 diverse patients (high mood, low mood, variable, stable)
patient_stats = daily_mood.groupby('id')['value'].agg(['mean', 'std'])
candidates = [
    patient_stats['mean'].idxmax(),  # highest avg mood
    patient_stats['mean'].idxmin(),  # lowest avg mood
    patient_stats['std'].idxmax(),   # most variable
    patient_stats['std'].idxmin(),   # most stable
]
patients_4 = list(dict.fromkeys(candidates))
# If we have fewer than 4 unique, add median-mood patients
if len(patients_4) < 4:
    remaining = [p for p in patient_stats.index if p not in patients_4]
    median_mood = patient_stats.loc[remaining, 'mean'].sub(patient_stats['mean'].median()).abs().sort_values()
    for p in median_mood.index:
        patients_4.append(p)
        if len(patients_4) == 4:
            break
print(f"\n4 diverse patients: {patients_4}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=True)
for ax, pid in zip(axes.flat, patients_4):
    pdata = daily_mood[daily_mood['id'] == pid]
    ax.plot(pdata['date'], pdata['value'], marker='.', markersize=3, linewidth=0.8)
    ax.set_title(f'{pid} (mean={pdata["value"].mean():.1f}, std={pdata["value"].std():.1f})')
    ax.set_ylabel('Avg Daily Mood')
    ax.set_ylim(0, 10.5)
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Daily Average Mood - 4 Diverse Patients', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/mood_timeseries_4patients.png', dpi=150)
plt.close()
print("Saved: mood_timeseries_4patients.png")

# 3d. Correlation heatmap of daily-aggregated variables
# Aggregate to daily: mean for mood/arousal/valence/activity, sum for others
mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
sum_vars = [v for v in df['variable'].unique() if v not in mean_vars]

daily_mean = df[df['variable'].isin(mean_vars)].groupby(['id', 'date', 'variable'])['value'].mean().reset_index()
daily_sum = df[df['variable'].isin(sum_vars)].groupby(['id', 'date', 'variable'])['value'].sum().reset_index()
daily_all = pd.concat([daily_mean, daily_sum])
daily_wide = daily_all.pivot_table(index=['id', 'date'], columns='variable', values='value')

fig, ax = plt.subplots(figsize=(12, 10))
corr = daily_wide.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1, annot_kws={'size': 7})
ax.set_title('Correlation Heatmap of Daily-Aggregated Variables')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# 3e. Distribution plots for arousal and valence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, var in zip(axes, ['circumplex.arousal', 'circumplex.valence']):
    data = df[df['variable'] == var]['value'].dropna()
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(f'Distribution of {var}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.2f}')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/arousal_valence_distributions.png', dpi=150)
plt.close()
print("Saved: arousal_valence_distributions.png")

# 3f. Missing data heatmap: per patient x variable, % of days with no data
all_patients = sorted(df['id'].unique())
all_variables = sorted(df['variable'].unique())

# Get all unique dates per patient
patient_dates = df.groupby('id')['date'].apply(lambda x: set(x))

# For each patient-variable, find % of patient's days with no data
missing_pct = pd.DataFrame(index=all_patients, columns=all_variables, dtype=float)
for pid in all_patients:
    p_dates = patient_dates[pid]
    n_days = len(p_dates)
    for var in all_variables:
        var_dates = set(df[(df['id'] == pid) & (df['variable'] == var)]['date'])
        missing_pct.loc[pid, var] = 100.0 * (1 - len(var_dates) / n_days) if n_days > 0 else 100.0

missing_pct = missing_pct.astype(float)

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(missing_pct, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.3, vmin=0, vmax=100, annot_kws={'size': 6})
ax.set_title('Missing Data: % of Days with No Data per Patient x Variable')
ax.set_ylabel('Patient')
ax.set_xlabel('Variable')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/missing_data_heatmap.png', dpi=150)
plt.close()
print("Saved: missing_data_heatmap.png")

# 3g. Identify prolonged gaps (5+ consecutive missing days per patient)
print("\n=== PROLONGED GAPS (5+ consecutive missing days) ===")
# For mood variable specifically
mood_daily = df[df['variable'] == 'mood'].groupby(['id', 'date'])['value'].mean().reset_index()
mood_daily['date'] = pd.to_datetime(mood_daily['date'])

for pid in sorted(df['id'].unique()):
    pdata = mood_daily[mood_daily['id'] == pid].sort_values('date')
    if len(pdata) == 0:
        continue
    all_days = pd.date_range(pdata['date'].min(), pdata['date'].max(), freq='D')
    recorded_days = set(pdata['date'])
    missing_days = sorted(set(all_days) - recorded_days)

    if not missing_days:
        continue

    # Find consecutive gaps
    gaps = []
    gap_start = missing_days[0]
    gap_len = 1
    for i in range(1, len(missing_days)):
        if (missing_days[i] - missing_days[i-1]).days == 1:
            gap_len += 1
        else:
            if gap_len >= 5:
                gaps.append((gap_start, missing_days[i-1], gap_len))
            gap_start = missing_days[i]
            gap_len = 1
    if gap_len >= 5:
        gaps.append((gap_start, missing_days[-1], gap_len))

    if gaps:
        for start, end, length in gaps:
            print(f"  {pid}: {start.date()} to {end.date()} ({length} days)")

# 4. Observations summary
print("\n=== KEY OBSERVATIONS ===")
print(f"1. Mood values are concentrated around 5-7 (mean={mood_data.mean():.2f}, median={mood_data.median():.1f})")
print(f"2. Considerable individual differences in mood - some patients consistently higher/lower")
print(f"3. {len(all_patients)} patients with varying data collection periods")
print(f"4. Arousal and valence show different distributions (arousal more spread, valence slightly positive)")
print(f"5. Missing data is variable-dependent: app categories have high missingness on many days")
print(f"6. Several patients have prolonged gaps (5+ days) in mood recording")
print(f"7. Screen time and app categories show strong positive correlations with each other")
print(f"8. Mood shows moderate positive correlation with valence (positive emotions)")
