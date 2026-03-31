"""Task 1B: Data Cleaning — Outlier removal and Imputation"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/figures'
PROCESSED_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/data/processed'

# Load raw data
df = pd.read_csv('/Users/valentijnheijnsbroek/Desktop/Data Mining/data/raw/dataset_mood_smartphone.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date

print(f"Original rows: {len(df):,}")

# =============================================
# OUTLIER REMOVAL
# =============================================
print("\n=== OUTLIER REMOVAL ===")

removal_counts = {}

# 1. Domain range filtering
domain_ranges = {
    'mood': (1, 10),
    'circumplex.arousal': (-2, 2),
    'circumplex.valence': (-2, 2),
    'activity': (0, 1),
    'call': (0, 1),
    'sms': (0, 1),
}

for var, (lo, hi) in domain_ranges.items():
    mask = (df['variable'] == var)
    before = mask.sum()
    invalid = mask & ((df['value'] < lo) | (df['value'] > hi))
    removal_counts[var] = invalid.sum()
    df = df[~invalid]
    print(f"  {var}: removed {removal_counts[var]} values outside [{lo}, {hi}]")

# 2. Non-negative + IQR capping for screen and appCat.*
iqr_vars = [v for v in df['variable'].unique() if v.startswith('appCat.') or v == 'screen']

for var in iqr_vars:
    mask = df['variable'] == var
    vals = df.loc[mask, 'value']

    # Remove negative values
    neg_mask = mask & (df['value'] < 0)
    n_neg = neg_mask.sum()
    df = df[~neg_mask]

    # Recompute after removing negatives
    mask = df['variable'] == var
    vals = df.loc[mask, 'value']

    Q1 = vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR

    cap_mask = mask & (df['value'] > upper)
    n_capped = cap_mask.sum()
    df.loc[cap_mask, 'value'] = upper

    removal_counts[var] = f"{n_neg} removed (neg), {n_capped} capped at {upper:.1f}"
    print(f"  {var}: {n_neg} negative removed, {n_capped} capped at {upper:.1f}")

print(f"\nRows after outlier removal: {len(df):,}")

# =============================================
# AGGREGATE TO DAILY LEVEL
# =============================================
print("\n=== AGGREGATING TO DAILY LEVEL ===")

mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
sum_vars = [v for v in df['variable'].unique() if v not in mean_vars]

daily_mean = df[df['variable'].isin(mean_vars)].groupby(['id', 'date', 'variable'])['value'].mean().reset_index()
daily_sum = df[df['variable'].isin(sum_vars)].groupby(['id', 'date', 'variable'])['value'].sum().reset_index()
daily_all = pd.concat([daily_mean, daily_sum])

# Pivot to wide format
daily_wide = daily_all.pivot_table(index=['id', 'date'], columns='variable', values='value').reset_index()
daily_wide.columns.name = None
daily_wide['date'] = pd.to_datetime(daily_wide['date'])

# Fill in all dates per patient (create complete date index)
all_frames = []
for pid in daily_wide['id'].unique():
    pdata = daily_wide[daily_wide['id'] == pid].set_index('date')
    date_range = pd.date_range(pdata.index.min(), pdata.index.max(), freq='D')
    pdata = pdata.reindex(date_range)
    pdata['id'] = pid
    pdata.index.name = 'date'
    all_frames.append(pdata.reset_index())

daily_complete = pd.concat(all_frames, ignore_index=True)
print(f"Daily wide shape (with all dates): {daily_complete.shape}")
print(f"Missing values per variable:")
feature_cols = [c for c in daily_complete.columns if c not in ['id', 'date']]
print(daily_complete[feature_cols].isnull().sum())

# =============================================
# IMPUTATION — TWO METHODS
# =============================================
print("\n=== IMPUTATION ===")

# Identify prolonged gaps (>7 consecutive missing days) per patient
def find_long_gaps(series, threshold=7):
    """Return boolean mask of positions that are in gaps > threshold days."""
    is_missing = series.isnull()
    gap_mask = pd.Series(False, index=series.index)
    gap_start = None
    gap_len = 0

    for i in range(len(is_missing)):
        if is_missing.iloc[i]:
            if gap_start is None:
                gap_start = i
            gap_len += 1
        else:
            if gap_len > threshold and gap_start is not None:
                gap_mask.iloc[gap_start:gap_start + gap_len] = True
            gap_start = None
            gap_len = 0
    # Handle trailing gap
    if gap_len > threshold and gap_start is not None:
        gap_mask.iloc[gap_start:gap_start + gap_len] = True

    return gap_mask


# METHOD 1: Linear Interpolation
daily_interp = daily_complete.copy()
for pid in daily_interp['id'].unique():
    pmask = daily_interp['id'] == pid
    for col in feature_cols:
        series = daily_interp.loc[pmask, col].copy()
        long_gap = find_long_gaps(series, threshold=7)
        # Interpolate
        series_filled = series.interpolate(method='linear', limit_direction='both')
        # Restore NaN for prolonged gaps
        series_filled[long_gap] = np.nan
        daily_interp.loc[pmask, col] = series_filled

# METHOD 2: LOCF
daily_locf = daily_complete.copy()
for pid in daily_locf['id'].unique():
    pmask = daily_locf['id'] == pid
    for col in feature_cols:
        series = daily_locf.loc[pmask, col].copy()
        long_gap = find_long_gaps(series, threshold=7)
        # LOCF then backfill
        series_filled = series.ffill().bfill()
        # Restore NaN for prolonged gaps
        series_filled[long_gap] = np.nan
        daily_locf.loc[pmask, col] = series_filled

# =============================================
# COMPARISON
# =============================================
print("\n--- Missing values BEFORE imputation ---")
print(daily_complete[feature_cols].isnull().sum())

print("\n--- Missing values AFTER linear interpolation ---")
print(daily_interp[feature_cols].isnull().sum())

print("\n--- Missing values AFTER LOCF ---")
print(daily_locf[feature_cols].isnull().sum())

# Stats comparison
print("\n--- Summary stats: Original ---")
print(daily_complete[feature_cols].describe().loc[['mean', 'std']].round(3))
print("\n--- Summary stats: Linear Interpolation ---")
print(daily_interp[feature_cols].describe().loc[['mean', 'std']].round(3))
print("\n--- Summary stats: LOCF ---")
print(daily_locf[feature_cols].describe().loc[['mean', 'std']].round(3))

# Plot comparison for one patient (mood)
example_pid = 'AS14.01'  # Has a 21-day gap
fig, ax = plt.subplots(figsize=(12, 5))

pmask_orig = daily_complete['id'] == example_pid
pmask_interp = daily_interp['id'] == example_pid
pmask_locf = daily_locf['id'] == example_pid

ax.plot(daily_complete.loc[pmask_orig, 'date'], daily_complete.loc[pmask_orig, 'mood'],
        'ko-', markersize=4, label='Original', linewidth=1, alpha=0.7)
ax.plot(daily_interp.loc[pmask_interp, 'date'], daily_interp.loc[pmask_interp, 'mood'],
        'b.--', markersize=3, label='Linear Interpolation', linewidth=0.8, alpha=0.7)
ax.plot(daily_locf.loc[pmask_locf, 'date'], daily_locf.loc[pmask_locf, 'mood'],
        'r.--', markersize=3, label='LOCF', linewidth=0.8, alpha=0.7)

ax.set_xlabel('Date')
ax.set_ylabel('Mood')
ax.set_title(f'Imputation Comparison — {example_pid} (has 21-day gap)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/imputation_comparison.png', dpi=150)
plt.close()
print("\nSaved: imputation_comparison.png")

# =============================================
# SELECT LINEAR INTERPOLATION & SAVE
# =============================================
print("\n=== SELECTING LINEAR INTERPOLATION ===")
print("Rationale: Linear interpolation better preserves temporal trends (Lepot et al., 2017)")
print("LOCF can create artificial plateaus that don't reflect gradual changes in mood/behavior")

daily_interp.to_csv(f'{PROCESSED_DIR}/daily_cleaned.csv', index=False)
print(f"Saved: data/processed/daily_cleaned.csv — shape {daily_interp.shape}")
print(f"Remaining NaN (from prolonged >7-day gaps): {daily_interp[feature_cols].isnull().sum().sum()}")
