"""Task 1C: Feature Engineering — Sliding window + creative features"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/figures'
PROCESSED_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/data/processed'

df = pd.read_csv(f'{PROCESSED_DIR}/daily_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['id', 'date']).reset_index(drop=True)

daily_features = [c for c in df.columns if c not in ['id', 'date']]
trend_vars = ['mood', 'circumplex.arousal', 'circumplex.valence']

# Fill NaN in daily data before feature extraction
app_cols = [c for c in daily_features if 'appCat' in c or c in ['call', 'sms', 'screen']]
core_cols = [c for c in daily_features if c not in app_cols]
for col in app_cols:
    df[col] = df[col].fillna(0)
for pid in df['id'].unique():
    pmask = df['id'] == pid
    df.loc[pmask, core_cols] = df.loc[pmask, core_cols].ffill().bfill()
df[daily_features] = df[daily_features].fillna(0)

print(f"Input shape: {df.shape}")
print(f"Daily feature columns ({len(daily_features)}): {daily_features}")


def ema(values, alpha=0.5):
    """Exponentially weighted mean — more weight on recent days."""
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def build_features(data, window_size, feat_cols):
    """Build sliding-window tabular features including creative additions."""
    rows = []
    for pid in sorted(data['id'].unique()):
        pdata = data[data['id'] == pid].reset_index(drop=True)

        for i in range(window_size, len(pdata)):
            window = pdata.iloc[i - window_size:i]
            target_row = pdata.iloc[i]

            if pd.isna(target_row['mood']):
                continue
            if window['mood'].isna().sum() > window_size // 2:
                continue

            row = {'id': pid, 'date': target_row['date']}

            # -------------------------------------------------------
            # STANDARD FEATURES: mean, std, last for every variable
            # -------------------------------------------------------
            for col in feat_cols:
                vals = window[col]
                row[f'{col}_mean'] = vals.mean()
                row[f'{col}_std'] = vals.std()
                row[f'{col}_last'] = vals.iloc[-1]

            # Mood range
            row['mood_min'] = window['mood'].min()
            row['mood_max'] = window['mood'].max()

            # Linear trend slopes for mood, arousal, valence
            for col in trend_vars:
                vals = window[col].dropna()
                if len(vals) >= 3:
                    slope, _, _, _, _ = stats.linregress(np.arange(len(vals)), vals.values)
                    row[f'{col}_trend'] = slope
                else:
                    row[f'{col}_trend'] = 0.0

            # Day-over-day mood change
            last_two = window['mood'].iloc[-2:]
            row['mood_change'] = (last_two.iloc[-1] - last_two.iloc[-2]) if last_two.notna().all() else 0.0

            # Calendar features
            row['day_of_week'] = target_row['date'].dayofweek
            row['is_weekend'] = 1 if target_row['date'].dayofweek >= 5 else 0

            # -------------------------------------------------------
            # CREATIVE FEATURE 1: Within-window z-score of last value
            # Captures "is today unusually high/low for THIS person THIS week?"
            # Addresses the large inter-individual variability we found in EDA.
            # -------------------------------------------------------
            for col in ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']:
                vals = window[col].values.astype(float)
                mu, sigma = vals.mean(), vals.std()
                last_val = vals[-1]
                row[f'{col}_zscore_last'] = (last_val - mu) / (sigma + 1e-8)

            # -------------------------------------------------------
            # CREATIVE FEATURE 2: Exponentially weighted moving average of mood
            # Gives more weight to recent days (alpha=0.5), capturing momentum
            # better than a simple mean. E.g. a mood crash yesterday counts more
            # than a good day 6 days ago.
            # -------------------------------------------------------
            mood_vals = window['mood'].values.astype(float)
            row['mood_ema'] = ema(mood_vals, alpha=0.5)

            # -------------------------------------------------------
            # CREATIVE FEATURE 3: Mood streak (persistence)
            # Count consecutive days the mood was above/below the window median.
            # A long bad-mood streak is a clinically meaningful warning signal.
            # -------------------------------------------------------
            mood_median = np.median(mood_vals)
            streak_low = 0
            streak_high = 0
            for v in reversed(mood_vals):
                if v < mood_median:
                    streak_low += 1
                else:
                    break
            for v in reversed(mood_vals):
                if v >= mood_median:
                    streak_high += 1
                else:
                    break
            row['mood_streak_low'] = streak_low
            row['mood_streak_high'] = streak_high

            # -------------------------------------------------------
            # CREATIVE FEATURE 4: Social engagement score
            # Aggregates communication behavior: calls + sms + communication apps
            # + social apps into a single composite score. Low social engagement
            # is a known risk factor for mood deterioration.
            # -------------------------------------------------------
            social_cols = ['call', 'sms', 'appCat.communication', 'appCat.social']
            social_vals = np.array([window[c].mean() for c in social_cols if c in feat_cols])
            # Normalize each component by its max in window then average
            social_maxes = np.array([window[c].max() for c in social_cols if c in feat_cols])
            social_norm = social_vals / (social_maxes + 1e-8)
            row['social_engagement_score'] = float(social_norm.mean())

            # -------------------------------------------------------
            # CREATIVE FEATURE 5: Mood acceleration
            # Second derivative approximation: is the mood trend speeding up
            # or slowing down? Split window in half and compare slopes.
            # -------------------------------------------------------
            if window_size >= 6:
                half = window_size // 2
                early_mood = mood_vals[:half]
                late_mood = mood_vals[half:]
                if len(early_mood) >= 2 and len(late_mood) >= 2:
                    slope_early, _, _, _, _ = stats.linregress(np.arange(len(early_mood)), early_mood)
                    slope_late, _, _, _, _ = stats.linregress(np.arange(len(late_mood)), late_mood)
                    row['mood_acceleration'] = slope_late - slope_early
                else:
                    row['mood_acceleration'] = 0.0
            else:
                row['mood_acceleration'] = 0.0

            # -------------------------------------------------------
            # CREATIVE FEATURE 6: Circumplex quadrant score
            # Arousal × valence product encodes the emotional quadrant:
            # (+,+) = excited/happy; (-,+) = calm/content;
            # (+,-) = stressed/angry; (-,-) = depressed/sad.
            # This interaction is theoretically grounded in the circumplex model.
            # -------------------------------------------------------
            aro_vals = window['circumplex.arousal'].values.astype(float)
            val_vals = window['circumplex.valence'].values.astype(float)
            row['circumplex_quadrant_mean'] = float((aro_vals * val_vals).mean())
            row['circumplex_quadrant_last'] = float(aro_vals[-1] * val_vals[-1])

            # -------------------------------------------------------
            # CREATIVE FEATURE 7: Behavioural passivity index
            # High screen time + low physical activity signals sedentary,
            # passive phone use — associated with worse mental health outcomes.
            # -------------------------------------------------------
            screen_mean = window['screen'].mean()
            activity_mean = window['activity'].mean()
            row['passivity_index'] = screen_mean / (activity_mean + 1e-3)

            # -------------------------------------------------------
            # CREATIVE FEATURE 8: Mood coefficient of variation (normalised volatility)
            # CV = std / |mean|. More meaningful than raw std when comparing
            # patients with different mood baselines.
            # -------------------------------------------------------
            row['mood_cv'] = float(np.std(mood_vals) / (abs(np.mean(mood_vals)) + 1e-8))

            row['mood_target'] = target_row['mood']
            rows.append(row)

    dataset = pd.DataFrame(rows)
    feat_names = [c for c in dataset.columns if c not in ['id', 'date', 'mood_target']]

    # Fill 0 for app/call/sms/screen features (missing = no usage)
    zero_fill = ['appCat.', 'call_', 'sms_', 'screen_']
    for col in feat_names:
        if any(col.startswith(p) for p in zero_fill):
            dataset[col] = dataset[col].fillna(0)

    # Drop rows missing core features
    core = [c for c in feat_names if any(c.startswith(p) for p in ['mood_', 'circumplex.', 'activity_'])]
    dataset = dataset.dropna(subset=core)
    dataset[feat_names] = dataset[feat_names].fillna(0)

    return dataset, feat_names


# =============================================
# 1. Window size comparison
# =============================================
print("\n=== WINDOW SIZE COMPARISON ===")

window_results = {}
for ws in [3, 5, 7, 14]:
    ds, feat_names_ws = build_features(df, ws, daily_features)
    tr, te = [], []
    for pid in sorted(ds['id'].unique()):
        pdata = ds[ds['id'] == pid].sort_values('date')
        split = int(len(pdata) * 0.8)
        tr.append(pdata.iloc[:split])
        te.append(pdata.iloc[split:])
    tr = pd.concat(tr)
    te = pd.concat(te)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                               random_state=42, n_jobs=-1)
    rf.fit(tr[feat_names_ws].values, tr['mood_target'].values)
    pred = rf.predict(te[feat_names_ws].values)

    rmse = np.sqrt(mean_squared_error(te['mood_target'], pred))
    mae = mean_absolute_error(te['mood_target'], pred)
    r2 = r2_score(te['mood_target'], pred)
    window_results[ws] = {'n_samples': len(ds), 'n_features': len(feat_names_ws),
                          'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"  Window={ws}: n={len(ds)}, features={len(feat_names_ws)}, "
          f"RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
windows = list(window_results.keys())
for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2']):
    vals = [window_results[w][metric] for w in windows]
    ax.plot(windows, vals, 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Window Size (days)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Window Size')
    ax.set_xticks(windows)
plt.suptitle('Effect of Window Size on RF Regressor Performance', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/window_size_comparison.png', dpi=150)
plt.close()

# =============================================
# 2. Build final dataset with window=7
# =============================================
print("\n=== BUILDING FINAL DATASET (window=7) ===")
dataset, feature_names = build_features(df, 7, daily_features)
print(f"Shape: {dataset.shape}, Features: {len(feature_names)}")

# List all feature groups
standard_feats = [f for f in feature_names if not any(f.startswith(x) for x in
    ['mood_zscore','circumplex.arousal_zscore','circumplex.valence_zscore',
     'activity_zscore','mood_ema','mood_streak','social_eng','mood_acc',
     'circumplex_quad','passivity','mood_cv'])]
creative_feats = [f for f in feature_names if f not in standard_feats]
print(f"\nStandard features: {len(standard_feats)}")
print(f"Creative features ({len(creative_feats)}): {creative_feats}")

# =============================================
# 3. Classification targets (tertile-based)
# =============================================
print("\n=== CLASSIFICATION TARGETS ===")

dataset['mood_class_original'] = dataset['mood_target'].apply(
    lambda x: 'Low' if x <= 4 else ('Medium' if x <= 6 else 'High'))
print("Original 3-class distribution (severely imbalanced):")
print(dataset['mood_class_original'].value_counts(normalize=True).round(3))

tertiles = dataset['mood_target'].quantile([1/3, 2/3])
t1, t2 = tertiles.iloc[0], tertiles.iloc[1]
print(f"\nTertile boundaries: {t1:.2f}, {t2:.2f}")
dataset['mood_class'] = dataset['mood_target'].apply(
    lambda x: 'Low' if x <= t1 else ('Medium' if x <= t2 else 'High'))
print("Tertile-based 3-class distribution:")
print(dataset['mood_class'].value_counts())
print(dataset['mood_class'].value_counts(normalize=True).round(3))

# =============================================
# 4. Mutual Information analysis
# =============================================
print("\n=== MUTUAL INFORMATION ANALYSIS ===")

train_frames = []
for pid in sorted(dataset['id'].unique()):
    pdata = dataset[dataset['id'] == pid].sort_values('date')
    split = int(len(pdata) * 0.8)
    train_frames.append(pdata.iloc[:split])
train_data = pd.concat(train_frames)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[feature_names].values)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(['Low', 'Medium', 'High'])
y_train_cls = le.transform(train_data['mood_class'].values)
y_train_reg = train_data['mood_target'].values

mi_cls = mutual_info_classif(X_train_scaled, y_train_cls, random_state=42)
mi_reg = mutual_info_regression(X_train_scaled, y_train_reg, random_state=42)

mi_df = pd.DataFrame({
    'feature': feature_names,
    'MI_classification': mi_cls,
    'MI_regression': mi_reg
}).sort_values('MI_regression', ascending=False)

print("Top 20 features by MI (regression):")
print(mi_df.head(20).to_string(index=False))

# Highlight where creative features rank
creative_mi = mi_df[mi_df['feature'].isin(creative_feats)].sort_values('MI_regression', ascending=False)
print("\nCreative feature MI rankings:")
print(creative_mi[['feature', 'MI_regression', 'MI_classification']].to_string(index=False))

# Plot MI
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
top20_cls = mi_df.sort_values('MI_classification', ascending=False).head(20)
top20_reg = mi_df.sort_values('MI_regression', ascending=False).head(20)

# Color bars: red for creative, blue for standard
def bar_colors(features, creative_set):
    return ['tomato' if f in creative_set else 'steelblue' for f in features]

axes[0].barh(range(20), top20_cls['MI_classification'].values,
             color=bar_colors(top20_cls['feature'].values, set(creative_feats)))
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top20_cls['feature'].values, fontsize=8)
axes[0].set_xlabel('Mutual Information')
axes[0].set_title('Top 20 Features — Classification\n(red = creative features)')
axes[0].invert_yaxis()

axes[1].barh(range(20), top20_reg['MI_regression'].values,
             color=bar_colors(top20_reg['feature'].values, set(creative_feats)))
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top20_reg['feature'].values, fontsize=8)
axes[1].set_xlabel('Mutual Information')
axes[1].set_title('Top 20 Features — Regression\n(red = creative features)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/mutual_information_features.png', dpi=150)
plt.close()
print("Saved: mutual_information_features.png")

# =============================================
# 5. RF test to show creative features improve performance
# =============================================
print("\n=== ABLATION: STANDARD vs STANDARD+CREATIVE FEATURES ===")
tr_list, te_list = [], []
for pid in sorted(dataset['id'].unique()):
    pdata = dataset[dataset['id'] == pid].sort_values('date')
    split = int(len(pdata) * 0.8)
    tr_list.append(pdata.iloc[:split])
    te_list.append(pdata.iloc[split:])
tr_all = pd.concat(tr_list)
te_all = pd.concat(te_list)

for label, feats in [('Standard only', standard_feats), ('Standard + Creative', feature_names)]:
    rf = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=10, random_state=42, n_jobs=-1)
    rf.fit(tr_all[feats].values, tr_all['mood_target'].values)
    pred = rf.predict(te_all[feats].values)
    rmse = np.sqrt(mean_squared_error(te_all['mood_target'], pred))
    mae = mean_absolute_error(te_all['mood_target'], pred)
    r2 = r2_score(te_all['mood_target'], pred)
    print(f"  {label}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# =============================================
# 6. Save final dataset
# =============================================
dataset.to_csv(f'{PROCESSED_DIR}/final_dataset.csv', index=False)
print(f"\nSaved: final_dataset.csv  shape={dataset.shape}")
print(f"Total features: {len(feature_names)}  (standard: {len(standard_feats)}, creative: {len(creative_feats)})")
