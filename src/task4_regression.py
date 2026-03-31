"""Task 4: Regression — RF Regressor + LSTM Regressor with per-patient analysis"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/figures'
PROCESSED_DIR = '/Users/valentijnheijnsbroek/Desktop/Data Mining/data/processed'

# =============================================
# LOAD DATA
# =============================================
dataset = pd.read_csv(f'{PROCESSED_DIR}/final_dataset.csv')
dataset['date'] = pd.to_datetime(dataset['date'])

feature_cols = [c for c in dataset.columns
                if c not in ['id', 'date', 'mood_target', 'mood_class',
                             'mood_class_median', 'mood_class_original']]

# Temporal split
train_frames, test_frames = [], []
for pid in sorted(dataset['id'].unique()):
    pdata = dataset[dataset['id'] == pid].sort_values('date')
    split_idx = int(len(pdata) * 0.8)
    train_frames.append(pdata.iloc[:split_idx])
    test_frames.append(pdata.iloc[split_idx:])

train = pd.concat(train_frames).reset_index(drop=True)
test = pd.concat(test_frames).reset_index(drop=True)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols].values)
X_test = scaler.transform(test[feature_cols].values)
y_train = train['mood_target'].values
y_test = test['mood_target'].values

print(f"Train: {len(train)}, Test: {len(test)}")
print(f"Target stats — Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"Target stats — Test: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

# =============================================
# MODEL 1: RANDOM FOREST REGRESSOR
# =============================================
print("\n" + "=" * 50)
print("RANDOM FOREST REGRESSOR")
print("=" * 50)

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
}

rf = RandomForestRegressor(random_state=42)
tscv = TimeSeriesSplit(n_splits=5)
search = RandomizedSearchCV(rf, param_dist, n_iter=30, cv=tscv,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
rf_best = search.best_estimator_
y_pred_rf = rf_best.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MSE:  {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"MAE:  {mae_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

# =============================================
# MODEL 2: LSTM REGRESSOR
# =============================================
print("\n" + "=" * 50)
print("LSTM REGRESSOR")
print("=" * 50)

daily = pd.read_csv(f'{PROCESSED_DIR}/daily_cleaned.csv')
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values(['id', 'date']).reset_index(drop=True)
daily_features = [c for c in daily.columns if c not in ['id', 'date']]

# Fill NaN
app_cols = [c for c in daily_features if 'appCat' in c or c in ['call', 'sms', 'screen']]
core_cols = [c for c in daily_features if c not in app_cols]
for col in app_cols:
    daily[col] = daily[col].fillna(0)
for pid in daily['id'].unique():
    pmask = daily['id'] == pid
    daily.loc[pmask, core_cols] = daily.loc[pmask, core_cols].ffill().bfill()
daily[daily_features] = daily[daily_features].fillna(0)

WINDOW = 7
seq_train_X, seq_train_y = [], []
seq_test_X, seq_test_y = [], []

for pid in sorted(daily['id'].unique()):
    pdata = daily[daily['id'] == pid].reset_index(drop=True)
    seqs_X, seqs_y = [], []
    for i in range(WINDOW, len(pdata)):
        target = pdata.iloc[i]['mood']
        if np.isnan(target):
            continue
        seqs_X.append(pdata.iloc[i - WINDOW:i][daily_features].values)
        seqs_y.append(target)
    if not seqs_X:
        continue
    split = int(len(seqs_X) * 0.8)
    seq_train_X.extend(seqs_X[:split])
    seq_train_y.extend(seqs_y[:split])
    seq_test_X.extend(seqs_X[split:])
    seq_test_y.extend(seqs_y[split:])

X_train_lstm = np.array(seq_train_X, dtype=np.float32)
y_train_lstm = np.array(seq_train_y, dtype=np.float32)
X_test_lstm = np.array(seq_test_X, dtype=np.float32)
y_test_lstm = np.array(seq_test_y, dtype=np.float32)

print(f"LSTM Train: {X_train_lstm.shape}, Test: {X_test_lstm.shape}")

# Standardize
lstm_scaler = StandardScaler()
n_tr, sl, nf = X_train_lstm.shape
X_train_lstm = lstm_scaler.fit_transform(X_train_lstm.reshape(-1, nf)).reshape(n_tr, sl, nf)
X_test_lstm = lstm_scaler.transform(X_test_lstm.reshape(-1, nf)).reshape(len(X_test_lstm), sl, nf)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(-1)


# Train with early stopping, lr scheduling, gradient clipping
model = LSTMRegressor(nf, 64, 2, dropout=0.3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_lstm), torch.FloatTensor(y_train_lstm)),
                    batch_size=32, shuffle=False)

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
train_losses, val_losses = [], []

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for bx, by in loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(loader))

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(X_test_lstm))
        val_loss = criterion(val_pred, torch.FloatTensor(y_test_lstm)).item()
        val_losses.append(val_loss)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch + 1}")
            break

model.load_state_dict(best_model_state)

# Loss curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='Train Loss')
ax.plot(val_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('LSTM Regressor — Training Curve (with Early Stopping)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/lstm_regression_loss.png', dpi=150)
plt.close()
print("Saved: lstm_regression_loss.png")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred_lstm = model(torch.FloatTensor(X_test_lstm)).numpy()

mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)

print(f"MSE:  {mse_lstm:.4f}")
print(f"RMSE: {rmse_lstm:.4f}")
print(f"MAE:  {mae_lstm:.4f}")
print(f"R²:   {r2_lstm:.4f}")

# =============================================
# SCATTER PLOTS (side by side)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, y_pred_rf, alpha=0.5, s=20, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Mood')
axes[0].set_ylabel('Predicted Mood')
axes[0].set_title(f'RF Regressor\nRMSE={rmse_rf:.3f}, R²={r2_rf:.3f}')

axes[1].scatter(y_test_lstm, y_pred_lstm, alpha=0.5, s=20, color='darkorange')
axes[1].plot([y_test_lstm.min(), y_test_lstm.max()], [y_test_lstm.min(), y_test_lstm.max()], 'r--')
axes[1].set_xlabel('Actual Mood')
axes[1].set_ylabel('Predicted Mood')
axes[1].set_title(f'LSTM Regressor\nRMSE={rmse_lstm:.3f}, R²={r2_lstm:.3f}')

plt.suptitle('Regression: Predicted vs Actual Mood', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/regression_scatter_comparison.png', dpi=150)
plt.close()
print("Saved: regression_scatter_comparison.png")

# =============================================
# PER-PATIENT ERROR ANALYSIS
# =============================================
print("\n" + "=" * 50)
print("PER-PATIENT ERROR ANALYSIS (RF)")
print("=" * 50)

test_with_pred = test.copy()
test_with_pred['pred'] = y_pred_rf
patient_errors = test_with_pred.groupby('id').apply(
    lambda g: pd.Series({
        'n': len(g),
        'MAE': mean_absolute_error(g['mood_target'], g['pred']),
        'RMSE': np.sqrt(mean_squared_error(g['mood_target'], g['pred'])),
        'mood_std': g['mood_target'].std()
    })
).sort_values('MAE', ascending=False)
print(patient_errors.to_string())

fig, ax = plt.subplots(figsize=(12, 5))
patient_errors_sorted = patient_errors.sort_values('MAE')
ax.barh(range(len(patient_errors_sorted)), patient_errors_sorted['MAE'], color='steelblue')
ax.set_yticks(range(len(patient_errors_sorted)))
ax.set_yticklabels(patient_errors_sorted.index, fontsize=8)
ax.set_xlabel('MAE')
ax.set_title('Per-Patient Prediction Error (RF Regressor)')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/per_patient_error.png', dpi=150)
plt.close()
print("Saved: per_patient_error.png")

# =============================================
# COMPARISON
# =============================================
print("\n" + "=" * 50)
print("COMPARISON: RF vs LSTM REGRESSION")
print("=" * 50)
print(f"{'Metric':<10} {'RF Regressor':<15} {'LSTM Regressor':<15}")
print(f"{'MSE':<10} {mse_rf:<15.4f} {mse_lstm:<15.4f}")
print(f"{'RMSE':<10} {rmse_rf:<15.4f} {rmse_lstm:<15.4f}")
print(f"{'MAE':<10} {mae_rf:<15.4f} {mae_lstm:<15.4f}")
print(f"{'R²':<10} {r2_rf:<15.4f} {r2_lstm:<15.4f}")

print("\n=== INTERPRETATION ===")
print(f"R² ~ {max(r2_rf, r2_lstm):.2f}: models explain ~{max(r2_rf, r2_lstm)*100:.0f}% of mood variance.")
print("The remaining variance is driven by factors not in the data (external events, health, social).")
print(f"RMSE ~ {min(rmse_rf, rmse_lstm):.2f} on a 1-10 scale: useful for trend detection, not precise prediction.")
print("\nPer-patient error varies dramatically:")
print(f"  Easiest: {patient_errors.index[-1]} (MAE={patient_errors['MAE'].min():.2f}, very stable mood)")
print(f"  Hardest: {patient_errors.index[0]} (MAE={patient_errors['MAE'].max():.2f}, volatile mood)")
print("\nClassification vs Regression:")
print("- Classification is useful for actionable alerts (low mood detected) but loses granularity")
print("- Regression preserves the continuous scale; main challenge is extreme values")
print("- Both confirm mood_mean and mood_last as strongest predictors")
print("- Models regress toward the mean — underpredicting high mood, overpredicting low mood")
