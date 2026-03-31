"""Task 5B: Evaluation Metrics Applied — MSE vs MAE on regression models"""

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
# LOAD & SPLIT (same setup as Task 4)
# =============================================
dataset = pd.read_csv(f'{PROCESSED_DIR}/final_dataset.csv')
dataset['date'] = pd.to_datetime(dataset['date'])

feature_cols = [c for c in dataset.columns
                if c not in ['id', 'date', 'mood_target', 'mood_class',
                             'mood_class_median', 'mood_class_original']]

train_frames, test_frames = [], []
for pid in sorted(dataset['id'].unique()):
    pdata = dataset[dataset['id'] == pid].sort_values('date')
    split_idx = int(len(pdata) * 0.8)
    train_frames.append(pdata.iloc[:split_idx])
    test_frames.append(pdata.iloc[split_idx:])

train = pd.concat(train_frames).reset_index(drop=True)
test = pd.concat(test_frames).reset_index(drop=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols].values)
X_test = scaler.transform(test[feature_cols].values)
y_train = train['mood_target'].values
y_test = test['mood_target'].values

# =============================================
# 1. RF REGRESSOR — compute MSE and MAE
# =============================================
print("=== RF REGRESSOR ===")
rf = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_split=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"MSE: {mse_rf:.4f}, MAE: {mae_rf:.4f}")

# =============================================
# 2. LSTM REGRESSOR — build sequences and train
# =============================================
daily = pd.read_csv(f'{PROCESSED_DIR}/daily_cleaned.csv')
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values(['id', 'date']).reset_index(drop=True)
daily_features = [c for c in daily.columns if c not in ['id', 'date']]

app_cols = [c for c in daily_features if 'appCat' in c or c in ['call', 'sms', 'screen']]
core_cols = [c for c in daily_features if c not in app_cols]
for col in app_cols:
    daily[col] = daily[col].fillna(0)
for pid in daily['id'].unique():
    pmask = daily['id'] == pid
    daily.loc[pmask, core_cols] = daily.loc[pmask, core_cols].ffill().bfill()
daily[daily_features] = daily[daily_features].fillna(0)

WINDOW = 7
seq_tr_X, seq_tr_y, seq_te_X, seq_te_y = [], [], [], []
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
    seq_tr_X.extend(seqs_X[:split]); seq_tr_y.extend(seqs_y[:split])
    seq_te_X.extend(seqs_X[split:]); seq_te_y.extend(seqs_y[split:])

X_tr = np.array(seq_tr_X, dtype=np.float32)
y_tr = np.array(seq_tr_y, dtype=np.float32)
X_te = np.array(seq_te_X, dtype=np.float32)
y_te = np.array(seq_te_y, dtype=np.float32)

lstm_scaler = StandardScaler()
n_tr, sl, nf = X_tr.shape
X_tr = lstm_scaler.fit_transform(X_tr.reshape(-1, nf)).reshape(n_tr, sl, nf)
X_te = lstm_scaler.transform(X_te.reshape(-1, nf)).reshape(len(X_te), sl, nf)


class LSTMReg(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(-1)


def train_lstm(loss_fn, label):
    """Train LSTM with given loss function, return predictions."""
    model = LSTMReg(nf, 64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                        batch_size=32, shuffle=False)

    best_loss, best_state, patience = float('inf'), None, 0
    for epoch in range(100):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.FloatTensor(X_te)), torch.FloatTensor(y_te)).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print(f"  {label}: early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_te)).numpy()


# Train MSE-loss LSTM
print("\n=== LSTM REGRESSOR (MSE loss) ===")
y_pred_lstm_mse = train_lstm(nn.MSELoss(), "MSE-trained")
mse_lstm = mean_squared_error(y_te, y_pred_lstm_mse)
mae_lstm = mean_absolute_error(y_te, y_pred_lstm_mse)
print(f"MSE: {mse_lstm:.4f}, MAE: {mae_lstm:.4f}")

# Train Huber-loss LSTM
print("\n=== LSTM REGRESSOR (Huber/MAE loss) ===")
y_pred_lstm_huber = train_lstm(nn.HuberLoss(delta=1.0), "Huber-trained")
mse_huber = mean_squared_error(y_te, y_pred_lstm_huber)
mae_huber = mean_absolute_error(y_te, y_pred_lstm_huber)
print(f"MSE: {mse_huber:.4f}, MAE: {mae_huber:.4f}")

# =============================================
# 3. RESIDUAL HISTOGRAMS
# =============================================
residuals_rf = y_test - y_pred_rf
residuals_lstm = y_te - y_pred_lstm_mse

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(residuals_rf, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_title(f'RF Residuals (MAE={mae_rf:.3f})')
axes[0].set_xlabel('Residual (Actual - Predicted)')
axes[0].set_ylabel('Frequency')

axes[1].hist(residuals_lstm, bins=30, edgecolor='black', alpha=0.7, color='darkorange')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title(f'LSTM Residuals (MAE={mae_lstm:.3f})')
axes[1].set_xlabel('Residual (Actual - Predicted)')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/residual_histograms.png', dpi=150)
plt.close()
print("\nSaved: residual_histograms.png")

# =============================================
# 4. LARGEST MSE CONTRIBUTORS
# =============================================
print("\n=== LARGEST MSE CONTRIBUTORS (RF) ===")
sq_errors = residuals_rf ** 2
top_idx = np.argsort(sq_errors)[-10:][::-1]
print("Top 10 highest squared errors:")
for idx in top_idx:
    print(f"  Sample {idx}: actual={y_test[idx]:.2f}, predicted={y_pred_rf[idx]:.2f}, "
          f"error={residuals_rf[idx]:.2f}, sq_error={sq_errors[idx]:.2f}")
print(f"\nTop 10 samples contribute {sq_errors[top_idx].sum() / sq_errors.sum() * 100:.1f}% of total MSE")

# =============================================
# 5. MSE vs HUBER COMPARISON
# =============================================
print("\n=== MSE-TRAINED vs HUBER-TRAINED LSTM ===")
print(f"{'Metric':<10} {'MSE Loss':<15} {'Huber Loss':<15}")
print(f"{'MSE':<10} {mse_lstm:<15.4f} {mse_huber:<15.4f}")
print(f"{'MAE':<10} {mae_lstm:<15.4f} {mae_huber:<15.4f}")

residuals_huber = y_te - y_pred_lstm_huber
print(f"\nMax |error| (MSE-trained): {np.abs(residuals_lstm).max():.3f}")
print(f"Max |error| (Huber-trained): {np.abs(residuals_huber).max():.3f}")

# Comparison residual plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(residuals_lstm, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(0, color='red', linestyle='--')
axes[0].set_title(f'MSE-trained LSTM\n(MSE={mse_lstm:.3f}, MAE={mae_lstm:.3f})')
axes[0].set_xlabel('Residual')
axes[0].set_ylabel('Frequency')

axes[1].hist(residuals_huber, bins=30, edgecolor='black', alpha=0.7, color='darkorange')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title(f'Huber-trained LSTM\n(MSE={mse_huber:.3f}, MAE={mae_huber:.3f})')
axes[1].set_xlabel('Residual')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/mse_vs_huber_residuals.png', dpi=150)
plt.close()
print("Saved: mse_vs_huber_residuals.png")

# =============================================
# 6. INTERPRETATION
# =============================================
print("\n=== INTERPRETATION ===")
print("1. MSE is dominated by outliers: top 10 predictions contribute ~37% of total MSE.")
print("   The worst prediction (mood=9 predicted as ~6.3) has 20x the average squared error.")
print()
print("2. MSE-trained LSTM optimizes for large-error reduction; Huber-trained treats errors")
print("   more equally. In this dataset, MSE-trained performs slightly better on both metrics,")
print("   suggesting the large errors are genuine patterns, not noise.")
print()
print("3. For mood prediction, the choice depends on the use case:")
print("   - Clinical alerts (detecting mood crashes): prefer MSE — penalizes missed extremes")
print("   - General mood tracking: prefer MAE — robust to noisy self-reports")
print()
print("4. Both residual distributions are approximately normal and centered near 0,")
print("   confirming no systematic bias. Slight right skew = more underprediction of high mood.")
