import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter

def make_lags(series, lags=10, prefix=None):
    """
    Creates lag features for a given series.
    """
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)],
        axis=1
    )

def make_multistep_target(series, steps=6, prefix=None):
    """
    Creates future step targets for a given series.
    By default, it creates 6 steps in the future.
    """
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)],
        axis=1
    )

# 1. Load Training and Validation Data
df_train = pd.read_csv("workspace/data/delay.csv")
df_val = pd.read_csv("workspace/data/Validation.dalay.csv")

# Convert TXbitrate from bits to MB if exists
if 'TXbitrate' in df_train.columns:
    df_train['TXbitrate'] = df_train['TXbitrate'] * 8 / 1e6

# 2. Split Validation (30% Train, 70% Final Validation)
val_split_idx = int(len(df_val) * 0.3)
df_val_for_train = df_val.iloc[:val_split_idx].copy()
df_val_new = df_val.iloc[val_split_idx:].copy()

df_train_extended = pd.concat([df_train, df_val_for_train], axis=0).reset_index(drop=True)

# 3. Define input features
num_cols = ['Latitude', 'Longitude', 'Latency', 'TXbitrate', 'RSRQ', 'RSRP', 'SINR']
input_features = ['Latitude', 'Longitude', 'Latency', 'TXbitrate', 'RSRQ', 'RSRP', 'SINR']

# 4. Preprocessing for Extended Training Set
imputer = SimpleImputer(strategy='mean')
df_train_extended[num_cols] = imputer.fit_transform(df_train_extended[num_cols])

# Smooth Latency with rolling window
df_train_extended['Latency'] = df_train_extended['Latency'].rolling(window=5, min_periods=1).mean()
latency_mean_ext = df_train_extended['Latency'].mean()
df_train_extended.loc[df_train_extended['Latency'] > 200, 'Latency'] = latency_mean_ext

scaler = StandardScaler()
df_train_extended[num_cols] = scaler.fit_transform(df_train_extended[num_cols])

# 5. Preprocessing for New Validation Set
df_val_new[num_cols] = imputer.transform(df_val_new[num_cols])
df_val_new['Latency'] = df_val_new['Latency'].rolling(window=5, min_periods=1).mean()
df_val_new.loc[df_val_new['Latency'] > 200, 'Latency'] = latency_mean_ext
df_val_new[num_cols] = scaler.transform(df_val_new[num_cols])

# 6. Correlation
writer = SummaryWriter(log_dir='runs/my_experiment')
corr_sample = df_train_extended.sample(n=min(len(df_train_extended), 5000), random_state=42)
corr_matrix = corr_sample[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="viridis", annot=False)
plt.title("Training Data Feature Correlation (Extended)")
writer.add_figure("Training Feature Correlation", plt.gcf())
plt.show()

# 7. Create Lags and Multi-step Targets for Training
# ------------------  lags=50 -------------------
num_lags = 50
lags_train = pd.concat(
    [make_lags(df_train_extended[feat], lags=num_lags, prefix=feat) for feat in input_features],
    axis=1
)

# ------------------  steps=40 -------------------
num_steps = 40
Latency_target_train = pd.concat(
    [
        df_train_extended["Latency"].shift(-s).rename(f"Latency_step{s}")
        for s in range(1, num_steps + 1)
    ],
    axis=1
)

df_train_all = pd.concat([df_train_extended, lags_train, Latency_target_train], axis=1).dropna()
target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# 8. Create Lags and Multi-step Targets for New Validation
lags_val = pd.concat(
    [make_lags(df_val_new[feat], lags=num_lags, prefix=feat) for feat in input_features],
    axis=1
)
Latency_target_val = pd.concat(
    [
        df_val_new["Latency"].shift(-s).rename(f"Latency_step{s}")
        for s in range(1, num_steps + 1)
    ],
    axis=1
)

df_val_all = pd.concat([df_val_new, lags_val, Latency_target_val], axis=1).dropna()
target_cols_val = [col for col in df_val_all.columns if 'step' in col]
feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]

X_val = df_val_all[feature_cols_val].copy()
y_val = df_val_all[target_cols_val].copy()

# 9. Train the Model and Predict
base_model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    tree_method='hist',
    random_state=42,
    learning_rate=0.1
)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred_df = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)

# 10. Inverse Scale Latency Predictions and Compute Metrics
lat_idx = num_cols.index("Latency")
lat_mean = scaler.mean_[lat_idx]
lat_std = scaler.scale_[lat_idx]

y_pred_inv = y_pred_df.copy().apply(lambda col: col * lat_std + lat_mean)
y_val_inv = y_val.copy().apply(lambda col: col * lat_std + lat_mean)

rmse_values = {}
mae_values = {}
for s in range(1, num_steps + 1):
    rmse = np.sqrt(mean_squared_error(y_val_inv[f'Latency_step{s}'], y_pred_inv[f'Latency_step{s}']))
    mae = mean_absolute_error(y_val_inv[f'Latency_step{s}'], y_pred_inv[f'Latency_step{s}'])
    rmse_values[f'step{s}'] = rmse
    mae_values[f'step{s}'] = mae


for s in range(1, num_steps + 1):
    print(f"Latency RMSE Step{s}: {rmse_values[f'step{s}']:.4f}")
for s in range(1, num_steps + 1):
    print(f"Latency MAE  Step{s}: {mae_values[f'step{s}']:.4f}")

# 11. Plot Step1, Step2, Step15, Step30
window_size = 10
plot_steps = [1, 2, 15, 30]

for step in plot_steps:
    y_val_inv[f'Latency_step{step}_smooth'] = (
        y_val_inv[f'Latency_step{step}']
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )
    y_pred_inv[f'Latency_step{step}_smooth'] = (
        y_pred_inv[f'Latency_step{step}']
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

# Step1
axes[0, 0].plot(
    y_val_inv.index, y_val_inv['Latency_step1_smooth'], 
    label='True Latency Step1 (Smooth)', color='C0'
)
axes[0, 0].plot(
    y_pred_inv.index, y_pred_inv['Latency_step1_smooth'], 
    label='Predicted Latency Step1 (Smooth)', color='C1'
)
axes[0, 0].legend()
axes[0, 0].set_title('Latency Step1 - Smoothed')

# Step2
axes[0, 1].plot(
    y_val_inv.index, y_val_inv['Latency_step2_smooth'], 
    label='True Latency Step2 (Smooth)', color='C0'
)
axes[0, 1].plot(
    y_pred_inv.index, y_pred_inv['Latency_step2_smooth'], 
    label='Predicted Latency Step2 (Smooth)', color='C1'
)
axes[0, 1].legend()
axes[0, 1].set_title('Latency Step2 - Smoothed')

# Step15
axes[1, 0].plot(
    y_val_inv.index, y_val_inv['Latency_step15_smooth'], 
    label='True Latency Step15 (Smooth)', color='C0'
)
axes[1, 0].plot(
    y_pred_inv.index, y_pred_inv['Latency_step15_smooth'], 
    label='Predicted Latency Step15 (Smooth)', color='C1'
)
axes[1, 0].legend()
axes[1, 0].set_title('Latency Step15 - Smoothed')

# Step30
axes[1, 1].plot(
    y_val_inv.index, y_val_inv['Latency_step30_smooth'], 
    label='True Latency Step30 (Smooth)', color='C0'
)
axes[1, 1].plot(
    y_pred_inv.index, y_pred_inv['Latency_step30_smooth'], 
    label='Predicted Latency Step30 (Smooth)', color='C1'
)
axes[1, 1].legend()
axes[1, 1].set_title('Latency Step30 - Smoothed')

plt.tight_layout()
plt.show()

writer.add_figure('Validation Latency Steps', fig, global_step=0)

abs_errors = np.abs(y_val_inv['Latency_step1'] - y_pred_inv['Latency_step1'])
abs_errors_sorted = np.sort(abs_errors)


n = len(abs_errors_sorted)
cdf = np.arange(1, n + 1) / n

plt.figure(figsize=(10, 6))

plt.plot(abs_errors_sorted, cdf, color='blue')
plt.fill_between(abs_errors_sorted, cdf, alpha=0.2, color='blue')

plt.xlabel('Absolute Error')
plt.ylabel('Probability')
plt.title('Latency Absolute Error (Step1)')
plt.legend()
fig_cdf = plt.gcf()
writer.add_figure('Latency Absolute Error (Step1)', fig_cdf, global_step=0)
writer.close()