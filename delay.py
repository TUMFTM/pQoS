import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter
# 1. Load dataset
df = pd.read_csv("data/delay.csv")
print("Data preview:")
print(df.head())

# 2. Numeric feature columns 
num_cols = [
    'ping_ms', 'datarate', 'jitter', 'Latitude', 'Longitude', 'Altitude',
    'speed_kmh', 'COG', 'temperature', 'windSpeed',
    'PCell_RSRP', 'PCell_RSRQ',
    'PCell_RSSI', 'PCell_SNR_1', 'PCell_SNR_2', 'PCell_Uplink_Num_RBs',
    'PCell_Uplink_TB_Size', 'PCell_Uplink_frequency',
    'PCell_Uplink_bandwidth_MHz',
]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Convert to float32
for c in num_cols:
    df[c] = df[c].astype(np.float32)

# Scale features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# correlation heatmap
writer = SummaryWriter(log_dir='runs/my_experiment')
corr_sample = df.sample(n=min(len(df), 5000), random_state=42)
corr_matrix = corr_sample[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="viridis", annot=False)
plt.title("Feature Correlation")
writer.add_figure("Feature Correlation", plt.gcf())
plt.show()

# 3. Helper functions to create lag features and multi-step targets
def make_lags(series: pd.Series, lags=6, prefix=None):
    if prefix is None:
        prefix = series.name
    df_lags = pd.DataFrame(index=series.index)
    for i in range(1, lags + 1):
        df_lags[f"{prefix}_lag{i}"] = series.shift(i)
    return df_lags

def make_multistep_target(series: pd.Series, steps=2, prefix=None):
    if prefix is None:
        prefix = series.name
    df_target = pd.DataFrame(index=series.index)
    for s in range(1, steps + 1):
        df_target[f"{prefix}_step{s}"] = series.shift(-s)
    return df_target

# 4. Create lag features for inputs, and multi-step target for "ping_ms"
input_features = [
    'jitter', 'Latitude', 'Longitude', 'Altitude',
    'speed_kmh', 'PCell_RSRP', 'PCell_RSRQ',
    'PCell_RSSI', 'PCell_SNR_2', 'PCell_Uplink_Num_RBs',
    'PCell_Uplink_TB_Size', 'PCell_Uplink_frequency',
    'PCell_Uplink_bandwidth_MHz', 'ping_ms',
]

df_input_lags = pd.DataFrame(index=df.index)
for feat in input_features:
    lag_data = make_lags(df[feat], lags=6, prefix=feat)
    df_input_lags = pd.concat([df_input_lags, lag_data], axis=1)

steps = 2
ping_multi = make_multistep_target(df["ping_ms"], steps=steps, prefix="ping")

df_features = pd.concat([df, df_input_lags], axis=1)
df_targets = pd.concat([ping_multi], axis=1)
df_all = pd.concat([df_features, df_targets], axis=1).dropna()
print("After feature/target construction:", df_all.shape)

# 5. Separate features (X) and targets (y)
target_cols = [c for c in df_all.columns if 'step' in c]
feature_cols = [c for c in df_all.columns if c not in target_cols]

X = df_all[feature_cols].copy()
y = df_all[target_cols].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)

# 6. Split train/test
n_total = len(X)
n_train = int(n_total * 0.8)
X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_test = X.iloc[n_train:]
y_test = y.iloc[n_train:]

print("Train size:", X_train.shape, y_train.shape)
print("Test size: ", X_test.shape, y_test.shape)

# 7. MultiOutputRegressor with XGB
base_estimator = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    tree_method='hist',
    random_state=42,
    learning_rate=0.1
)
model = MultiOutputRegressor(base_estimator)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

# 9. Evaluation
ping_idx = num_cols.index("ping_ms")
ping_mean = scaler.mean_[ping_idx]
ping_std = scaler.scale_[ping_idx]

y_pred_inv = y_pred_df.copy()
y_test_inv = y_test.copy()

for col in y_pred_inv.columns:
    y_pred_inv[col] = y_pred_inv[col] * ping_std + ping_mean
    y_test_inv[col] = y_test_inv[col] * ping_std + ping_mean


rmse_ping_orig = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae_ping_orig = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"Multi-step ping_ms RMSE (Original Scale): {rmse_ping_orig:.4f}")
print(f"Multi-step ping_ms MAE  (Original Scale):  {mae_ping_orig:.4f}")

# 10. Visualization in original scale
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

# ping step1 
axes[0].plot(y_test_inv.index, y_test_inv['ping_step1'], label='True ping Step1')
axes[0].plot(y_pred_inv.index, y_pred_inv['ping_step1'], label='Pred ping Step1')
axes[0].legend()
axes[0].set_title('Ping: True vs Pred (step1, original scale)')

# ping step2 
axes[1].plot(y_test_inv.index, y_test_inv['ping_step2'], label='True ping Step2')
axes[1].plot(y_pred_inv.index, y_pred_inv['ping_step2'], label='Pred ping Step2')
axes[1].legend()
axes[1].set_title('Ping: True vs Pred (step2, original scale)')

plt.tight_layout()
plt.show()

writer.add_figure('ping_steps_original_scale', fig, global_step=0)

writer.close()