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
import torch
from torch.utils.tensorboard import SummaryWriter

###############################################################################
# 1. Load the dataset
###############################################################################
df = pd.read_csv('workspace/data/dataset2.csv')
print("Data preview:")
print(df.head())

###############################################################################
# 2. Data preprocessing: missing values, outliers, scaling
###############################################################################
# Select numeric columns (adjust based on your data)
num_cols = [
    'ping_ms', 'datarate', 'jitter', 'Latitude', 'Longitude', 'Altitude',
    'speed_kmh', 'COG', 'temperature', 'windSpeed',
    'PCell_RSRP', 'PCell_RSRQ',
    'PCell_RSSI', 'PCell_SNR_1', 'PCell_SNR_2', 'PCell_Uplink_Num_RBs',
    'PCell_Uplink_TB_Size', 'PCell_Uplink_frequency',
    'PCell_Uplink_bandwidth_MHz',
]

# Fill missing values
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Remove simple outliers (e.g., values < 0 for specific columns)
df = df[(df['ping_ms'] >= 0) & (df['datarate'] >= 0)].copy()

df = df.iloc[:3253].copy()

# Convert to float32 to save memory
for c in num_cols:
    df[c] = df[c].astype(np.float32)

# Scale features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Correlation heatmap (sampled for efficiency)
corr_sample = df.sample(n=min(len(df), 5000), random_state=42)
corr_matrix = corr_sample[num_cols].corr()
writer = SummaryWriter(log_dir='runs/my_experiment')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap="viridis", fmt=".2f")
plt.title("Feature Correlation (sampled)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
writer.add_figure("Feature Correlation", plt.gcf())
writer.close()
plt.show()

###############################################################################
# 3. Helper functions to create lag features and multi-step targets
###############################################################################
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

###############################################################################
# 4. Generate lag features and multi-step targets
###############################################################################
input_features = [
    'jitter', 'Latitude', 'Longitude', 'Altitude',
    'speed_kmh', 'PCell_RSRP', 'PCell_RSRQ',
    'PCell_RSSI', 'PCell_SNR_2', 'PCell_Uplink_Num_RBs',
    'PCell_Uplink_TB_Size', 'PCell_Uplink_frequency',
    'PCell_Uplink_bandwidth_MHz',# ping 
]

df_input_lags = pd.DataFrame(index=df.index)
for feat in input_features:
    temp_lags = make_lags(df[feat], lags=6, prefix=feat)
    df_input_lags = pd.concat([df_input_lags, temp_lags], axis=1)

steps = 2
ping_multi = make_multistep_target(df['ping_ms'], steps=steps, prefix='ping')
data_multi = make_multistep_target(df['datarate'], steps=steps, prefix='datarate')

# Combine original data, lag features, and targets
df_features = pd.concat([df, df_input_lags], axis=1)
df_targets = pd.concat([ping_multi, data_multi], axis=1)
df_all = pd.concat([df_features, df_targets], axis=1).dropna()
print("After constructing features and targets, data size:", df_all.shape)

###############################################################################
# 5. Split data into features (X) and targets (y)
###############################################################################
target_cols = [c for c in df_all.columns if 'step' in c]
feature_cols = [c for c in df_all.columns if c not in target_cols]

X = df_all[feature_cols].copy()
y = df_all[target_cols].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)

###############################################################################
# 6. Split data into training and test sets
###############################################################################
n_total = 3253
n_train = int(n_total * 0.8)

X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_test = X.iloc[n_train:]
y_test = y.iloc[n_train:]

print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

###############################################################################
# 7. Train MultiOutputRegressor with XGBRegressor
###############################################################################
base_estimator = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    tree_method='hist',
    random_state=42,
    learning_rate=0.1
)

model = MultiOutputRegressor(base_estimator)
model.fit(X_train, y_train)

###############################################################################
# 8. Predict on test data
###############################################################################
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

###############################################################################
# 9. Evaluate model performance
###############################################################################
def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def mae(true, pred):
    return mean_absolute_error(true, pred)

ping_cols = [c for c in y_test.columns if 'ping_step' in c]
data_cols = [c for c in y_test.columns if 'datarate_step' in c]

rmse_ping = rmse(y_test[ping_cols], y_pred_df[ping_cols])
mae_ping = mae(y_test[ping_cols], y_pred_df[ping_cols])
rmse_data = rmse(y_test[data_cols], y_pred_df[data_cols])
mae_data = mae(y_test[data_cols], y_pred_df[data_cols])

print(f"Multi-step prediction Ping RMSE:    {rmse_ping:.4f}")
print(f"Multi-step prediction Ping MAE:     {mae_ping:.4f}")
print(f"Multi-step prediction Datarate RMSE:{rmse_data:.4f}")
print(f"Multi-step prediction Datarate MAE: {mae_data:.4f}")

###############################################################################
# 10. Visualize predictions
###############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# ping step1
axes[0, 0].plot(y_test.index, y_test['ping_step1'], label='True Ping Step1')
axes[0, 0].plot(y_pred_df.index, y_pred_df['ping_step1'], label='Pred Ping Step1')
axes[0, 0].legend()
axes[0, 0].set_title('Ping: True vs Pred (step1)')

# ping step2
axes[0, 1].plot(y_test.index, y_test['ping_step2'], label='True Ping Step2')
axes[0, 1].plot(y_pred_df.index, y_pred_df['ping_step2'], label='Pred Ping Step2')
axes[0, 1].legend()
axes[0, 1].set_title('Ping: True vs Pred (step2)')

# datarate step1
axes[1, 0].plot(y_test.index, y_test['datarate_step1'], label='True Datarate Step1')
axes[1, 0].plot(y_pred_df.index, y_pred_df['datarate_step1'], label='Pred Datarate Step1')
axes[1, 0].legend()
axes[1, 0].set_title('Datarate: True vs Pred (step1)')

# datarate step2
axes[1, 1].plot(y_test.index, y_test['datarate_step2'], label='True Datarate Step2')
axes[1, 1].plot(y_pred_df.index, y_pred_df['datarate_step2'], label='Pred Datarate Step2')
axes[1, 1].legend()
axes[1, 1].set_title('Datarate: True vs Pred (step2)')

plt.tight_layout()
plt.show()

writer = SummaryWriter(log_dir='runs/my_experiment')
writer.add_figure('ping_datarate_steps', fig, global_step=0)
writer.close()
