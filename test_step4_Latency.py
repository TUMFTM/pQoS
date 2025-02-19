import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter

# 1. Load training data and convert TXbitrate to MB
df_train = pd.read_csv("workspace/data/delay.csv")
if 'TXbitrate' in df_train.columns:
    df_train['TXbitrate'] = df_train['TXbitrate'] * 8 / 1e6  # convert to MB

# 2. Preprocessing: impute, convert type, and scale.
num_cols = [
    'Latitude', 'Longitude',
    'Latency', 'TXbitrate', 'RSRQ', 'RSRP', 'SINR'
]
imputer = SimpleImputer(strategy='mean')
df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
for col in num_cols:
    df_train[col] = df_train[col].astype(np.float32)

# Smooth Latency with rolling window and cap extremely high values
df_train['Latency'] = df_train['Latency'].rolling(window=5, min_periods=1).mean()
latency_mean = df_train['Latency'].mean()
df_train.loc[df_train['Latency'] > 200, 'Latency'] = latency_mean

# Scale features
scaler = StandardScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

# 3. Plot feature correlation
writer = SummaryWriter(log_dir='runs/my_experiment')
corr_sample = df_train.sample(n=min(len(df_train), 5000), random_state=42)
corr_matrix = corr_sample[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="viridis", annot=False)
plt.title("Training Data Feature Correlation")
writer.add_figure("Training Feature Correlation", plt.gcf())
plt.show()

# 4. Helper functions for lags and multi-step targets
def make_lags(series, lags=10, prefix=None):
    prefix = prefix or series.name
    return pd.concat([series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)], axis=1)

def make_multistep_target(series, steps=4, prefix=None):
    prefix = prefix or series.name
    return pd.concat([series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)], axis=1)

# 5. Create lag features and multi-step target for Latency
input_features = [
    'Latitude', 'Longitude',
    'Latency', 'TXbitrate', 'RSRQ', 'RSRP', 'SINR'
]

lags_df = pd.concat([make_lags(df_train[feat], lags=10, prefix=feat) for feat in input_features], axis=1)

steps = 4
Latency_target = make_multistep_target(df_train["Latency"], steps=steps, prefix="Latency")

df_train_all = pd.concat([df_train, lags_df, Latency_target], axis=1).dropna()
target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]
X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# 6. Train the model
base_model = XGBRegressor(n_estimators=100, max_depth=8, tree_method='hist', random_state=42, learning_rate=0.1)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# 7. Load validation data and preprocess
df_val = pd.read_csv("workspace/data/Validation.dalay.csv")

df_val['Latency'] = df_val['Latency'].rolling(window=5, min_periods=1).mean()
latency_idx = num_cols.index("Latency")
train_latency_mean = scaler.mean_[latency_idx]
df_val.loc[df_val['Latency'] > 200, 'Latency'] = train_latency_mean

df_val[num_cols] = imputer.transform(df_val[num_cols])
df_val[num_cols] = scaler.transform(df_val[num_cols])

# Create lag features and multi-step target for validation set
lags_df_val = pd.concat([make_lags(df_val[feat], lags=10, prefix=feat) for feat in input_features], axis=1)
Latency_target_val = make_multistep_target(df_val["Latency"], steps=steps, prefix="Latency")

df_val_all = pd.concat([df_val, lags_df_val, Latency_target_val], axis=1).dropna()
target_cols_val = [col for col in df_val_all.columns if 'step' in col]
feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]
X_val = df_val_all[feature_cols_val].copy()
y_val = df_val_all[target_cols_val].copy()

# 8. Predict on validation set
y_pred = model.predict(X_val)
y_pred_df = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)

# 9. Inverse scale Latency predictions and true values
Latency_idx = num_cols.index("Latency")
Latency_mean = scaler.mean_[Latency_idx]
Latency_std = scaler.scale_[Latency_idx]

y_pred_inv = y_pred_df.copy().apply(lambda col: col * Latency_std + Latency_mean)
y_val_inv = y_val.copy().apply(lambda col: col * Latency_std + Latency_mean)

# Calculate RMSE for steps 1 to 4
rmse_step1 = np.sqrt(mean_squared_error(y_val_inv['Latency_step1'], y_pred_inv['Latency_step1']))
rmse_step2 = np.sqrt(mean_squared_error(y_val_inv['Latency_step2'], y_pred_inv['Latency_step2']))
rmse_step3 = np.sqrt(mean_squared_error(y_val_inv['Latency_step3'], y_pred_inv['Latency_step3']))
rmse_step4 = np.sqrt(mean_squared_error(y_val_inv['Latency_step4'], y_pred_inv['Latency_step4']))

# Calculate MAE for steps 1 to 4
mae_step1 = mean_absolute_error(y_val_inv['Latency_step1'], y_pred_inv['Latency_step1'])
mae_step2 = mean_absolute_error(y_val_inv['Latency_step2'], y_pred_inv['Latency_step2'])
mae_step3 = mean_absolute_error(y_val_inv['Latency_step3'], y_pred_inv['Latency_step3'])
mae_step4 = mean_absolute_error(y_val_inv['Latency_step4'], y_pred_inv['Latency_step4'])

print(f"Latency RMSE Step1: {rmse_step1:.4f}")
print(f"Latency RMSE Step2: {rmse_step2:.4f}")
print(f"Latency RMSE Step3: {rmse_step3:.4f}")
print(f"Latency RMSE Step4: {rmse_step4:.4f}")

print(f"Latency MAE  Step1: {mae_step1:.4f}")
print(f"Latency MAE  Step2: {mae_step2:.4f}")
print(f"Latency MAE  Step3: {mae_step3:.4f}")
print(f"Latency MAE  Step4: {mae_step4:.4f}")

