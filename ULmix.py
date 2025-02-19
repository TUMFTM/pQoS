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

def make_lags(series, lags=5, prefix=None):
    prefix = prefix or series.name
    return pd.concat([series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)], axis=1)

def make_multistep_target(series, steps=2, prefix=None):
    prefix = prefix or series.name
    return pd.concat([series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)], axis=1)


# 1. Load Data 
df_train = pd.read_csv("workspace/data/datarate.csv")
if 'UL' in df_train.columns:
    df_train['UL'] = df_train['UL'] * 8 / 1e6

df_val = pd.read_csv("workspace/data/Validation.datarate.csv")

# 2. Split Validation Data
# Take the first 30% of the original validation data to extend the training set
val_split_idx = int(len(df_val) * 0.3)
df_val_for_train = df_val.iloc[:val_split_idx].copy()
df_val_new = df_val.iloc[val_split_idx:].copy()
df_train_extended = pd.concat([df_train, df_val_for_train], axis=0).reset_index(drop=True)

# 3. Define input features
num_cols = [
    'Latitude', 'Longitude',
    'RSRQ', 'RSRP', 'SINR',
    'UL'
]
input_features = [
    'Latitude', 'Longitude',
    'RSRQ', 'RSRP', 'SINR',
    'UL'
]

# 4. Data Preprocessing
# Impute missing values, apply rolling average, then scale
imputer = SimpleImputer(strategy='mean')
df_train_extended[num_cols] = imputer.fit_transform(df_train_extended[num_cols])
df_train_extended['UL'] = df_train_extended['UL'].rolling(window=5, min_periods=1).mean()
scaler = StandardScaler()
df_train_extended[num_cols] = scaler.fit_transform(df_train_extended[num_cols])

df_val_new[num_cols] = imputer.transform(df_val_new[num_cols])
df_val_new['UL'] = df_val_new['UL'].rolling(window=5, min_periods=1).mean()
df_val_new[num_cols] = scaler.transform(df_val_new[num_cols])

# 5. Correlation 
writer = SummaryWriter(log_dir='runs/my_experiment')
corr_sample = df_train_extended.sample(
    n=min(len(df_train_extended), 5000),
    random_state=42
)
corr_matrix = corr_sample[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="viridis", annot=False)
plt.title("Training Data Feature Correlation (Extended)")
writer.add_figure("Training Feature Correlation", plt.gcf())
plt.show()

# 6. Build Features and Targets
# Create lag features and multi-step targets for the training set
lags_train = pd.concat([make_lags(df_train_extended[feat], lags=5, prefix=feat) for feat in input_features],axis=1)
UL_target_train = make_multistep_target(df_train_extended["UL"], steps=2, prefix="UL")

df_train_all = pd.concat([df_train_extended, lags_train, UL_target_train], axis=1).dropna()
target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# Create lag features and multi-step targets for the new validation set
lags_val = pd.concat([make_lags(df_val_new[feat], lags=5, prefix=feat)for feat in input_features],axis=1)
UL_target_val = make_multistep_target(df_val_new["UL"], steps=2, prefix="UL")

df_val_all = pd.concat([df_val_new, lags_val, UL_target_val], axis=1).dropna()
target_cols_val = [col for col in df_val_all.columns if 'step' in col]
feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]

X_val = df_val_all[feature_cols_val].copy()
y_val = df_val_all[target_cols_val].copy()

# 7. Train Model and predict
base_model = XGBRegressor(n_estimators=100, max_depth=8, tree_method='hist', random_state=42, learning_rate=0.1)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred_df = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)

#  8. Inverse Scaling & Error Metrics
ul_idx = num_cols.index("UL")
ul_mean = scaler.mean_[ul_idx]
ul_std = scaler.scale_[ul_idx]

y_pred_inv = y_pred_df.copy().apply(lambda col: col * ul_std + ul_mean)
y_val_inv = y_val.copy().apply(lambda col: col * ul_std + ul_mean)

rmse_step1 = np.sqrt(mean_squared_error(y_val_inv['UL_step1'], y_pred_inv['UL_step1']))
rmse_step2 = np.sqrt(mean_squared_error(y_val_inv['UL_step2'], y_pred_inv['UL_step2']))

mae_step1 = mean_absolute_error(y_val_inv['UL_step1'], y_pred_inv['UL_step1'])
mae_step2 = mean_absolute_error(y_val_inv['UL_step2'], y_pred_inv['UL_step2'])

print(f"UL RMSE Step1: {rmse_step1:.4f}")
print(f"UL RMSE Step2: {rmse_step2:.4f}")
print(f"UL MAE Step1: {mae_step1:.4f}")
print(f"UL MAE Step2: {mae_step2:.4f}")

# 9. Visualization 
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
window_size = 10

# Smooth step1
y_val_inv['UL_step1_smooth'] = (
    y_val_inv['UL_step1']
    .rolling(window=window_size, min_periods=1, center=True)
    .mean()
)
y_pred_inv['UL_step1_smooth'] = (
    y_pred_inv['UL_step1']
    .rolling(window=window_size, min_periods=1, center=True)
    .mean()
)

# Smooth step2
y_val_inv['UL_step2_smooth'] = (
    y_val_inv['UL_step2']
    .rolling(window=window_size, min_periods=1, center=True)
    .mean()
)
y_pred_inv['UL_step2_smooth'] = (
    y_pred_inv['UL_step2']
    .rolling(window=window_size, min_periods=1, center=True)
    .mean()
)

axes[0].plot(y_val_inv.index, y_val_inv['UL_step1_smooth'], 
             label='True UL Step1 (Smooth)', color='C0')
axes[0].plot(y_pred_inv.index, y_pred_inv['UL_step1_smooth'], 
             label='Predicted UL Step1 (Smooth)', color='C1')
axes[0].legend()
axes[0].set_title('UL Step1 - Smoothed')

axes[1].tick_params(axis='y', labelleft=True, left=True)
axes[1].plot(y_val_inv.index, y_val_inv['UL_step2_smooth'], 
             label='True UL Step2 (Smooth)', color='C0')
axes[1].plot(y_pred_inv.index, y_pred_inv['UL_step2_smooth'], 
             label='Predicted UL Step2 (Smooth)', color='C1')
axes[1].legend()
axes[1].set_title('UL Step2 - Smoothed')

plt.tight_layout()
plt.show()

writer.add_figure('Validation UL Steps_mix', fig, global_step=0)
writer.close()
