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

# Helper function: create lag features
def make_lags(series, lags=10, prefix=None):
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)],
        axis=1
    )

# Helper function: create multi-step future targets
def make_multistep_target(series, steps=6, prefix=None):
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)],
        axis=1
    )

# 1. Load training data and convert UL if needed
df_train = pd.read_csv("workspace/data/datarate.csv")
if 'UL' in df_train.columns:
    df_train['UL'] = df_train['UL'] * 8 / 1e6  # convert UL to MB

# 2. Load validation data and split (first 30% is appended to training)
df_val = pd.read_csv("workspace/data/Validation.datarate.csv")
val_split_idx = int(len(df_val) * 0.3)
df_val_for_train = df_val.iloc[:val_split_idx].copy()
df_val_new = df_val.iloc[val_split_idx:].copy()

df_train_extended = pd.concat([df_train, df_val_for_train], axis=0).reset_index(drop=True)

# 3. Define columns
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

# 4. Preprocessing
imputer = SimpleImputer(strategy='mean')
df_train_extended[num_cols] = imputer.fit_transform(df_train_extended[num_cols])
df_train_extended['UL'] = df_train_extended['UL'].rolling(window=5, min_periods=1).mean()

scaler = StandardScaler()
df_train_extended[num_cols] = scaler.fit_transform(df_train_extended[num_cols])

df_val_new[num_cols] = imputer.transform(df_val_new[num_cols])
df_val_new['UL'] = df_val_new['UL'].rolling(window=5, min_periods=1).mean()
df_val_new[num_cols] = scaler.transform(df_val_new[num_cols])

# 5. Correlation (optional)
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

# 6. Build features and targets for the extended training data

num_lags = 50
lags_train = pd.concat(
    [make_lags(df_train_extended[feat], lags=num_lags, prefix=feat) for feat in input_features],
    axis=1
)

steps = 40
UL_target_train = make_multistep_target(df_train_extended["UL"], steps=steps, prefix="UL")

df_train_all = pd.concat([df_train_extended, lags_train, UL_target_train], axis=1).dropna()
target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# 7. Build features and targets for the new validation data
lags_val = pd.concat(
    [make_lags(df_val_new[feat], lags=num_lags, prefix=feat) for feat in input_features],
    axis=1
)
UL_target_val = make_multistep_target(df_val_new["UL"], steps=steps, prefix="UL")

df_val_all = pd.concat([df_val_new, lags_val, UL_target_val], axis=1).dropna()
target_cols_val = [col for col in df_val_all.columns if 'step' in col]
feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]

X_val = df_val_all[feature_cols_val].copy()
y_val = df_val_all[target_cols_val].copy()

# 8. Train and predict
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

# 9. Inverse transform for UL
ul_idx = num_cols.index("UL")
ul_mean = scaler.mean_[ul_idx]
ul_std = scaler.scale_[ul_idx]

y_pred_inv = y_pred_df.copy().apply(lambda col: col * ul_std + ul_mean)
y_val_inv = y_val.copy().apply(lambda col: col * ul_std + ul_mean)

# 10. Calculate RMSE and MAE 
rmse_values = {}
mae_values = {}
for s in range(1, steps + 1):
    rmse = np.sqrt(mean_squared_error(y_val_inv[f'UL_step{s}'], y_pred_inv[f'UL_step{s}']))
    mae = mean_absolute_error(y_val_inv[f'UL_step{s}'], y_pred_inv[f'UL_step{s}'])
    rmse_values[f'step{s}'] = rmse
    mae_values[f'step{s}'] = mae


for s in range(1, steps + 1):
    print(f"UL RMSE Step{s}: {rmse_values[f'step{s}']:.4f}")
for s in range(1, steps + 1):
    print(f"UL MAE  Step{s}: {mae_values[f'step{s}']:.4f}")

# 11. Plot Step1, Step2, Step15, Step30
window_size = 10
plot_steps = [1, 2, 15, 30]


for step in plot_steps:
    y_val_inv[f'UL_step{step}_smooth'] = (
        y_val_inv[f'UL_step{step}']
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )
    y_pred_inv[f'UL_step{step}_smooth'] = (
        y_pred_inv[f'UL_step{step}']
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

# Step1
axes[0, 0].plot(y_val_inv.index, y_val_inv['UL_step1_smooth'], 
                label='True UL Step1 (Smooth)', color='C0')
axes[0, 0].plot(y_pred_inv.index, y_pred_inv['UL_step1_smooth'], 
                label='Predicted UL Step1 (Smooth)', color='C1')
axes[0, 0].legend()
axes[0, 0].set_title('UL Step1 - Smoothed')

# Step2
axes[0, 1].plot(y_val_inv.index, y_val_inv['UL_step2_smooth'], 
                label='True UL Step2 (Smooth)', color='C0')
axes[0, 1].plot(y_pred_inv.index, y_pred_inv['UL_step2_smooth'], 
                label='Predicted UL Step2 (Smooth)', color='C1')
axes[0, 1].legend()
axes[0, 1].set_title('UL Step2 - Smoothed')

# Step15
axes[1, 0].plot(y_val_inv.index, y_val_inv['UL_step15_smooth'], 
                label='True UL Step15 (Smooth)', color='C0')
axes[1, 0].plot(y_pred_inv.index, y_pred_inv['UL_step15_smooth'], 
                label='Predicted UL Step15 (Smooth)', color='C1')
axes[1, 0].legend()
axes[1, 0].set_title('UL Step15 - Smoothed')

# Step30
axes[1, 1].plot(y_val_inv.index, y_val_inv['UL_step30_smooth'], 
                label='True UL Step30 (Smooth)', color='C0')
axes[1, 1].plot(y_pred_inv.index, y_pred_inv['UL_step30_smooth'], 
                label='Predicted UL Step30 (Smooth)', color='C1')
axes[1, 1].legend()
axes[1, 1].set_title('UL Step30 - Smoothed')

plt.tight_layout()
plt.show()

writer.add_figure('Validation UL Steps', fig, global_step=0)


abs_errors = np.abs(y_val_inv['UL_step1'] - y_pred_inv['UL_step1'])

abs_errors_sorted = np.sort(abs_errors)
n = len(abs_errors_sorted)
cdf = np.arange(1, n + 1) / n

plt.figure(figsize=(10, 6))
plt.plot(abs_errors_sorted, cdf, color='blue')
plt.fill_between(abs_errors_sorted, cdf, alpha=0.2, color='blue')
plt.xlabel("Absolute Error")
plt.ylabel("Probability")
plt.title("UL Absolute Error (Step1)")
plt.legend()

fig_cdf = plt.gcf()
writer.add_figure('UL Absolute Error (Step1)', fig_cdf, global_step=0)
plt.show()

writer.close()
