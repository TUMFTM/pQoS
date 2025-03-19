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

# Create lag features
def make_lags(series, lags=10, prefix=None):
    """
    Creates lag features for a given series.
    """
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)],
        axis=1
    )

# Create multi-step targets
def make_multistep_target(series, steps=2, prefix=None):
    """
    Creates future step targets for a given series.
    """
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)],
        axis=1
    )

# 1. Load Dataset
df_all = pd.read_csv("workspace/data/Validation.datarate.csv")

# Split the dataset: 80% for training, 20% for validation
split_idx = int(len(df_all) * 0.8)
df_train = df_all.iloc[:split_idx].copy()
df_val = df_all.iloc[split_idx:].copy()

# 2. Preprocessing
num_cols = [
    'Latitude', 'Longitude',
    'RSRQ', 'RSRP', 'SINR',
    'UL'
]

imputer = SimpleImputer(strategy='mean')

df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
for col in num_cols:
    df_train[col] = df_train[col].astype(np.float32)

# Smooth UL column
df_train['UL'] = df_train['UL'].rolling(window=5, min_periods=1).mean()

scaler = StandardScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

# Apply the same transformations to the validation set
df_val[num_cols] = imputer.transform(df_val[num_cols])
df_val['UL'] = df_val['UL'].rolling(window=5, min_periods=1).mean()
df_val[num_cols] = scaler.transform(df_val[num_cols])

# 3. Feature Correlation
writer = SummaryWriter(log_dir='runs/my_experiment')
corr_sample = df_train.sample(n=min(len(df_train), 5000), random_state=42)
corr_matrix = corr_sample[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="viridis", annot=False)
plt.title("Training Data Feature Correlation")
writer.add_figure("Training Feature Correlation", plt.gcf())
plt.show()

# 4. Create Lag Features and Multi-step Targets
input_features = [
    'Latitude', 'Longitude',
    'RSRQ', 'RSRP', 'SINR',
    'UL'
]

# 这里改成想要的参数
lags = 50      # 调整为 50
steps = 40     # 调整为 40

# For training
lags_df = pd.concat(
    [make_lags(df_train[feat], lags=lags, prefix=feat) for feat in input_features],
    axis=1
)
UL_target = make_multistep_target(df_train["UL"], steps=steps, prefix="UL")

df_train_all = pd.concat([df_train, lags_df, UL_target], axis=1).dropna()

target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# For validation
lags_df_val = pd.concat(
    [make_lags(df_val[feat], lags=lags, prefix=feat) for feat in input_features],
    axis=1
)
UL_target_val = make_multistep_target(df_val["UL"], steps=steps, prefix="UL")

df_val_all = pd.concat([df_val, lags_df_val, UL_target_val], axis=1).dropna()

target_cols_val = [col for col in df_val_all.columns if 'step' in col]
feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]

X_val = df_val_all[feature_cols_val].copy()
y_val = df_val_all[target_cols_val].copy()

# 5. Train the Model and Predict
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

# 6. Inverse Scale UL Predictions
ul_idx = num_cols.index("UL")
ul_mean = scaler.mean_[ul_idx]
ul_std = scaler.scale_[ul_idx]

y_pred_inv = y_pred_df.copy().apply(lambda col: col * ul_std + ul_mean)
y_val_inv = y_val.copy().apply(lambda col: col * ul_std + ul_mean)

# Calculate RMSE for steps 1 to 40
rmse_values = {}
for s in range(1, steps + 1):
    rmse_values[f'step{s}'] = np.sqrt(
        mean_squared_error(y_val_inv[f'UL_step{s}'], y_pred_inv[f'UL_step{s}'])
    )

# Calculate MAE for steps 1 to 40
mae_values = {}
for s in range(1, steps + 1):
    mae_values[f'step{s}'] = mean_absolute_error(
        y_val_inv[f'UL_step{s}'], y_pred_inv[f'UL_step{s}']
    )

# Print results
for s in range(1, steps + 1):
    print(f"UL RMSE Step{s}: {rmse_values[f'step{s}']:.4f}")
for s in range(1, steps + 1):
    print(f"UL MAE  Step{s}: {mae_values[f'step{s}']:.4f}")

# 7. Plot Step1, Step2, Step15, Step30

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


axes[0, 0].plot(
    y_val_inv.index, y_val_inv['UL_step1_smooth'], 
    label='True UL Step1 (Smooth)', color='C0'
)
axes[0, 0].plot(
    y_pred_inv.index, y_pred_inv['UL_step1_smooth'], 
    label='Predicted UL Step1 (Smooth)', color='C1'
)
axes[0, 0].legend()
axes[0, 0].set_title('UL Step1 - Smoothed')

axes[0, 1].plot(
    y_val_inv.index, y_val_inv['UL_step2_smooth'], 
    label='True UL Step2 (Smooth)', color='C0'
)
axes[0, 1].plot(
    y_pred_inv.index, y_pred_inv['UL_step2_smooth'], 
    label='Predicted UL Step2 (Smooth)', color='C1'
)
axes[0, 1].legend()
axes[0, 1].set_title('UL Step2 - Smoothed')

axes[1, 0].plot(
    y_val_inv.index, y_val_inv['UL_step15_smooth'], 
    label='True UL Step15 (Smooth)', color='C0'
)
axes[1, 0].plot(
    y_pred_inv.index, y_pred_inv['UL_step15_smooth'], 
    label='Predicted UL Step15 (Smooth)', color='C1'
)
axes[1, 0].legend()
axes[1, 0].set_title('UL Step15 - Smoothed')

axes[1, 1].plot(
    y_val_inv.index, y_val_inv['UL_step30_smooth'], 
    label='True UL Step30 (Smooth)', color='C0'
)
axes[1, 1].plot(
    y_pred_inv.index, y_pred_inv['UL_step30_smooth'], 
    label='Predicted UL Step30 (Smooth)', color='C1'
)
axes[1, 1].legend()
axes[1, 1].set_title('UL Step30 - Smoothed')

plt.tight_layout()
plt.show()

writer.add_figure('Validation UL Steps', fig, global_step=0)
writer.close()
