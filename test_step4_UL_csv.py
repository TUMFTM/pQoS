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

def make_lags(series, lags=10, prefix=None):
    prefix = prefix or series.name
    return pd.concat([series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)], axis=1)

def make_multistep_target(series, steps=2, prefix=None):
    prefix = prefix or series.name
    return pd.concat([series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)], axis=1)


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

steps = 4

# For training
lags_df = pd.concat([make_lags(df_train[feat], lags=10, prefix=feat) for feat in input_features], axis=1)
UL_target = make_multistep_target(df_train["UL"], steps=steps, prefix="UL")
df_train_all = pd.concat([df_train, lags_df, UL_target], axis=1).dropna()

target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# For validation
lags_df_val = pd.concat([make_lags(df_val[feat], lags=10, prefix=feat) for feat in input_features], axis=1)
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

# Calculate RMSE for steps 1 to 4
rmse_step1 = np.sqrt(mean_squared_error(y_val_inv['UL_step1'], y_pred_inv['UL_step1']))
rmse_step2 = np.sqrt(mean_squared_error(y_val_inv['UL_step2'], y_pred_inv['UL_step2']))
rmse_step3 = np.sqrt(mean_squared_error(y_val_inv['UL_step3'], y_pred_inv['UL_step3']))
rmse_step4 = np.sqrt(mean_squared_error(y_val_inv['UL_step4'], y_pred_inv['UL_step4']))

# Calculate MAE for steps 1 to 4
mae_step1 = mean_absolute_error(y_val_inv['UL_step1'], y_pred_inv['UL_step1'])
mae_step2 = mean_absolute_error(y_val_inv['UL_step2'], y_pred_inv['UL_step2'])
mae_step3 = mean_absolute_error(y_val_inv['UL_step3'], y_pred_inv['UL_step3'])
mae_step4 = mean_absolute_error(y_val_inv['UL_step4'], y_pred_inv['UL_step4'])

print(f"UL RMSE Step1: {rmse_step1:.4f}")
print(f"UL RMSE Step2: {rmse_step2:.4f}")
print(f"UL RMSE Step3: {rmse_step3:.4f}")
print(f"UL RMSE Step4: {rmse_step4:.4f}")

print(f"UL MAE  Step1: {mae_step1:.4f}")
print(f"UL MAE  Step2: {mae_step2:.4f}")
print(f"UL MAE  Step3: {mae_step3:.4f}")
print(f"UL MAE  Step4: {mae_step4:.4f}")

