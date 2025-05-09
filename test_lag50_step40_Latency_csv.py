import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter

from tum_color import color_pallet

import pickle

import pdb

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7
})

training = False

def make_lags(series, lags=10, prefix=None):
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(i).rename(f"{prefix}_lag{i}") for i in range(1, lags + 1)],
        axis=1
    )

def make_multistep_target(series, steps=4, prefix=None):
    prefix = prefix or series.name
    return pd.concat(
        [series.shift(-s).rename(f"{prefix}_step{s}") for s in range(1, steps + 1)],
        axis=1
    )

# 1. Load Dataset 
print("Loadking dataset...")
# df_all = pd.read_csv("data/Validation.dalay.csv")
df_train = pd.read_csv("data/difference_prediction/radio_map_train_latency.csv")
df_val = pd.read_csv("data/difference_prediction/radio_map_eval_latency.csv")
print("Dataset loaded.")

# Split into training and validation sets (80% train, 20% validation)
print("Data preprocessing...")
# split_idx = int(len(df_all) * 0.8)
# df_train = df_all.iloc[:split_idx].copy()
# df_val = df_all.iloc[split_idx:].copy()

# 2. Preprocessing 
num_cols = [
    'Latitude', 'Longitude',
    'Latency', 'TXbitrate',
    'RSRQ', 'RSRP', 'SINR', "CQI", 
    "Latency_prediction_new", "Latency_difference", 
    "RSRP_prediction_new", "RSRQ_prediction_new", "SINR_prediction_new", "CQI_prediction_new"
]

# pdb.set_trace()

# for col in num_cols:
#     df_train[col] = df_train[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)
#     df_val[col] = df_val[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)

# df_train_pruned = pd.DataFrame()

# Remove units
for col in df_train.columns:
    if col not in num_cols:
        df_train = df_train.drop(col, axis=1)
    else: 
        df_train[col] = df_train[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)

for col in df_val.columns:
    if col not in num_cols:
        df_val = df_val.drop(col, axis=1)
    else:
        df_val[col] = df_val[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)

# Normalize the data
imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy="constant", fill_value=-1)  # Change to fill the data with -1, not the average
df_train[num_cols] = imputer.fit_transform(df_train[num_cols])

# Convert to float32

# Smooth Latency with rolling window
# df_train['Latency'] = df_train['Latency'].rolling(window=5, min_periods=1).mean()

# Why is this required?
# latency_mean = df_train['Latency'].mean()
# df_train.loc[df_train['Latency'] > 200, 'Latency'] = 100

scaler = StandardScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

df_val[num_cols] = imputer.transform(df_val[num_cols])
# df_val['Latency'] = df_val['Latency'].rolling(window=5, min_periods=1).mean()
# df_val.loc[df_val['Latency'] > 200, 'Latency'] = 200
df_val[num_cols] = scaler.transform(df_val[num_cols])

# # 3. Correlation Plot
# writer = SummaryWriter(log_dir='runs/my_experiment')
# corr_sample = df_train.sample(n=min(len(df_train), 5000), random_state=42)
# corr_matrix = corr_sample[num_cols].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, cmap="viridis", annot=False)
# plt.title("Training Data Feature Correlation")
# writer.add_figure("Training Feature Correlation", plt.gcf())
# plt.show()

# 4. Create Lags and Multi-Step Targets
input_features = [
    # 'Latitude', 'Longitude',
    'Latency', 'TXbitrate',
    'RSRQ', 'RSRP', 'SINR', "CQI", 
    "Latency_prediction_new", "Latency_difference", 
    "RSRP_prediction_new", "RSRQ_prediction_new", "SINR_prediction_new", "CQI_prediction_new"
]

# -------------------- lags=50 --------------------
lags = 60
lags_df = [make_lags(df_train[feat], lags=lags, prefix=feat) for feat in input_features]
lags_df = pd.concat(
    lags_df,
    axis=1
)

# -------------------- steps=40 --------------------
steps = 60
Latency_target = pd.concat(
    [df_train["Latency_difference"].shift(-s).rename(f"Latency_step{s}") for s in range(1, steps + 1)],
    axis=1
)

# pdb.set_trace()

# Combine all training data
df_train_all = pd.concat([lags_df, Latency_target], axis=1).dropna()
target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

# pdb.set_trace()

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# Validation data
lags_df_val = [make_lags(df_val[feat], lags=lags, prefix=feat) for feat in input_features]
lags_df_val = pd.concat(
    lags_df_val,
    axis=1
)
Latency_target_val = pd.concat(
    [df_val["Latency_difference"].shift(-s).rename(f"Latency_step{s}") for s in range(1, steps + 1)],
    axis=1
)

df_val_all = pd.concat([lags_df_val, Latency_target_val], axis=1).dropna()
target_cols_val = [col for col in df_val_all.columns if 'step' in col]
feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]

X_val = df_val_all[feature_cols_val].copy()
y_val = df_val_all[target_cols_val].copy()

# pdb.set_trace()

print("Preprocessing done.")

# 5. Train the Model and Predict
base_model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    tree_method='hist',
    random_state=42,
    learning_rate=0.1
)
model = MultiOutputRegressor(base_model)

file_name = f"model/latency_{lags}_{steps}_wo.pkl"

if training:
    print("Training...")
    model.fit(X_train, y_train)
    # pdb.set_trace()
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
    print("Training done.")
else:
    print("Loading model from file")
    with open(file_name, "rb") as f:
        model = pickle.load(f)

print("Inferencing...")
# pdb.set_trace()
y_pred = model.predict(X_val)
# y_pred_df = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)

# 6. Inverse Scale "Latency" and Calculate Errors
lat_idx = num_cols.index("Latency_difference")
lat_mean = scaler.mean_[lat_idx]
lat_std = scaler.scale_[lat_idx]

# y_pred_inv = y_pred_df.copy().apply(lambda col: col * lat_std + lat_mean)
# y_val_inv = y_val.copy().apply(lambda col: col * lat_std + lat_mean)

y_pred_inv = y_pred * lat_std + lat_mean
y_val_inv = y_val.to_numpy() * lat_std + lat_mean
y_anchors = df_val["Latency_prediction_new"].copy().apply(lambda col: col * scaler.scale_[num_cols.index("Latency_prediction_new")] + scaler.mean_[num_cols.index("Latency_prediction_new")])[lags:].to_numpy()

for i in range(steps - 1):
    y_pred_inv[:,i] = - y_pred_inv[:,i] + y_anchors[i+1:-steps+i+1]
    y_val_inv[:,i] = - y_val_inv[:,i] + y_anchors[i+1:-steps+i+1]

y_pred_inv[:,steps-1] = - y_pred_inv[:,steps-1] + y_anchors[steps:]
y_val_inv[:,steps-1] = - y_val_inv[:,steps-1] + y_anchors[steps:]

y_pred_inv = pd.DataFrame(y_pred_inv, index=y_val.index, columns=y_val.columns)
y_val_inv = pd.DataFrame(y_val_inv, index=y_val.index, columns=y_val.columns)

pdb.set_trace()

print("Inference done.")

rmse_values = {}
mae_values = {}
std_values = {}
for s in range(1, steps + 1):
    rmse = np.sqrt(mean_squared_error(y_val_inv[f'Latency_step{s}'], y_pred_inv[f'Latency_step{s}']))
    mae = mean_absolute_error(y_val_inv[f'Latency_step{s}'], y_pred_inv[f'Latency_step{s}'])
    rmse_values[f'step{s}'] = rmse
    mae_values[f'step{s}'] = mae
    std_values[f'step{s}'] = np.std(np.abs(y_val_inv[f'Latency_step{s}'].values - y_pred_inv[f'Latency_step{s}']))

# pdb.set_trace()


# Calculate the percentage error and cummulative error distribution
y_percentage_error = (y_pred_inv - y_val_inv).abs() / y_val_inv * 100

for s in range(1, steps + 1):
    print(f"Latency RMSE Step{s}: {rmse_values[f'step{s}']:.4f}")
for s in range(1, steps + 1):
    print(f"Latency MAE  Step{s}: {mae_values[f'step{s}']:.4f}")

# 7. Plot Step1, Step2, Step15, Step30
window_size = 10
plot_steps = [1, 2, 10, 15, 30, 45, 60]

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

sufix = "wo"

######################
# Plot MAE, RMSE, STD
######################

rmse_ls = [rmse_values[f'step{s}'] for s in range(1, steps + 1)]
mae_ls = [mae_values[f'step{s}'] for s in range(1, steps + 1)]
std_ls = [std_values[f'step{s}'] for s in range(1, steps + 1)]

print(f"RMSE: {rmse_ls}")
print(f"MAE: {mae_ls}")
print(f"STD: {std_ls}")

# # Plot erstellen
# plt.figure(figsize=(4, 3))
# plt.plot(range(1, steps + 1), rmse_ls, label='RMSE', marker='o', linewidth=0.8, markersize=3)
# plt.plot(range(1, steps + 1), mae_ls, label='MAE', marker='s', linewidth=0.8, markersize=3)
# plt.plot(range(1, steps + 1), std_ls, label='STD', marker='^', linewidth=0.8, markersize=3)

# plt.title('Error Metrics over Prediction Horizon')
# plt.xlabel('Step (s)')
# plt.ylabel('Metric Value (ms)')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"out/mae_latency_{lags}_{steps}_{sufix}.pdf", bbox_inches="tight")
    
########################
# Plot percentage error
########################

plt.figure(figsize=(4, 3))

error_steps = [1, 2, 15, 30, 60]

for idx, step in enumerate(error_steps): 
    column_name = f'Latency_step{step}'
    data = y_percentage_error[column_name].dropna()
    # Cap values over 100 for plotting but include them in calculations
    capped_data = data.copy()
    capped_data[capped_data > 100] = 100
    sorted_data = capped_data.sort_values()
    cumulative_percentage = [(i + 1) / len(sorted_data) * 100 for i in range(len(sorted_data))]

    # print(f'Step {step} sorted data: {sorted_data}')
    # print(f'Step {step} cumulative percentage: {cumulative_percentage}')
    plt.plot(sorted_data, cumulative_percentage, color=color_pallet[idx], linestyle='-', linewidth=0.8, label=f"Step {step}")

plt.axvline(x=25, color='r', linestyle='--', label='Error = 25%', linewidth=0.8)

plt.xlim(0, 75)
plt.xlabel("Percentage Error in %")
plt.ylabel("Cumulative Percentage of Prediction in %")
plt.title("Cumulative Distribution of Latency Percentage Error")
plt.legend(loc='lower right')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig(f"out/cdf_latency_{lags}_{steps}_{sufix}.pdf", bbox_inches="tight")


fig, axes = plt.subplots(3, 1, figsize=(4, 6), sharex=True, sharey=True)

# pdb.set_trace()

# Step1
axes[0].plot(
    y_val_inv.index, y_val_inv['Latency_step1_smooth'], 
    label='Ground Truth', color='C0', linewidth=0.8
)
axes[0].plot(
    y_pred_inv.index, y_pred_inv['Latency_step1_smooth'], 
    label='Prediction', color='C1', linewidth=0.8
)
axes[0].legend(loc='upper right')
axes[0].set_title('Latency Prediction vs Groundtruth for Step 1')
axes[0].set_ylabel("Latency in ms")
axes[0].grid(True, linestyle='--', alpha=0.6)

# # Step2
# axes[0, 1].plot(
#     y_val_inv.index, y_val_inv['Latency_step2_smooth'], 
#     label='True Latency Step2 (Smooth)', color='C0'
# )
# axes[0, 1].plot(
#     y_pred_inv.index, y_pred_inv['Latency_step2_smooth'], 
#     label='Predicted Latency Step2 (Smooth)', color='C1'
# )
# axes[0, 1].legend()
# axes[0, 1].set_title('Latency Step2 - Smoothed')

# Step5
axes[1].plot(
    y_val_inv.index, y_val_inv['Latency_step10_smooth'], 
    label='Ground Truth', color='C0', linewidth=0.8
)
axes[1].plot(
    y_pred_inv.index, y_pred_inv['Latency_step10_smooth'], 
    label='Prediction', color='C1', linewidth=0.8
)
axes[1].legend(loc='upper right')
axes[1].set_title('Latency Prediction vs Groundtruth for Step 10')
axes[1].set_ylabel("Latency in ms")
axes[1].grid(True, linestyle='--', alpha=0.6)

# # Step10
# axes[1, 1].plot(
#     y_val_inv.index, y_val_inv['Latency_step10_smooth'], 
#     label='True Latency Step10 (Smooth)', color='C0'
# )
# axes[1, 1].plot(
#     y_pred_inv.index, y_pred_inv['Latency_step10_smooth'], 
#     label='Predicted Latency Step10 (Smooth)', color='C1'
# )
# axes[1, 1].legend()
# axes[1, 1].set_title('Latency Step10 - Smoothed')

# Step15
axes[2].plot(
    y_val_inv.index, y_val_inv['Latency_step30_smooth'], 
    label='Ground Truth', color='C0', linewidth=0.8
)
axes[2].plot(
    y_pred_inv.index, y_pred_inv['Latency_step30_smooth'], 
    label='Prediction', color='C1', linewidth=0.8
)
axes[2].legend(loc='upper right')
axes[2].set_title('Latency Prediction vs Groundtruth for Step 30')
axes[2].set_ylabel("Latency in ms")
axes[2].grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Step in s")

# # Step30
# axes[2, 1].plot(
#     y_val_inv.index, y_val_inv['Latency_step30_smooth'], 
#     label='True Latency Step30 (Smooth)', color='C0'
# )
# axes[2, 1].plot(
#     y_pred_inv.index, y_pred_inv['Latency_step30_smooth'], 
#     label='Predicted Latency Step30 (Smooth)', color='C1'
# )
# axes[2, 1].legend()
# axes[2, 1].set_title('Latency Step30 - Smoothed')

plt.tight_layout()
plt.savefig(f"out/latency_{lags}_{steps}_{sufix}.pdf", bbox_inches="tight")
plt.show()

pdb.set_trace()
