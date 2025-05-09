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

import pickle
# from tqdm import tqdm

from tum_color import color_pallet, TUMColor

with_knn = False
training = True

ieee_single_column = (3.5, 3)
ieee_double_column = (7.16, 3.5)

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7
})

import pdb

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

print('Loading data...')
# 1. Load Dataset
df_train = pd.read_csv("data/difference_prediction/radio_map_train_ul.csv")
df_test = pd.read_csv("data/difference_prediction/radio_map_eval_ul_unseen.csv")


print('Data loaded.')

# Split the dataset: 80% for training, 20% for validation
print('Data preprocessing...')
# split_idx = int(len(df_train) * 0.8)
# df_val = df_train.iloc[split_idx:].copy()
# df_train = df_train.iloc[:split_idx].copy()

# 2. Preprocessing
num_cols = [
    'Latitude', 'Longitude',
    'RSRQ', 'RSRP', 'SINR', "CQI",
    'UL', 'UL_prediction_new', 'UL_difference', "RSRP_prediction_new", 
    "RSRQ_prediction_new", "SINR_prediction_new", "CQI_prediction_new"
]

# Remove unused colums and format used colums
for col in df_train.columns:
    if col not in num_cols:
        df_train = df_train.drop(col, axis=1)
    else: 
        df_train[col] = df_train[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)

# for col in df_val.columns:
#     if col not in num_cols:
#         df_val = df_val.drop(col, axis=1)
#     else:
#         df_val[col] = df_val[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)

for col in df_test.columns:
    if col not in num_cols:
        df_test = df_test.drop(col, axis=1)
    else:
        df_test[col] = df_test[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)

# pdb.set_trace()

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
for col in num_cols:
    df_train[col] = df_train[col].astype(np.float32)

# Smooth training data
# df_train['UL'] = df_train['UL'].rolling(window=5, min_periods=1).mean() # Smooth UL column
# df_train['UL_prediction_new'] = df_train['UL_prediction_new'].rolling(window=5, min_periods=1).mean() # Smooth UL 
# df_train['UL_difference'] = df_train['UL_difference'].rolling(window=5, min_periods=1).mean() # Smooth UL column

df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

# # Apply the same transformations to the validation set
# df_val[num_cols] = imputer.transform(df_val[num_cols])
# df_val['UL'] = df_val['UL'].rolling(window=5, min_periods=1).mean()
# df_val[num_cols] = scaler.transform(df_val[num_cols])

# Apply the same transformations to the validation set
df_test[num_cols] = imputer.transform(df_test[num_cols])
for col in num_cols:
    df_test[col] = df_test[col].astype(np.float32)
# Smooth testing data
# df_test['UL'] = df_test['UL'].rolling(window=5, min_periods=1).mean() # Smooth UL column
# df_test['UL_prediction_new'] = df_test['UL_prediction_new'].rolling(window=5, min_periods=1).mean() # Smooth UL column
# df_test['U_difference'] = df_test['UL_difference'].rolling(window=5, min_periods=1).mean() # Smooth UL column
df_test[num_cols] = scaler.transform(df_test[num_cols])

# # 3. Feature Correlation
# writer = SummaryWriter(log_dir='runs/my_experiment')
# corr_sample = df_train.sample(n=min(len(df_train), 5000), random_state=42)
# corr_matrix = corr_sample[num_cols].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, cmap="viridis", annot=False)
# plt.title("Training Data Feature Correlation")
# writer.add_figure("Training Feature Correlation", plt.gcf())
# plt.show()

# 4. Create Lag Features and Multi-step Targets
# input_features = [
#     'Latitude', 'Longitude',
#     'RSRQ', 'RSRP', 'SINR',
#     'UL'
# ]
input_features = [
    'Latitude', 'Longitude',
    'RSRQ', 'RSRP', 'SINR', "CQI", 
    # "RSRP_prediction_new", "RSRQ_prediction_new", 
    # "SINR_prediction_new", "CQI_prediction_new"
]



# 这里改成想要的参数
lags = 60      # 调整为 50
steps = 60     # 调整为 40

# For training
# pdb.set_trace()
lags_df = [make_lags(df_train[feat], lags=lags, prefix=feat) for feat in input_features]
if with_knn:
    lags_df.append(pd.concat([df_train['UL_prediction_new']]))
lags_df = pd.concat(lags_df,axis=1)
# UL_target = make_multistep_target(df_train["UL_difference"], steps=steps, prefix="UL")
# UL_target = make_multistep_target(df_train["UL_difference"], steps=steps, prefix="UL")
UL_target = make_multistep_target(df_train["UL"], steps=steps, prefix="UL")

# pdb.set_trace()

df_train_all = pd.concat([lags_df, UL_target], axis=1).dropna()

target_cols = [col for col in df_train_all.columns if 'step' in col]
feature_cols = [col for col in df_train_all.columns if col not in target_cols]

X_train = df_train_all[feature_cols].copy()
y_train = df_train_all[target_cols].copy()

# pdb.set_trace()

# # For validation
# lags_df_val = [make_lags(df_val[feat], lags=lags, prefix=feat) for feat in input_features]
# lags_df_val.append(df_val['UL_prediction_new'])
# lags_df_val = pd.concat(lags_df_val, axis=1)
# UL_target_val = make_multistep_target(df_val["UL"], steps=steps, prefix="UL")

# df_val_all = pd.concat([lags_df_val, UL_target_val], axis=1).dropna()

# target_cols_val = [col for col in df_val_all.columns if 'step' in col]
# feature_cols_val = [col for col in df_val_all.columns if col not in target_cols_val]

# X_val = df_val_all[feature_cols_val].copy()
# y_val = df_val_all[target_cols_val].copy()

# For test
lags_df_test = [make_lags(df_test[feat], lags=lags, prefix=feat) for feat in input_features]
if with_knn:
    lags_df_test.append(df_test['UL_prediction_new'])

lags_df_test = pd.concat(lags_df_test, axis=1)
# UL_target_test = make_multistep_target(df_test["UL_difference"], steps=steps, prefix="UL")
# UL_target_test = make_multistep_target(df_test["UL_difference"], steps=steps, prefix="UL")
UL_target_test = make_multistep_target(df_test["UL"], steps=steps, prefix="UL")


df_test_all = pd.concat([lags_df_test, UL_target_test], axis=1).dropna()

target_cols_test = [col for col in df_test_all.columns if 'step' in col]
feature_cols_test = [col for col in df_test_all.columns if col not in target_cols_test]

X_test = df_test_all[feature_cols_test].copy()
y_test = df_test_all[target_cols_test].copy()

# pdb.set_trace()

print('Preprocessing done.')

# pdb.set_trace()

print('Training...')

# 5. Train the Model and Predict
base_model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    tree_method='hist',
    random_state=42,
    learning_rate=0.1
)

model = MultiOutputRegressor(base_model)

# for col in X_train.columns:
#     print(col)
# for col in X_val.columns:
#     print(col)

# pdb.set_trace()

if with_knn:
    file_name = f"model/difference_prediction/ul_{lags}_{steps}_ul_knn.pkl"
else:
    file_name = f"model/difference_prediction/ul_{lags}_{steps}_wo_ul.pkl"

if training:
    print("Training...")
    # pdb.set_trace() 
    model.fit(X_train, y_train)
    # pdb.set_trace()
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
    print("Training done.")
else:
    print("Loading model from file")
    with open(file_name, "rb") as f:
        model = pickle.load(f)

print('Training done.')

print('Inferencing...')

y_pred = model.predict(X_test)
# y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)
# pdb.set_trace()

# y_pred = []
# for row in tqdm(range(X_val.shape[0])):
#     y_pred.append(model.predict(X_val.iloc[[row]]))
# y_pred_df = pd.DataFrame(y_pred, index=y_val.index, columns=y_val.columns)

# pdb.set_trace()

# 6. Inverse Scale UL Predictions
# ul_idx = num_cols.index("UL_difference")
ul_idx = num_cols.index("UL")
ul_mean = scaler.mean_[ul_idx]
ul_std = scaler.scale_[ul_idx]

y_pred_inv = y_pred * ul_std + ul_mean
y_val_inv = y_test.to_numpy() * ul_std + ul_mean
y_anchors = df_test['UL_prediction_new'].copy().apply(lambda col: col * scaler.scale_[num_cols.index("UL_prediction_new")] + scaler.mean_[num_cols.index("UL_prediction_new")])[lags:].to_numpy()

# # pdb.set_trace()
for i in range(steps - 1):
    y_pred_inv[:,i] = - y_pred_inv[:,i] + y_anchors[i+1:-steps+i+1]
    y_val_inv[:,i] = - y_val_inv[:,i] + y_anchors[i+1:-steps+i+1]

y_pred_inv[:,steps-1] = - y_pred_inv[:,steps-1] + y_anchors[steps:]
y_val_inv[:,steps-1] = - y_val_inv[:,steps-1] + y_anchors[steps:]

y_pred_inv = pd.DataFrame(y_pred_inv, index=y_test.index, columns=y_test.columns)
y_val_inv = pd.DataFrame(y_val_inv, index=y_test.index, columns=y_test.columns)


# pdb.set_trace()

# y_pred_inv = y_pred_df.copy().apply(lambda col: col * ul_std + ul_mean)
# for i in range(steps):
#     y_pred_inv[f"UL_step{i+1}"] += y_anchors[i:-steps+i]
#     pdb.set_trace()
# # y_pred_inv = y_pred_inv.apply(lambda col: y_anchors - col)
# y_val_inv = y_test.copy().apply(lambda col: col * ul_std + ul_mean)
# y_val_inv = y_val_inv.apply(lambda col: y_anchors - col)

# Calculate the percentage error and cummulative error distribution
y_percentage_error = ((y_pred_inv - y_val_inv) / y_val_inv).abs() * 100

# Calculate RMSE for steps 1 to 40
rmse_values = {}
mae_values = {}
std_values = {}
for s in range(1, steps + 1):
    rmse_values[f'step{s}'] = np.sqrt(
        mean_squared_error(y_val_inv[f'UL_step{s}'], y_pred_inv[f'UL_step{s}'])
    )
    mae_values[f'step{s}'] = mean_absolute_error(
        y_val_inv[f'UL_step{s}'], y_pred_inv[f'UL_step{s}']
    )
    std_values[f'step{s}'] = np.std(np.abs(y_val_inv[f'UL_step{s}'].values - y_pred_inv[f'UL_step{s}']))

# pdb.set_trace()

# Print results
for s in range(1, steps + 1):
    print(f"UL RMSE Step{s}: {rmse_values[f'step{s}']:.4f}")
for s in range(1, steps + 1):
    print(f"UL MAE  Step{s}: {mae_values[f'step{s}']:.4f}")

# 7. Plot Step1, Step2, Step15, Step30

window_size = 5
plot_steps = [1, 2, 5, 10, 15, 30]

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

if with_knn:
    sufix = "with_knn"
else:
    sufix = "wo"

######################
# Plot MAE, RMSE, STD
######################

rmse_ls = [rmse_values[f'step{s}'] for s in range(1, steps + 1)]
mae_ls = [mae_values[f'step{s}'] for s in range(1, steps + 1)]
# std_ls = [std_values[f'step{s}'] for s in range(1, steps + 1)]
print(f"MAE: {mae_ls}")

# Plot erstellen
plt.figure(figsize=(4, 3))
plt.plot(range(1, steps + 1), rmse_ls, label='RMSE', color=color_pallet[0], linewidth=0.8)
plt.plot(range(1, steps + 1), mae_ls, label='MAE', color=color_pallet[1], linewidth=0.8)

plt.axhline(y=1.36, color=TUMColor.TUM_DARKRED, linestyle='--', label='Historic Data Overall MAE', linewidth=1)
plt.axhline(y=1.73, color=TUMColor.TUM_GREEN, linestyle='--', label='Historic Data Overall RMSE', linewidth=1)
# plt.plot(range(1, steps + 1), std_ls, label='STD', marker='^')

plt.title('Error Metrics over Prediction Horizons')
plt.xlabel('Step in s')
plt.ylabel('Metric Value in Mbps')
plt.ylim(bottom=0)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"out/mae_ul_{lags}_{steps}_{sufix}.pdf")
    

########################
# Plot percentage error
########################

error_steps = [1, 2 ,15, 30, 60]

plt.figure(figsize=(4, 3))

for idx, step in enumerate(error_steps):
    column_name = f'UL_step{step}'
    data = y_percentage_error[column_name].dropna()
    # Cap values over 100 for plotting but include them in calculations
    capped_data = data.copy()
    capped_data[capped_data > 100] = 100
    sorted_data = capped_data.sort_values()
    cumulative_percentage = [(i + 1) / len(sorted_data) * 100 for i in range(len(sorted_data))]
    plt.plot(sorted_data, cumulative_percentage, color=color_pallet[idx], linestyle='-', linewidth=0.8, label=f"Step {step}")

plt.axvline(x=25, color=TUMColor.TUM_DARKRED, linestyle='--', label='Error = 25%', linewidth=0.8)

# plt.xlim(0, 50)
# plt.ylim(50, 105)
plt.xlabel("Percentage Error in %")
plt.ylabel("Cumulative Percentage of Prediction in %")
plt.title("Cumulative Distribution of Uplink Percentage Error")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig(f"out/cdf_ul_{lags}_{steps}_{sufix}.pdf")

#######################
# Plot smoothed error
#######################

fig, axes = plt.subplots(3, 1, figsize=(3.5, 6), sharex=True, sharey=True)


axes[0].plot(
    y_val_inv.index, y_val_inv['UL_step1_smooth'], 
    label='Ground Truth', color=color_pallet[0], linewidth=0.8
)
axes[0].plot(
    y_pred_inv.index, y_pred_inv['UL_step1_smooth'], 
    label='Prediction', color=color_pallet[1], linewidth=0.8
)
axes[0].legend(loc='lower right')
axes[0].set_title('Uplink Prediction vs Groundtruth for Step 1')
axes[0].set_ylabel("Uplink Data-rate in Mbps")
axes[0].grid(True, linestyle='--', alpha=0.6)

# axes[0, 1].plot(
#     y_val_inv.index, y_val_inv['UL_step2_smooth'], 
#     label='True UL Step2 (Smooth)', color='C0'
# )
# axes[0, 1].plot(
#     y_pred_inv.index, y_pred_inv['UL_step2_smooth'], 
#     label='Predicted UL Step2 (Smooth)', color='C1'
# )
# axes[0, 1].legend()
# axes[0, 1].set_title('UL Step2 - Smoothed')

axes[1].plot(
    y_val_inv.index, y_val_inv['UL_step10_smooth'], 
    label='Ground Truth', color=color_pallet[0], linewidth=0.8
)
axes[1].plot(
    y_pred_inv.index, y_pred_inv['UL_step10_smooth'], 
    label='Prediction', color=color_pallet[1], linewidth=0.8
)
axes[1].legend(loc='lower right')
axes[1].set_title('Uplink Prediction vs Groundtruth for Step 10')
axes[1].set_ylabel("Uplink Data-rate in Mbps")
axes[1].grid(True, linestyle='--', alpha=0.6)

# axes[1, 1].plot(
#     y_val_inv.index, y_val_inv['UL_step10_smooth'], 
#     label='True UL Step10 (Smooth)', color='C0'
# )
# axes[1, 1].plot(
#     y_pred_inv.index, y_pred_inv['UL_step10_smooth'], 
#     label='Predicted UL Step10 (Smooth)', color='C1'
# )
# axes[1, 1].legend()
# axes[1, 1].set_title('UL Step10 - Smoothed')


axes[2].plot(
    y_val_inv.index, y_val_inv['UL_step30_smooth'], 
    label='Ground Truth', color=color_pallet[0], linewidth=0.8
)
axes[2].plot(
    y_pred_inv.index, y_pred_inv['UL_step30_smooth'], 
    label='Prediction', color=color_pallet[1], linewidth=0.8
)
axes[2].legend(loc='lower right')
axes[2].set_title('Uplink Prediction vs Groundtruth for Step 30')
axes[2].set_ylabel("Uplink Data-rate in Mbps")
axes[2].grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Step in s")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.ylabel("Uplink Data-rate (Mbps)")

# axes[2, 1].plot(
#     y_val_inv.index, y_val_inv['UL_step30_smooth'], 
#     label='True UL Step30 (Smooth)', color='C0'
# )
# axes[2, 1].plot(
#     y_pred_inv.index, y_pred_inv['UL_step30_smooth'], 
#     label='Predicted UL Step30 (Smooth)', color='C1'
# )
# axes[2, 1].legend()
# axes[2, 1].set_title('UL Step30 - Smoothed')

plt.tight_layout()
plt.savefig(f"out/ul_{lags}_{steps}_{sufix}.pdf", bbox_inches="tight")
plt.show()

pdb.set_trace()

# writer.add_figure('Validation UL Steps', fig, global_step=0)
# writer.close()
