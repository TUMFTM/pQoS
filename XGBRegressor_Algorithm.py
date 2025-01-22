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
# 1. Read data
###############################################################################
df = pd.read_csv('workspace/data/dataset.csv')
print("Data preview:")
print(df.head())

###############################################################################
# 2. Data preprocessing: missing values, outliers, scaling
###############################################################################
# Assuming these are all numerical columns (remove any irrelevant columns if needed)
num_cols = [
    'X', 'Y',
    'throughput', 'mcs', 'distance', 'pktsize', 
    'delay', 'jitter', 'sinr', 'power', 'dlpathloss', 'pdr'
]

# (2.1) Fill missing values
imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# (2.2) Simple outlier handling: distance >= 0
df = df[df['distance'] >= 0].copy()

# (2.3) Standard scaling
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Visualize correlation matrix
corr_matrix = df[num_cols].corr()
writer = SummaryWriter(log_dir='runs/my_experiment')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Feature Correlation")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
writer.add_figure("Feature Correlation", plt.gcf())
writer.close()
plt.show()

###############################################################################
# 3. Helper functions to generate lag features and multi-step targets
###############################################################################
def make_lags(series: pd.Series, lags=6, prefix=None):
    """
    Generate multiple lag features from one series: t-1, t-2, ... t-lags.
    """
    if prefix is None:
        prefix = series.name
    df_lags = pd.DataFrame(index=series.index)
    for i in range(1, lags + 1):
        df_lags[f"{prefix}_lag{i}"] = series.shift(i)
    return df_lags

def make_multistep_target(series: pd.Series, steps=2, prefix=None):
    """
    Generate multi-step target columns for one series.
    If steps=2, it will create prefix_step1 and prefix_step2.
    """
    if prefix is None:
        prefix = series.name
    df_target = pd.DataFrame(index=series.index)
    for s in range(1, steps + 1):
        df_target[f"{prefix}_step{s}"] = series.shift(-s)
    return df_target

###############################################################################
# 4. Create lag features for the input features and multi-step targets
###############################################################################
#   We only create lags for "input features"
#   Then generate 2-step targets for throughput and delay
###############################################################################

# (4.1) Generate lag features for input features
input_features = ['mcs', 'distance', 'pktsize', 
                  'jitter', 'sinr', 'power', 
                  'dlpathloss', 'pdr']

df_input_lags = pd.DataFrame(index=df.index)
for feat in input_features:
    temp_lags = make_lags(df[feat], lags=6, prefix=feat)
    df_input_lags = pd.concat([df_input_lags, temp_lags], axis=1)

# (4.2) Generate multi-step targets (steps=2) for throughput and delay
steps = 2
throughput_multi = make_multistep_target(df['throughput'], steps=steps, prefix='throughput')
delay_multi = make_multistep_target(df['delay'], steps=steps, prefix='delay')

# Combine original df with the generated lag features
df_features = pd.concat([df, df_input_lags], axis=1)

# Combine multi-step targets
df_targets = pd.concat([throughput_multi, delay_multi], axis=1)

# Final combined dataset (drop NaN caused by shifting)
df_all = pd.concat([df_features, df_targets], axis=1).dropna()
print("After constructing features and targets, data size: ", df_all.shape)

###############################################################################
# 5. Split into feature matrix X and target matrix y
###############################################################################
#   The target columns contain "_step" (throughput_step1, step2, delay_step1, step2)
###############################################################################
target_cols = [c for c in df_all.columns if 'step' in c]
feature_cols = [c for c in df_all.columns if c not in target_cols]

X = df_all[feature_cols].copy()
y = df_all[target_cols].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)

###############################################################################
# 6. Time-based train/test split (80% train, 20% test)
###############################################################################
#   Ensure df_all is in ascending time order if your data isn't already sorted
###############################################################################
n_total = len(X)
n_train = int(n_total * 0.8)

X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]
X_test = X.iloc[n_train:]
y_test = y.iloc[n_train:]

print("Training set size: ", X_train.shape, y_train.shape)
print("Test set size: ", X_test.shape, y_test.shape)

###############################################################################
# 7. Model training with MultiOutputRegressor (XGBRegressor)
###############################################################################
base_estimator = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
model = MultiOutputRegressor(base_estimator)
model.fit(X_train, y_train)

###############################################################################
# 8. Model prediction
###############################################################################
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

###############################################################################
# 9. Evaluation metrics: multi-step RMSE and MAE
###############################################################################
def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def mae(true, pred):
    return mean_absolute_error(true, pred)

throughput_cols = [c for c in y_test.columns if 'throughput_step' in c]
delay_cols = [c for c in y_test.columns if 'delay_step' in c]

rmse_throughput = rmse(y_test[throughput_cols], y_pred_df[throughput_cols])
rmse_delay = rmse(y_test[delay_cols], y_pred_df[delay_cols])

mae_throughput = mae(y_test[throughput_cols], y_pred_df[throughput_cols])
mae_delay = mae(y_test[delay_cols], y_pred_df[delay_cols])

print(f"Multi-step prediction Throughput RMSE: {rmse_throughput:.4f}")
print(f"Multi-step prediction Throughput MAE:  {mae_throughput:.4f}")
print(f"Multi-step prediction Delay RMSE:      {rmse_delay:.4f}")
print(f"Multi-step prediction Delay MAE:       {mae_delay:.4f}")

###############################################################################
# 10. Visualization of true vs. predicted (for step1 and step2)
###############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# Throughput step1
axes[0, 0].plot(y_test.index, y_test['throughput_step1'], label='True Throughput Step1')
axes[0, 0].plot(y_pred_df.index, y_pred_df['throughput_step1'], label='Pred Throughput Step1')
axes[0, 0].legend()
axes[0, 0].set_title('Throughput: True vs Pred (step1)')

# Throughput step2
axes[0, 1].plot(y_test.index, y_test['throughput_step2'], label='True Throughput Step2')
axes[0, 1].plot(y_pred_df.index, y_pred_df['throughput_step2'], label='Pred Throughput Step2')
axes[0, 1].legend()
axes[0, 1].set_title('Throughput: True vs Pred (step2)')

# Delay step1
axes[1, 0].plot(y_test.index, y_test['delay_step1'], label='True Delay Step1')
axes[1, 0].plot(y_pred_df.index, y_pred_df['delay_step1'], label='Pred Delay Step1')
axes[1, 0].legend()
axes[1, 0].set_title('Delay: True vs Pred (step1)')

# Delay step2
axes[1, 1].plot(y_test.index, y_test['delay_step2'], label='True Delay Step2')
axes[1, 1].plot(y_pred_df.index, y_pred_df['delay_step2'], label='Pred Delay Step2')
axes[1, 1].legend()
axes[1, 1].set_title('Delay: True vs Pred (step2)')

plt.tight_layout()
plt.show()

writer = SummaryWriter(log_dir='runs/my_experiment')
writer.add_figure('throughput_delay_steps', fig, global_step=0)
writer.close()
