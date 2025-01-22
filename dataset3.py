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
df = pd.read_csv("workspace/data/dataset3.csv")
print("Data preview:")
print(df.head())

# 2. Numeric feature columns 
num_cols = [
    "Latitude",
    "Longitude",
    "Txbitrate",
    "Linkquality",
    "CQI",
    "PCID",
    "RSRQ",
    "RSRP",
    "SINR",
]

# Handle missing values
df = df.apply(pd.to_numeric, errors='coerce')
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

# 4. Create lag features for inputs, and multi-step target for "Txbitrate"
input_features = [
    "Latitude",
    "Longitude",
    "Linkquality",
    "CQI",
    "PCID",
    "RSRQ",
    "RSRP",
    "SINR",
]

df_input_lags = pd.DataFrame(index=df.index)
for feat in input_features:
    lag_data = make_lags(df[feat], lags=6, prefix=feat)
    df_input_lags = pd.concat([df_input_lags, lag_data], axis=1)

steps = 2
tx_multi = make_multistep_target(df["Txbitrate"], steps=steps, prefix="tx")

df_features = pd.concat([df, df_input_lags], axis=1)
df_targets = pd.concat([tx_multi], axis=1)
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
def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def mae(true, pred):
    return mean_absolute_error(true, pred)

rmse_tx = rmse(y_test, y_pred_df)
mae_tx = mae(y_test, y_pred_df)

print(f"Multi-step Txbitrate RMSE: {rmse_tx:.4f}")
print(f"Multi-step Txbitrate MAE:  {mae_tx:.4f}")

# 10. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

# Tx step1 (top-left)
axes[0].plot(y_test.index, y_test['tx_step1'], label='True Tx Step1')
axes[0].plot(y_pred_df.index, y_pred_df['tx_step1'], label='Pred Tx Step1')
axes[0].legend()
axes[0].set_title('Tx: True vs Pred (step1)')

# Tx step2 (top-right)
axes[1].plot(y_test.index, y_test['tx_step2'], label='True Tx Step2')
axes[1].plot(y_pred_df.index, y_pred_df['tx_step2'], label='Pred Tx Step2')
axes[1].legend()
axes[1].set_title('Tx: True vs Pred (step2)')

plt.tight_layout()
plt.show()

writer.add_figure('tx_steps', fig, global_step=0)
writer.close()
