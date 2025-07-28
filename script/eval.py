import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataloader import DataLoader
from train import train

def eval_difference(model, X_eval, y_eval, eval_dataloader, mode, model_params):
    """
    Evaluate models that predict the difference between Radio Environmental Map (REM) and target
    """
    y_pred = model.predict(X_eval)

    columns = eval_dataloader.data.columns
    scaler = eval_dataloader.scaler

    idx = columns.get_loc(f"{mode}_difference")
    mean = scaler.mean_[idx]
    std = scaler.scale_[idx]

    lags = model_params["lags"]
    steps = model_params["steps"]

    y_pred_inv = y_pred * std + mean
    y_val_inv = y_eval.to_numpy() * std + mean

    y_anchors = eval_dataloader.data_normalized[f"{mode}_prediction"].copy().apply(lambda col: col * scaler.scale_[columns.get_loc(f"{mode}_prediction")] + scaler.mean_[columns.get_loc(f"{mode}_prediction")])[lags:].to_numpy()

    for i in range(steps - 1):
        y_pred_inv[:,i] = - y_pred_inv[:,i] + y_anchors[i+1:-steps+i+1]
        y_val_inv[:,i] = - y_val_inv[:,i] + y_anchors[i+1:-steps+i+1]

    y_pred_inv[:,steps-1] = - y_pred_inv[:,steps-1] + y_anchors[steps:]
    y_val_inv[:,steps-1] = - y_val_inv[:,steps-1] + y_anchors[steps:]

    # Here the column names are no longer right, should be UL/Latency, but instead as UL/Latency_difference
    y_pred_inv = pd.DataFrame(y_pred_inv, index=y_eval.index, columns=y_eval.columns)
    y_val_inv = pd.DataFrame(y_val_inv, index=y_eval.index, columns=y_eval.columns)

    y_percentage_error = ((y_pred_inv - y_val_inv) / y_val_inv).abs() * 100

    # Calculate RMSE for steps 1 to 40
    rmse_values = {}
    mae_values = {}
    std_values = {}
    for s in range(1, steps + 1):
        rmse_values[f'step{s}'] = np.sqrt(
            mean_squared_error(y_val_inv[f'{mode}_difference_step{s}'], y_pred_inv[f'{mode}_difference_step{s}'])
        )
        mae_values[f'step{s}'] = mean_absolute_error(
            y_val_inv[f'{mode}_difference_step{s}'], y_pred_inv[f'{mode}_difference_step{s}']
        )
        std_values[f'step{s}'] = np.std(np.abs(y_val_inv[f'{mode}_difference_step{s}'].values - y_pred_inv[f'{mode}_difference_step{s}']))

    return y_pred_inv, y_val_inv, y_percentage_error, mae_values, rmse_values, std_values

def eval_direct(model, X_eval, y_eval, eval_dataloader, mode, model_params):
    """
    Evaluate model that directly predict target
    """
    y_pred = model.predict(X_eval)

    columns = eval_dataloader.data.columns
    scaler = eval_dataloader.scaler

    idx = columns.get_loc(f"{mode}")
    mean = scaler.mean_[idx]
    std = scaler.scale_[idx]

    lags = model_params["lags"]
    steps = model_params["steps"]

    y_pred_inv = y_pred * std + mean
    y_val_inv = y_eval.to_numpy() * std + mean


    y_pred_inv = pd.DataFrame(y_pred_inv, index=y_eval.index, columns=y_eval.columns)
    y_val_inv = pd.DataFrame(y_val_inv, index=y_eval.index, columns=y_eval.columns)

    y_percentage_error = ((y_pred_inv - y_val_inv) / y_val_inv).abs() * 100

    # Calculate RMSE for steps 1 to 40
    rmse_values = {}
    mae_values = {}
    std_values = {}
    for s in range(1, steps + 1):
        rmse_values[f'step{s}'] = np.sqrt(
            mean_squared_error(y_val_inv[f'{mode}_step{s}'], y_pred_inv[f'{mode}_step{s}'])
        )
        mae_values[f'step{s}'] = mean_absolute_error(
            y_val_inv[f'{mode}_step{s}'], y_pred_inv[f'{mode}_step{s}']
        )
        std_values[f'step{s}'] = np.std(np.abs(y_val_inv[f'{mode}_step{s}'].values - y_pred_inv[f'{mode}_step{s}']))

    print(f"mae_values: {mae_values}")
    print(f"mae mean: {sum(mae_values.values()) / len(mae_values.values())}")

    return y_pred_inv, y_val_inv, y_percentage_error, mae_values, rmse_values, std_values

# Exmaple code to train and evaluate one model
if __name__ == "__main__":
    # Define feature columns
    num_cols = [
        'Latitude', 'Longitude',
        'RSRQ', 'RSRP', 'SINR', "CQI",
        'UL', 'UL_prediction', 'UL_difference', "RSRP_prediction", 
        "RSRQ_prediction", "SINR_prediction", "CQI_prediction"
    ]
    input_features = [
        'Latitude', 'Longitude',
        'RSRQ', 'RSRP', 'SINR', "CQI"
    ]
    target_features = ["UL_difference"]

    # Load training data
    train_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_train.csv", num_cols)
    X_train, y_train = train_dataloader.process(input_features, target_features, 60, 60)

    # Define model params and train
    model_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "tree_method": "hist",
        "random_state": 42,
        "learning_rate": 0.1,
        "mode": "UL", 
        "lags": 60, 
        "steps": 60,
        "model_dir": "./model"
    }
    print("Model Training...")
    model = train(X_train, y_train, model_params)

    # Load evaluation data
    eval_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_eval_route_1.csv", num_cols, train_dataloader.imputer, train_dataloader.scaler)
    X_eval, y_eval = eval_dataloader.process(input_features, target_features, 60, 60)


    _, _, _, mae_values, _, _ = eval(model, X_eval, y_eval, eval_dataloader, "UL", model_params)

    print(f"mae_values: {mae_values}")
    print(f"mae mean: {sum(mae_values.values()) / len(mae_values.values())}")

    sys.exit(0)