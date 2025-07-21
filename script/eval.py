import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataloader import DataLoader
from train import train

import pdb

def eval(model, X_eval, y_eval, eval_dataloader, mode, model_params):
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

    pdb.set_trace()

    return y_pred_inv, y_val_inv, y_percentage_error, mae_values, rmse_values, std_values

if __name__ == "__main__":
        # num_cols = [
    #     'Latitude', 'Longitude',
    #     'Latency', 'TXbitrate',
    #     'RSRQ', 'RSRP', 'SINR', "CQI", 
    #     "Latency_prediction", "Latency_difference", 
    #     "RSRP_prediction", "RSRQ_prediction", "SINR_prediction", "CQI_prediction"
    # ]

    num_cols = [
        'Latitude', 'Longitude',
        'RSRQ', 'RSRP', 'SINR', "CQI",
        'UL', 'UL_prediction', 'UL_difference', "RSRP_prediction", 
        "RSRQ_prediction", "SINR_prediction", "CQI_prediction"
    ]


    train_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_train.csv", num_cols)

    # input_features = [
    #     # 'Latitude', 'Longitude',
    #     'Latency', 'TXbitrate',
    #     'RSRQ', 'RSRP', 'SINR', "CQI", 
    #     "Latency_prediction", "Latency_difference", 
    #     # "RSRP_prediction_new", "RSRQ_prediction_new", "SINR_prediction_new", "CQI_prediction_new"
    # ]

    input_features = [
        'Latitude', 'Longitude',
        'RSRQ', 'RSRP', 'SINR', "CQI", 
        # "RSRP_prediction_new", "RSRQ_prediction_new", 
        # "SINR_prediction_new", "CQI_prediction_new"
    ]
    # target_features = ["Latency_difference"]
    target_features = ["UL_difference"]

    X_train, y_train = train_dataloader.process(input_features, target_features, 60, 60)


    model_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "tree_method": "hist",
        "random_state": 42,
        "learning_rate": 0.1,
        "mode": "UL", 
        "lags": 60, 
        "steps": 60,
        "model_dir": "model/final_paper"
    }

    print("Model Training...")

    model = train(X_train, y_train, model_params)

    eval_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_eval_route_1.csv", num_cols, train_dataloader.imputer, train_dataloader.scaler)

    X_eval, y_eval = eval_dataloader.process(input_features, target_features, 60, 60)

    eval(model, X_eval, y_eval, eval_dataloader, "UL", model_params)

    pdb.set_trace()

    sys.exit(0)