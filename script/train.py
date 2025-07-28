import os
import pathlib

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


from dataloader import DataLoader

import pickle

import pdb

def train(X, y, model_params):
    base_model = XGBRegressor(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        tree_method=model_params["tree_method"],
        random_state=model_params["random_state"],
        learning_rate=model_params["learning_rate"]
    )

    model = MultiOutputRegressor(base_model)

    model_dir = model_params["model_dir"]

    model_name = f'{model_params["mode"]}_n_{model_params["n_estimators"]}_depth_{model_params["max_depth"]}_state_{model_params["random_state"]}_lags_{model_params["lags"]}_steps_{model_params["steps"]}.pkl'

    file = os.path.join(
        model_dir,
        model_name
    )
    
    if os.path.exists(file):
        print(f"Model {model_name} exists. Skip training.")
        with open(file, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        print(f"Training model {model_name}")
        model.fit(X, y)
        with open(file, "wb") as f:
            pickle.dump(model, f)
    
    return model
    
if __name__ == "__main__":
    num_cols = [
        'Latitude', 'Longitude',
        'Latency', 'TXbitrate',
        'RSRQ', 'RSRP', 'SINR', "CQI", 
        "Latency_prediction", "Latency_difference", 
        "RSRP_prediction", "RSRQ_prediction", "SINR_prediction", "CQI_prediction"
    ]

    print("Loading Training Data...")

    train_dataloader = DataLoader.load_from_csv("data/difference_prediction/latency_train.csv", num_cols)

    input_features = [
        # 'Latitude', 'Longitude',
        'Latency', 'TXbitrate',
        'RSRQ', 'RSRP', 'SINR', "CQI", 
        "Latency_prediction", "Latency_difference", 
        # "RSRP_prediction_new", "RSRQ_prediction_new", "SINR_prediction_new", "CQI_prediction_new"
    ]
    target_features = ["Latency_difference"]

    print("Processing Training Data...")

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
        "model_dir": "./model"
    }

    print("Model Training...")

    model = train(X_train, y_train, model_params)

    # pdb.set_trace()
