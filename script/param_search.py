import sys

from eval import eval_difference
from dataloader import DataLoader
from train import train

def main():
    ######################
    # Param search for UL
    ######################
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

    best_performer = dict()

    for lags, steps in [(10, 10), (20, 10), (30, 10), (60, 10), (20, 20), (30, 20), (60, 20), (30, 30), (60, 30), (60, 60)]:
        train_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_train.csv", num_cols)
        X_train, y_train = train_dataloader.process(input_features, target_features, lags, steps)

        eval_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_eval_route_1.csv", num_cols, train_dataloader.imputer, train_dataloader.scaler)
        X_eval, y_eval = eval_dataloader.process(input_features, target_features, lags, steps)
        for n_estimators in [100, 200, 500, 1000, 2000]:
            for max_depth in [4, 6, 8, 10]:

                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "tree_method": "hist",
                    "random_state": 42,
                    "learning_rate": 0.1,
                    "mode": "UL", 
                    "lags": lags, 
                    "steps": steps,
                    "model_dir": "./model"
                }

                print("Model Training...")

                model = train(X_train, y_train, model_params)

                _, _, _, mae_values, _, _ = eval_difference(model, X_eval, y_eval, eval_dataloader, "UL", model_params)

                best_performer[sum(mae_values.values()) / len(mae_values.values())] = model_params
    
    print(sorted(best_performer.items()))

    ###########################
    # Param search for Latency
    ###########################
    num_cols = [
        'Latitude', 'Longitude',
        'Latency', 'TXbitrate',
        'RSRQ', 'RSRP', 'SINR', "CQI", 
        "Latency_prediction", "Latency_difference", 
        "RSRP_prediction", "RSRQ_prediction", "SINR_prediction", "CQI_prediction"
    ]
    input_features = [
        # 'Latitude', 'Longitude',
        'Latency', 'TXbitrate',
        'RSRQ', 'RSRP', 'SINR', "CQI", 
        "Latency_prediction", "Latency_difference", 
        # "RSRP_prediction_new", "RSRQ_prediction_new", "SINR_prediction_new", "CQI_prediction_new"
    ]
    target_features = ["Latency_difference"]

    best_performer = dict()

    for lags, steps in [(10, 10), (20, 10), (30, 10), (60, 10), (20, 20), (30, 20), (60, 20), (30, 30), (60, 30), (60, 60)]:
        train_dataloader = DataLoader.load_from_csv("data/difference_prediction/latency_train.csv", num_cols)
        X_train, y_train = train_dataloader.process(input_features, target_features, lags, steps)

        eval_dataloader = DataLoader.load_from_csv("data/difference_prediction/latency_eval_route_1.csv", num_cols, train_dataloader.imputer, train_dataloader.scaler)
        X_eval, y_eval = eval_dataloader.process(input_features, target_features, lags, steps)
        for n_estimators in [100, 200, 500, 1000, 2000]:
            for max_depth in [4, 6, 8, 10]:

                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "tree_method": "hist",
                    "random_state": 42,
                    "learning_rate": 0.1,
                    "mode": "Latency", 
                    "lags": lags, 
                    "steps": steps,
                    "model_dir": "./model"
                }

                print("Model Training...")

                model = train(X_train, y_train, model_params)

                _, _, _, mae_values, _, _ = eval_difference(model, X_eval, y_eval, eval_dataloader, "Latency", model_params)

                best_performer[sum(mae_values.values()) / len(mae_values.values())] = model_params
    
    print(sorted(best_performer.items()))

    sys.exit(0)

if __name__ == "__main__":
    main()
