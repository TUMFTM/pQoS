import sys

import numpy as np

from train import train
from eval import eval
from dataloader import DataLoader

from scipy.stats import randint, loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.multioutput import MultiOutputRegressor
   
def neg_mean_mse(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred, multioutput='raw_values').mean()

def main():
    param_dist = {
        "estimator__n_estimators":      randint(200, 2000),
        "estimator__learning_rate":     loguniform(0.005, 0.3),
        "estimator__max_depth":         randint(3, 12),
        "estimator__min_child_weight":  randint(1, 20),
        "estimator__gamma":             loguniform(1e-3, 2),
        "estimator__subsample":         uniform(0.5, 1.0),
        "estimator__colsample_bytree":  uniform(0.5, 1.0),
        "estimator__reg_alpha":         loguniform(1e-3, 100),
        "estimator__reg_lambda":        loguniform(0.1, 1000),
        "estimator__max_bin":           randint(64, 512)
    }

    ### UL ###
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

    train_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_train.csv", num_cols)

    X_train, y_train = train_dataloader.process(input_features, target_features, 60, 60)

    # Train data in the form of tuple (X_train, y_train)
    train_data = train_dataloader.process(input_features, target_features, 60, 60)

    eval_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_eval_route_2.csv", num_cols, train_dataloader.imputer, train_dataloader.scaler)

    X_eval, y_eval = eval_dataloader.process(input_features, target_features, 60, 60)


    ### Searching ###
    print("Start searching...")

    # 1. Combine train & eval to let PredefinedSplit control splits
    X = np.vstack([X_train, X_eval])
    y = np.vstack([y_train, y_eval])

    test_fold = np.concatenate([
        -1 * np.ones(len(X_train), dtype=int),   # -1 => always train
        0 * np.ones(len(X_eval), dtype=int)     #  0 => test fold index 0
    ])
    ps = PredefinedSplit(test_fold)


    base_model = XGBRegressor(
        tree_method="hist",
        random_state=42,
        objective="reg:squarederror"
    )

    model = MultiOutputRegressor(base_model)

    scorer = make_scorer(neg_mean_mse, greater_is_better=True)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=60,
        scoring=scorer,
        cv=ps,
        verbose=1,
        n_jobs=-1,
        refit=True  # keep best model
    )

    search.fit(
        X, y
    )
    print("Searching finished")

if __name__=="__main__":
    main()
