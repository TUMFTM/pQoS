import sys

import numpy as np  
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

class DataLoader:
    def __init__(self, data, imputer = None, scaler = None):
        self.data = data
        self.data_normalized = pd.DataFrame()
        self.fit_imputer = True
        self.fit_scaler = True
        if imputer:
            self.imputer = imputer
            self.fit_imputer = False 
        else:
            self.imputer = SimpleImputer(strategy="mean")
        if scaler:
            self.scaler = scaler
            self.fit_scaler = False
        else:
            self.scaler = StandardScaler()

        self.transform()

    @classmethod
    def load_from_csv(cls, file: str, features: list, imputer = None, scaler = None):
        df = pd.read_csv(file)
        for col in df.columns:
            if col not in features:
                df = df.drop(col, axis=1)
            else:
                # Strip the unit and turn into float
                df[col] = df[col].astype(str).str.strip().str.extract(r'([-+]?\d+(?:\.\d+)?)') .astype(float)
        return cls(df, imputer, scaler)
    
    def transform(self):
        self.data_normalized[self.data.columns] = self.imputer.fit_transform(self.data[self.data.columns]) if self.fit_imputer else self.imputer.transform(self.data[self.data.columns])
        for col in self.data_normalized.columns:
            self.data_normalized[col] = self.data_normalized[col].astype(np.float32)  # Round up in 32-bit float
        self.data_normalized[self.data_normalized.columns] = self.scaler.fit_transform(self.data_normalized[self.data_normalized.columns]) if self.fit_scaler else self.scaler.transform(self.data_normalized[self.data_normalized.columns])


    def process(self, input_features: list, target_features: list, lags: int, steps: int):
        input = pd.concat([make_lags(self.data_normalized[feat], lags=lags, prefix=feat) for feat in input_features], axis=1)
        target = pd.concat([make_multistep_target(self.data_normalized[feat], steps=steps, prefix=feat) for feat in target_features], axis=1)
        df_all = pd.concat([input, target], axis=1).dropna()
        target_cols = [col for col in df_all.columns if 'step' in col]
        feature_cols = [col for col in df_all.columns if col not in target_cols]
        x = df_all[feature_cols].copy()
        y = df_all[target_cols].copy()      
        return x, y


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

    pdb.set_trace()

    eval_dataloader = DataLoader.load_from_csv("data/difference_prediction/uplink_train.csv", num_cols, train_dataloader.imputer, train_dataloader.scaler)

    X_eval, y_eval = eval_dataloader.process(input_features, target_features, 60, 60)

    pdb.set_trace()

    sys.exit(0)
