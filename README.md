# Predictive Quality of Service
This repository contains the code for predictive quality of service (pQoS) for teleoperation of autonomous vehicle. 

The method aims to predict two important metrics for teleoperation: uplink bandwidth and round-trip time latency.

For documentation of the code, please refer to [documentation](doc/documentation.md)

## Concept

The concept of the pQoS pipeline is to use historically recorder QoS data as anchor points to mitigate the concept drift of the trained machine learning models. The machine learing models are trained to predict the difference between the anchor points and the target values, instead of directly prdicting the target value. 

![Prediction Pipeline](doc/figure/prediction_pipeline.jpg)

## Data

The data is stored at mediaTUM and can be downloaded from [here](http://doi.org/10.14459/2025mp1776662).

Put the downloaded data under [data](data/), before running the examples.

## Model

The repository contains two best performing models for uplink and latency prediction based on parameter searching. 

Both [`UL_n_100_depth_4_state_42_lags_60_steps_60.pkl`](model/UL_n_100_depth_4_state_42_lags_60_steps_60.pkl) and [`Latency_n_100_depth_4_state_42_lags_60_steps_60.pkl`](model/Latency_n_100_depth_4_state_42_lags_60_steps_60.pkl) takes 60 frames (lag) of last measurements and predicts 60 frames of future target (step). The input and target features are shown:

| Models            | Input Feature                                                                                          | Target Feature        |
|-------------------|--------------------------------------------------------------------------------------------------------|-----------------------|
| Uplink Model     | Latitude, Longitude, RSRQ, RSRP, SINR, CQI                                                             | UL_difference         |
| Latency Model     | Latitude, Longitude, Latency, TXbitrate, RSRQ,<br>RSRP, SINR, CQI, Latency_prediction, Latency_difference | Latency_difference    |

## Usage

### Install

Create a virtual environment with venv

```
python3 -m venv venv

source venv/bin/activate
```

Install dependencies
```
python3 -m pip install -r requirements
```

### Examples

Run the evaluation example with 
```
source venv/vin/activate && python3 script/eval.py
```

<!-- ## Licsense -->

## Citation

The related paper is currently under publication process. A citation will be added here once the publication becomes available.


