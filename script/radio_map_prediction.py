###### This code contains the prediction process used for the thesis, including the plots and search optimization functions.


import pandas as pd
import math
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
import math
import haversine as hs
from haversine import Unit
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import re

import pdb


class LocationTimeKey:
    def __init__(self, latitude, longitude, timestamp):
        self.latitude = latitude
        self.longitude = longitude
        self.timestamp = timestamp

    def __lt__(self, other):
        return (self.latitude, self.longitude, self.timestamp) < (other.latitude, other.longitude, other.timestamp)

    def __eq__(self, other):
        return (self.latitude, self.longitude, self.timestamp) == (other.latitude, other.longitude, other.timestamp)

    def __hash__(self):
        return hash((self.latitude, self.longitude, self.timestamp))

class LocationKPIData:
    def __init__(self, latency, uplink, rsrp, rsrq, sinr, cqi):
        self.latency = latency
        self.uplink = uplink
        self.rsrp = rsrp
        self.rsrq = rsrq
        self.sinr = sinr
        self.cqi = cqi

class LocationKPIServiceServer:

    def __init__(self, config_file_path):
        self.radio_map_latency = {}
        self.radio_map_uplink = {}
        self.load_yaml_config(config_file_path)
        #### insert here the path of teh file containing the radio map data
        self.load_csv_data('data/radio_map_train.csv')

    def load_yaml_config(self, config_file_path):
        try:
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)
                self.k = config['k']
                self.radius = config['radius']
                self.x = config['x']
                self.w1 = config['w1']
                self.w2 = config['w2']
                self.w3 = config['w3']
                print(f"Loaded config - k: {self.k}, radius:{self.radius}, x:{self.x}, w1:{self.w1}, w2:{self.w2}, w3:{self.w3}")
        except Exception as e:
            self.k = 20
            self.radius = 100
            self.x = 1
            self.w1 = 0.8
            self.w2 = 0.6
            self.w3 = 0.2
            print(f"Using default params- k: {self.k}, radius:{self.radius}, x:{self.x}, w1:{self.w1}, w2:{self.w2}, w3:{self.w3}")

    def load_csv_data(self, radiomapfile):
        df = pd.read_csv(radiomapfile)
        for _, row in df.iterrows():
            timestamp = row['Timestamp']
            latitude = row['Latitude']
            longitude = row['Longitude']
            latency = self.parse_value(row['Latency'])
            uplink = row['UL']
            rsrp = self.parse_value(row['RSRP'])
            rsrq = self.parse_value(row['RSRQ'])
            sinr = self.parse_value(str(row['SINR']))
            cqi = self.parse_value(str(row['CQI']))

            key = LocationTimeKey(latitude, longitude, timestamp)
            data = LocationKPIData(latency, uplink, rsrp, rsrq, sinr, cqi)
            # data = LocationKPIData(latency, uplink)

            if not math.isnan(latency):
                self.radio_map_latency[key] = data

            if not math.isnan(uplink):
                self.radio_map_uplink[key] = data

    # Parse other KPIs
    def parse_value(self, value_str):
        match = re.search(r'([-+]?\d+(?:\.\d+)?)', value_str.strip())
        if match:
            result = float(match.group(1))
            return result
        else:
            return 10.3


    def handle_request(self, type, csv_file_path):
        df = pd.read_csv(csv_file_path)
        # Prepare the data for prediction
        latitudes = df['Latitude'].values
        longitudes = df['Longitude'].values
        timestamps = df['Timestamp'].values
        # pdb.set_trace()
        # Predict based on the type (latency or uplink)
        true_values = []
        if type == 'Latency':
            for _, row in df.iterrows():
                true_values.append(self.parse_value(str(row['Latency'])))
            true_values = np.array(true_values)
            predicted_values = self.knn_predict_latency(latitudes, longitudes, timestamps)
        elif type == 'UL':
            for _, row in df.iterrows():
                true_values.append(self.parse_value(str(row['UL'])))
            true_values = np.array(true_values)
            predicted_values = self.knn_predict_uplink(latitudes, longitudes, timestamps)
        else:
            predicted_values = []  # Default empty list in case of an invalid type
        # pdb.set_trace()
        df[f'{type}_prediction'] = predicted_values[:,0]
        df[f'{type}_difference'] = predicted_values[:,0] - true_values
        df[f'RSRP_prediction'] = predicted_values[:,1]
        df[f'RSRQ_prediction'] = predicted_values[:,2]
        df[f'SINR_prediction'] = predicted_values[:,3]
        df[f'CQI_prediction'] = predicted_values[:,4]
        # Write the updated DataFrame back to the same CSV file
        df.to_csv(csv_file_path, index=False)


    #### This function is used to split the data into training, validation, and testing sets needed for evaluation 
    def split_data(self, is_latency):
        all_data = []
        Sunday_data = []
        Thursday_data = []
        other_days_data = []
        radio_map = self.radio_map_latency if is_latency else self.radio_map_uplink
        
        for key, data in radio_map.items():
            input_hour, input_day_of_week = self.convert_time(key.timestamp)
            X_entry = [key.latitude, key.longitude, key.timestamp]
            y_entry = data.latency if is_latency else data.uplink
            all_data.append((X_entry, y_entry))
            if input_day_of_week == 3:
                Thursday_data.append((X_entry, y_entry))
            elif input_day_of_week == 6 and 15 <= input_hour < 18: 
                Sunday_data.append((X_entry, y_entry))
            else:
                other_days_data.append((X_entry, y_entry))
        X_all, y_all = zip(*other_days_data)
        X_Thu, y_Thu = zip(*Thursday_data)
        X_Sun, y_Sun = zip(*Sunday_data)
        X_thu_train, X_thu_rest, y_thu_train, y_thu_rest = train_test_split(X_Thu, y_Thu, test_size=0.4, random_state=42)
        X_sun_train, X_sun_rest, y_sun_train, y_sun_rest = train_test_split(X_Sun, y_Sun, test_size=0.4, random_state=42)
        X_thu_val, X_thu_test, y_thu_val, y_thu_test = train_test_split( X_thu_rest, y_thu_rest, test_size=0.5, random_state=42)
        X_sun_val, X_sun_test, y_sun_val, y_sun_test = train_test_split(X_sun_rest, y_sun_rest, test_size=0.5, random_state=42)

        X_train = list(X_all) + list(X_thu_train) + list(X_sun_train)
        y_train = list(y_all) + list(y_thu_train) + list(y_sun_train)
        X_val = list(X_sun_val)  #list(X_thu_val) +
        y_val = list(y_sun_val)  #list(y_thu_val) +
        X_test = list(X_sun_test)  #list(X_thu_test) +
        y_test =  list(y_sun_test) #list(y_thu_test) +
        # self.save_data_to_csv(X_test, y_test, "X_test_y_test.csv")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def save_data_to_csv(self, X, y, filename):
        # Flatten X tuples (which contain timestamp, latitude, longitude) into lists
        X_flat = [list(entry) for entry in X]  # Flatten the X tuples

        # Ensure data is ordered: timestamp, latitude, longitude, value
        X_with_value = [entry + [value] for entry, value in zip(X_flat, y)]
        # Write the data to CSV with correct header and ordering
        np.savetxt(
            filename,
            X_with_value,
            delimiter=",",
            fmt='%s',
            header="timestamp,latitude,longitude,value",  # Ensuring header order
            comments=""
    )

    def knn_predict_latency(self, latitudes, longitudes, timestamps):
        return self.knn_predict(latitudes, longitudes, timestamps, is_latency=True)

    def knn_predict_uplink(self, latitudes, longitudes, timestamps):
        return self.knn_predict(latitudes, longitudes, timestamps, is_latency=False)

    def spatial_distance(self, lat1, lon1, lat2, lon2):
         loc1 = (lat1, lon1)
         loc2 = (lat2, lon2)
         hav_distance = hs.haversine(loc1, loc2, unit=Unit.METERS)
         return hav_distance


    ### please adjust this code by uncommenting the version you want to use
    def custom_distance(self, x1, x2):
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3

        ### for version 1 and 2
        # lat1, lon1 = x1[0], x1[1]
        # lat2, lon2 = x2[0], x2[1]
        # geo_distance = self.spatial_distance(lat1, lon1, lat2, lon2)
        # total_distance = geo_distance 

        ### for version 3
        lat1, lon1, time1 = x1[0], x1[1], x1[2]
        lat2, lon2, time2 = x2[0], x2[1], x2[2]
        geo_distance = self.spatial_distance(lat1, lon1, lat2, lon2)
        if geo_distance == 0:
            normalized_geo_distance = 0
        elif geo_distance <= 40:
            # Scale distances <= 40 meters to fit within [0, 0.8]
            normalized_geo_distance = (geo_distance / 40) * 0.8
        else:
            # Scale distances > 40 meters to fit within [0.8, 1]
            normalized_geo_distance = 0.8 + ((geo_distance - 40) / (geo_distance + 1)) * 0.2

        time_diff_seconds = abs(time1 - time2)
        time_diff_within_week_seconds = time_diff_seconds % 604800  # Time within the same week
        if time_diff_within_week_seconds <= 7200:
            normalized_time_diff_within_week = (time_diff_within_week_seconds / 7200) * 0.8
        else:
            # Scale the time > 2 hours to fit within [0.8, 1]
            normalized_time_diff_within_week = 0.8 + ((time_diff_within_week_seconds - 7200) / (time_diff_within_week_seconds + 1)) * 0.2
        time_diff_weeks = time_diff_seconds // 604800  # Time difference in weeks
        normalized_time_diff_weeks = time_diff_weeks / (time_diff_weeks + 1)

        total_distance = w1 * normalized_geo_distance + w2 * normalized_time_diff_within_week + w3 * normalized_time_diff_weeks 
        return total_distance 

    def custom_distance_callable(self, x1, x2):
        return self.custom_distance(x1, x2)

    ### please adjust this code by uncommenting the version you want to use
    def knn_predict(self, latitudes, longitudes, timestamps, is_latency): 
        predicted_values = [] 
        radio_map = self.radio_map_latency if is_latency else self.radio_map_uplink  
        X_train = []  
        y_train = []  
        X_train_2 = []  
        y_train_2 = []  
        X_test_2 = []
        x = self.x
        radius = self.radius
        k = self.k
        # X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(is_latency)
        # self.save_data_to_csv(X_test, y_test, "X_test_y_test.csv")

        # #### Version 1 normal
        # # Extract the features and target values from the radio_map
        # for key, data in radio_map.items(): 
        #     X_train.append([key.latitude, key.longitude]) 
        #     y_train.append(data.uplink)
        # X_train = np.array(X_train)
        # y_train = np.array(y_train)
        # X_test = np.column_stack((latitudes, longitudes))

        # ## Version 1 using data split
        # X_train_2 = np.array([entry[:2] for entry in X_train])  
        # y_train_2 = np.array(y_train)
        # X_test_2 = np.array([entry[:2] for entry in X_test])
        
        # ### Version 2 normal
        # input_hour, input_day_of_week = self.convert_time(timestamps[0])
        # filtered_radio_map = {}
        # # Function to filter the radio map based on day and time range
        # def filter_radio_map_by_time(time_range, day_check_func):
        #     for key, data in radio_map.items():
        #         map_hour, map_day_of_week = self.convert_time(key.timestamp)
        #         if day_check_func(map_day_of_week) and time_range(map_hour):
        #             filtered_radio_map[key] = data
        #     print("Filtered Radio Map:")
        #     for key, data in filtered_radio_map.items():
        #         # Assuming 'key' is an instance of LocationTimeKey, and 'data' is an instance of LocationKPIData
        #         print(f"Location: ({key.latitude}, {key.longitude}), Timestamp: {key.timestamp}, Uplink: {data.uplink}")
        # # Weekday (Mon-Fri)
        # if input_day_of_week <= 4:  
        #     # Check the time range and filter the radio map
        #     if 9 <= input_hour < 12:
        #         filter_radio_map_by_time(lambda hour: 9 <= hour < 12, lambda day: day <= 4)
        #     elif 12 <= input_hour <= 13:
        #         filter_radio_map_by_time(lambda hour: 12 <= hour <= 13, lambda day: day <= 4)
        #     elif 16 <= input_hour <= 20:
        #         filter_radio_map_by_time(lambda hour: 16 <= hour <= 20, lambda day: day <= 4)
        # # Weekend (Sat-Sun)
        # elif input_day_of_week >= 5:  
        #     # Check the time range and filter the radio map
        #     if 15 <= input_hour < 18:
        #         filter_radio_map_by_time(lambda hour: 15 <= hour < 18, lambda day: day >= 5)
        #     elif input_hour == 10:
        #         filter_radio_map_by_time(lambda hour: hour == 10, lambda day: day >= 5)
        #     elif input_hour == 19:
        #         filter_radio_map_by_time(lambda hour: hour == 19, lambda day: day >= 5)
        # # Extract the features and target values from the radio_map
        # for key, data in filtered_radio_map.items():
        #     X_train.append([key.latitude, key.longitude]) 
        #     y_train.append(data.uplink)
        # X_train = np.array(X_train)
        # y_train = np.array(y_train)
        # X_test = np.column_stack((latitudes, longitudes))

        # #### Version 2 using  data split
        # X_train = np.array(X_train)
        # y_train = np.array(y_train)
        # X_train_filtered = []
        # y_train_filtered = []
        # test_sample = X_test[0] 
        # test_timestamp = test_sample[2] 
        # test_hour, test_day_of_week = self.convert_time(test_timestamp)
        # def filter_X_train_by_time(time_range, day_check_func):
        #     for i in range(len(X_train)):
        #         train_timestamp = X_train[i][2]  
        #         train_hour, train_day_of_week = self.convert_time(train_timestamp)
        #         if day_check_func(train_day_of_week) and time_range(train_hour):
        #             X_train_filtered.append([float(X_train[i][0]), float(X_train[i][1]),(X_train[i][2])])
        #             y_train_filtered.append(y_train[i])
        # # Weekday (Mon-Fri)
        # if test_day_of_week <= 4:  
        #     # Check the time range and filter the radio map
        #     if 9 <= test_hour < 12:
        #         filter_X_train_by_time(lambda hour: 9 <= hour < 12, lambda day: day <= 4)
        #     elif 12 <= test_hour <= 13:
        #         filter_X_train_by_time(lambda hour: 12 <= hour <= 13, lambda day: day <= 4)
        #     elif 16 <= test_hour <= 20:
        #        filter_X_train_by_time(lambda hour: 16 <= hour <= 20, lambda day: day <= 4)
        # # Weekend (Sat-Sun)
        # elif test_day_of_week >= 5:  
        #     # Check the time range and filter the radio map
        #     if 15 <= test_hour < 18:
        #         filter_X_train_by_time(lambda hour: 15 <= hour < 18, lambda day: day >= 5)
        #     elif test_hour == 10:
        #         filter_X_train_by_time(lambda hour: hour == 10, lambda day: day >= 5)
        #     elif test_hour == 19:
        #         filter_X_train_by_time(lambda hour: hour == 19, lambda day: day >= 5)
        # X_train_2 = np.array([entry[:2] for entry in X_train_filtered])  
        # y_train_2 = np.array(y_train_filtered)
        # X_test_2 = np.array([entry[:2] for entry in X_test])           

        ### Version 3 normal            
        # Extract the features and target values from the radio_map
        for key, data in radio_map.items():
            timestamp = self.convert_time_to_unix(key.timestamp)
            X_train.append([key.latitude, key.longitude, timestamp]) 
            # y_train.append(data.latency if is_latency else data.uplink)
            if is_latency:
                y_train.append([data.latency, data.rsrp, data.rsrq, data.sinr, data.cqi])
            else:
                y_train.append([data.uplink, data.rsrp, data.rsrq, data.sinr, data.cqi])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        unix_timestamps = [self.convert_time_to_unix(ts) for ts in timestamps]
        X_test = np.column_stack((latitudes, longitudes, unix_timestamps))

        # ### Verison 3 data split
        # for entry in X_train:
        #     # Convert each element to float
        #     latitude = float(entry[0])
        #     longitude = float(entry[1])
        #     timestamp = self.convert_time_to_unix(entry[2])  # Convert the timestamp to Unix time
        #     # Append the processed entry as a list [latitude, longitude, timestamp]
        #     X_train_2.append([latitude, longitude, timestamp])
        # # Convert X_train_2 to a NumPy array
        # X_train_2 = np.array(X_train_2)
        # y_train_2 = np.array(y_train)
        # for entry in X_test:
        #     # Convert each element to float
        #     latitude = float(entry[0])
        #     longitude = float(entry[1])
        #     timestamp = self.convert_time_to_unix(entry[2])  # Convert the timestamp to Unix time
        #     # Append the processed entry as a list [latitude, longitude, timestamp]
        #     X_test_2.append([latitude, longitude, timestamp])
        # # Convert X_train_2 to a NumPy array
        # X_test_2 = np.array(X_test_2)    # Convert to NumPy array


        if x ==1:
            # K-NN Regression
            knn_regressor = KNeighborsRegressor(n_neighbors=k, metric=self.custom_distance_callable, weights='uniform')
            # pdb.set_trace()
            # knn_regressor.fit(X_train_2, y_train_2)
            # predicted_values = knn_regressor.predict(X_test_2)
            knn_regressor.fit(X_train, y_train)
            predicted_values = knn_regressor.predict(X_test)
            # # Get the nearest neighbors for the last test point (X_test[-1])
            # distances, indices = knn_regressor.kneighbors(X_val_2[-1].reshape(1, -1))        
            # print("Nearest neighbors for the last test point:")
            # for idx in indices[0]:
            #     lat, lon, timestamp_unix = X_train_2[idx]  # Extract latitude, longitude, and timestamp
            #     timestamp_readable = datetime.utcfromtimestamp(timestamp_unix).strftime('%Y-%m-%d %H:%M:%S')
            #     print(f"Neighbor {idx}: Lat: {lat}, Lon: {lon}, timestamp_readable: {timestamp_readable},  - Distance: {distances[0][indices[0].tolist().index(idx)]}")

        elif x == 2:
            #  RadiusNeighborsRegressor
            distance_limit = radius
            radius_regressor = RadiusNeighborsRegressor(radius=radius, metric=self.custom_distance_callable, weights='uniform')
            # radius_regressor.fit(X_train_no_timestamp, y_train_2)
            # predicted_values = radius_regressor.predict(X_val_no_timestamp)

        return predicted_values

    def convert_time(self, timestamp_str):
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        hours = timestamp_dt.hour + timestamp_dt.minute / 60 + timestamp_dt.second / 3600 # Convert to hours in decimal form
        day_of_week = timestamp_dt.weekday()  # Get day of the week (0=Monday, 6=Sunday)
        return hours, day_of_week
    def convert_time_to_unix(self, timestamp_str):
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        unix_timestamp = int(timestamp_dt.timestamp())
        return unix_timestamp

    #### Plots and metrics
    # Plot 1: AE over location
    def plot_error_over_location(self, csv_file_path, k_value, data_type):
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        # Apply parse_value only to the true values (data_type column) that have "ms"
        df[data_type] = df[data_type].apply(self.parse_value)
        # Calculate the absolute error between true values and predictions
        abs_error = abs(df[data_type] - df[f'{data_type}_prediction'])
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        # Scatter plot with longitude and latitude, colored by absolute error
        sc1 = ax.scatter(df['Longitude'], df['Latitude'], c=abs_error, cmap='coolwarm', s=20)
        # Add color bar for the prediction error
        plt.colorbar(sc1, ax=ax, label='Prediction Error')
        # Set plot title and axis labels
        ax.set_title(f'AE over Location (K={k_value}) - {data_type.capitalize()}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # Add interactive tooltip showing absolute error when hovering over points
        mplcursors.cursor(sc1, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(f'AE: {abs_error[sel.index]:.2f}')
        )
        # Show the legend and the plot
        plt.legend(title=f"Prediction Model (K={k_value})", loc="upper right")
        plt.show()
    
    # Plot 2: Latency/uplink difference over location
    def plot_prediction_difference(self, csv_file_path, k_value, data_type):
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        # Apply parse_value to the true values (data_type column) to remove 'ms' and convert to numerical values
        df[data_type] = df[data_type].apply(self.parse_value)
        # Extract true values and predictions
        true_values = df[data_type]
        pred_k1 = df[f'{data_type}_prediction']
        # Calculate latency difference (True - Predicted)
        latency_diff = true_values - pred_k1
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(df['Longitude'], df['Latitude'], c=latency_diff, cmap='coolwarm', s=20)
        # Add color bar for the latency difference
        plt.colorbar(label='Latency Difference (True - Predicted)')
        #Handle outliers by marking points outside the range (-25 to 25)
        outliers = (latency_diff < -25) | (latency_diff > 25)
        plt.scatter(
            df[outliers]['Longitude'], df[outliers]['Latitude'],
            color='blue', marker='x', label='Outliers (Latency Difference < -25 or > 25)', alpha=0.7
        )
        # Set plot title and axis labels
        plt.title(f'Latency Difference (True - Predicted) for K={k_value} - {data_type.capitalize()}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # Add interactive tooltip showing latency difference when hovering over points
        mplcursors.cursor(plt.gca(), hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(f'Diff: {latency_diff[sel.index]:.2f} ms')
        )
        # Show the legend and the plot
        plt.legend(title=f"Prediction Model (K={k_value})", loc="upper right")
        plt.show()
    
    # Plot 3: True vs prediction
    def plot_true_vs_pred(self, csv_file_path, k_value, data_type):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        # Apply parse_value to the true values to remove 'ms' and convert to numerical values
        df[data_type] = df[data_type].apply(self.parse_value)
        df[f'{data_type}_prediction'] = df[f'{data_type}_prediction'].apply(self.parse_value)
        # Extract true values and predictions
        true_values = df[data_type]
        pred_k1 = df[f'{data_type}_prediction']
        # Plotting the true vs predicted values
        plt.figure(figsize=(10, 8))  # Set a larger figure size
        # Scatter plot for true values (blue)
        plt.scatter(true_values, true_values, label='True Values', alpha=0.8, color='blue', marker='o')
        # Scatter plot for predicted values (orange) with connecting lines to true values
        plt.scatter(true_values, pred_k1, label='Predicted Values (K=1)', alpha=0.8, color='orange', marker='x')
        # Draw lines between true and predicted values for clarity
        for true_val, pred_val in zip(true_values, pred_k1):
            plt.plot([true_val, true_val], [true_val, pred_val], color='gray', linestyle='--', alpha=0.6)
        # Diagonal reference line (perfect prediction)
        min_val = min(true_values.min(), pred_k1.min())
        max_val = max(true_values.max(), pred_k1.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=1.5, label='Perfect Prediction')
        # Customize plot with titles, labels, and legend
        #plt.title(f'Predictions vs True {data_type.capitalize()} (K={k_value})', fontsize=14)
        plt.xlabel(f'True {data_type.capitalize()}', fontsize=18)
        plt.ylabel(f'Predicted {data_type.capitalize()}', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(loc="lower right", fontsize=18)

        # Display the plot
        plt.tight_layout()
        plt.show()

    # Plot 4: index
    def plot_latency_true_vs_pred(self, csv_file_path, k_value, data_type):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        # Apply parse_value to the true values to remove 'ms' and convert to numerical values
        df[data_type] = df[data_type].apply(self.parse_value)
        df[f'{data_type}_prediction'] = df[f'{data_type}_prediction'].apply(self.parse_value)
        # Extract true values and predictions
        true_values = df[data_type]
        pred_k1 = df[f'{data_type}_prediction']
        # Plotting the true vs predicted values
        plt.figure(figsize=(8, 5))  # Set a larger figure size
        # Scatter plot for true values (blue)
        new_index1 = range(len(df))
        plt.plot(new_index1, true_values, label='True Values', alpha=0.8, color='blue')
        # Scatter plot for predicted values (orange) with connecting lines to true values
        plt.plot(new_index1, pred_k1, label='Predicted Values ', alpha=0.8, color='orange')

        # Customize plot with titles, labels, and legend
        #plt.title(f'Predictions vs True {data_type.capitalize()} (K={k_value})', fontsize=14)
        plt.xlabel('Observation Index', fontsize=17)
        plt.ylabel('UL / Mbit/s', fontsize=17)
        plt.legend(loc="upper left", fontsize=17)

        # Display the plot
        plt.tight_layout()
        plt.show()
    
    # Plot 5:  error distribution
    def plot_prediction_error_distribution(self, csv_file_path, k_value, data_type):
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        # Apply parse_value to the true values (data_type column) to remove 'ms' and convert to numerical values
        df[data_type] = df[data_type].apply(self.parse_value)
        df[f'{data_type}_prediction'] = df[f'{data_type}_prediction'].apply(self.parse_value)
        # Calculate prediction errors (True - Predicted)
        true_values = df[data_type]
        predicted_values = df[f'{data_type}_prediction']
        prediction_error = true_values - predicted_values
        # Plotting the error distribution
        plt.figure(figsize=(12, 8))  # Increase figure size for better clarity
        # Plot the histogram with more customizations
        plt.hist(prediction_error, bins=20, alpha=0.75, label=f'Error Distribution (K={k_value})', color='skyblue', edgecolor='black')
        # Add a vertical line at zero for clarity (indicating no error)
        plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
        # Add titles, labels, and formatting
        plt.title(f'Prediction Error Distribution for {data_type.capitalize()} (K={k_value})', fontsize=16)
        plt.xlabel('Prediction Error (True - Predicted)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(title="Prediction Model", loc="upper right", fontsize=12)
        # Add grid lines for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        # Customize ticks and labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Display the plot
        plt.tight_layout()
        plt.show()

    ## metrics 
    def calculate_and_print_metrics(self, csv_file_path, k_value, data_type):
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        # Clean and convert the true values (data_type) and predicted values
        true_values = df[data_type].apply(self.parse_value)  # Assuming parse_value cleans 'ms' or other unwanted text
        predicted_values = df[f'{data_type}_prediction']  # Predicted values
        # Calculate Mean and Standard Deviation of errors
        mean_error = np.mean(true_values - predicted_values)
        std_error = np.std(true_values - predicted_values)
        # Calculate MAE and RMSE
        mae = mean_absolute_error(true_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        median_error = np.median(true_values - predicted_values)
        mad_error = np.median(np.abs(true_values - predicted_values - median_error))
        #Print the regression metrics
        print(f'Mean Error: {mean_error:.2f}')
        print(f'Median Error: {median_error:.2f}')
        print(f'Mean Absolute Error (MAE): {mae:.2f}')
        print(f'Median Absolute Deviation (MAD): {mad_error:.2f}')
        print(f'Standard Deviation of Error: {std_error:.2f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

    ## plot 6: classification comparison
    def plot_classification_comparison(self, csv_file_path, k_value, data_type):
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        # Apply parse_value to the true values to remove 'ms' and convert to numerical values
        df[data_type] = df[data_type].apply(self.parse_value)
        df[f'{data_type}_prediction'] = df[f'{data_type}_prediction'].apply(self.parse_value)
        # Classification logic based on the thresholds
        if data_type == 'latency':
            # Classify as 0 (red) if latency > 200ms, else 1 (green)
            df['true_class'] = df[data_type].apply(lambda x: 0 if x <= 200 else 1)
            df['pred_class'] = df[f'{data_type}_prediction'].apply(lambda x: 0 if x <= 200 else 1)
        elif data_type == 'UL':
            # Classify as 0 (red) if uplink < 6, else 1 (green)
            df['true_class'] = df[data_type].apply(lambda x: 0 if x >= 6 else 1)
            df['pred_class'] = df[f'{data_type}_prediction'].apply(lambda x: 0 if x >= 6 else 1)
       # Count the number of red (bad) values for true and predicted classifications
        # Calculate True Positives (both true and predicted are red)
        true_positives = len(df[(df['true_class'] == 1) & (df['pred_class'] == 1)])
        # Calculate False Positives (predicted is red but true is not red)
        false_positives = len(df[(df['true_class'] == 0) & (df['pred_class'] == 1)])
        # Count the number of red (bad) values for true and predicted classifications
        num_true_red = len(df[df['true_class'] == 0])
        num_pred_red = len(df[df['pred_class'] == 0])
        # Plotting
        plt.figure(figsize=(8, 5))
        # True values classification line (green/red)
        plt.plot(df.index, df['true_class'], label=f'True Classification', color='blue', linestyle='-', marker='o')
        # Predicted values classification line (green/red)
        plt.plot(df.index, df['pred_class'], label=f'Predicted Classification ', color='orange', linestyle='--', marker='x')
        # Customize plot with titles, labels, and legend
        #plt.title(f'Prediction vs True Classification ({data_type.capitalize()}) - K={k_value}', fontsize=14)
        plt.xlabel('Observation Index', fontsize=17)
        plt.ylabel('Classification', fontsize=17)
        plt.xticks(fontsize=17)  # Increase x-axis tick label font size
        plt.yticks([0, 1], ['safe', 'not safe'],fontsize=17)
        plt.legend(loc='lower right', fontsize=17)
        print(f'True Positives (correct red predictions): {true_positives}')
        print(f'False Positives (incorrect red predictions): {false_positives}')
        # Display the plot
        plt.tight_layout()
        plt.show()

    ## plot 7: combines plots 3 and 4
    def plot_combined_true_vs_pred(self, csv_file_path, k_value, data_type):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Apply parse_value to the true values to remove 'ms' and convert to numerical values
        df[data_type] = df[data_type].apply(self.parse_value)
        df[f'{data_type}_prediction'] = df[f'{data_type}_prediction'].apply(self.parse_value)
        
        # Extract true values and predictions
        true_values = df[data_type]
        pred_k1 = df[f'{data_type}_prediction']
        
        # Create figure and subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # Two subplots side by side
        
        # Plot 1: True vs Predicted (Scatter plot)
        ax1 = axes[0]
        ax1.scatter(true_values, true_values, label='True Values', alpha=0.8, color='blue', marker='o')
        ax1.scatter(true_values, pred_k1, label='Predicted Values', alpha=0.8, color='orange', marker='x')
        for true_val, pred_val in zip(true_values, pred_k1):
            ax1.plot([true_val, true_val], [true_val, pred_val], color='gray', linestyle='--', alpha=0.6)
        min_val = min(true_values.min(), pred_k1.min())
        max_val = max(true_values.max(), pred_k1.max())
        ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=1.5, label='Perfect Prediction')
        ax1.set_xlabel('True Latency / ms', fontsize=18)
        ax1.set_ylabel('Predicted Latency / ms', fontsize=18)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.set_ylim(0, 2750)
        ax1.set_xlim(0, 2750)   
        ax1.legend(loc="upper left", fontsize=18)
        # Plot 2: True vs Predicted over Index
        ax2 = axes[1]
        new_index1 = range(len(df))
        ax2.plot(new_index1, true_values, label='True Values', alpha=0.8, color='blue')
        ax2.plot(new_index1, pred_k1, label='Predicted Values', alpha=0.8, color='orange')
        ax2.set_xlabel('Observation Index', fontsize=18)
        ax2.set_ylabel(' Latency / ms', fontsize=18)
        ax2.tick_params(axis='both', labelsize=18)
        # ax1.set_ylim(0, 20)
        ax2.set_ylim(0, 2750)
        ax2.set_xlim(0, 2750)   
        ax2.legend(loc="upper left", fontsize=18)
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()    

    ##### Hyperparameter tuning
    def grid_search_optimization(self, csv_file_path, k_range, radius_range, x_range, w1_range, w2_range, w3_range, data_type, output_file):
        # Define parameter grid for k, w_geo, and w_time
        param_grid = {
            'k': k_range,
            'radius': radius_range,
            'x': x_range,
            'w1': w1_range,
            'w2': w2_range,
            'w3': w3_range
        }
        best_std = float('inf') 
        best_mae = float('inf') 
        best_classification_accuracy = 0  
        best_params_mae = None  
        best_params_std = None  
        best_params_classification = None   
        results = []
        # Iterate over all combinations of hyperparameters
        for params in ParameterGrid(param_grid):
            self.k = params['k']
            self.radius = params['radius']
            self.x = params['x']
            self.w1 = params['w1']
            self.w2 = params['w2']
            self.w3 = params['w3']
            # Handle request (this updates the predictions in the CSV file)
            self.handle_request(kpi_type, csv_file_path)
            # Calculate metrics
            df = pd.read_csv(csv_file_path)
            true_values = df[data_type].apply(server.parse_value)
            predicted_values = df[f'{data_type}_prediction']
            # Classification logic based on the thresholds
            if data_type == 'latency':
                df['true_class'] = df[data_type].apply(lambda x: 1 if x <= 200 else 0)
                df['pred_class'] = df[f'{data_type}_prediction'].apply(lambda x: 1 if x <= 200 else 0)
            elif data_type == 'UL':
                df['true_class'] = df[data_type].apply(lambda x: 1 if x >= 6 else 0)
                df['pred_class'] = df[f'{data_type}_prediction'].apply(lambda x: 1 if x >= 6 else 0)
            # Calculate classification accuracy (percentage of correct classifications)
            correct_classifications = df[df['true_class'] == df['pred_class']]
            classification_accuracy = len(correct_classifications) / len(df)    
            # Calculate the standard deviation of the errors
            std_error = np.std(true_values - predicted_values)
            errors = true_values - predicted_values
            median_error = np.median(errors)
            mad_error = np.median(np.abs(errors - median_error))
            mae = mean_absolute_error(true_values, predicted_values)
            # Append the result for this combination
            results.append({
                'k': params['k'],
                'radius': params['radius'],
                'x': params['x'],
                'w1': params['w1'],
                'w2': params['w2'],
                'w3': params['w3'],
                'MAD': mad_error,
                'MAE': mae,
                'STD': std_error,
                'Classification Accuracy': classification_accuracy
            })
            # If this combination has the lowest standard deviation, update best parameters
            if mae < best_mae:
                best_mae = mae 
                best_params_mae = params
            if std_error < best_std:
                best_std = std_error
                best_params_std = params
            if classification_accuracy > best_classification_accuracy:
                best_classification_accuracy = classification_accuracy
                best_params_classification = params
        # Sort results by MAD or STD
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        # Output top 5 best results based on MAe
        results_sorted_by_mae = results_df.sort_values(by='MAE')
        print("Top 5 Best Results Based on MAE:")
        print(results_sorted_by_mae.head(5).to_string(index=False))
        # Output top 5 best results based on STD
        results_sorted_by_std = results_df.sort_values(by='STD')
        print("\nTop 5 Best Results Based on STD:")
        print(results_sorted_by_std.head(5).to_string(index=False))
        # Output best parameters for classification accuracy
        results_sorted_by_classification = results_df.sort_values(by='Classification Accuracy', ascending=False)
        print("\nTop 5 Best Results Based on Classification Accuracy:")
        print(results_sorted_by_classification.head(5).to_string(index=False))
        # Output best parameters and their corresponding standard deviation and MAE
        print(f"Best Params Based on MAE: k={best_params_mae['k']}, radius={best_params_mae['radius']}")
        print(f"Best MAE: {best_mae:.4f}")
        print(f"Best Params Based on STD: k={best_params_std['k']}, radius={best_params_std['radius']}")
        print(f"Best Standard Deviation (STD): {best_std:.4f}")
        print(f"Best Classification Accuracy: {best_classification_accuracy * 100:.2f}%")
        # Optional: Return all results if needed for further analysis
        #return results

if __name__ == "__main__":
    ### insert the path of the hyperparameters file
    server = LocationKPIServiceServer('/Users/mahmoudashri/Desktop/MA_1/MA/9_PC/Hyperparameters.yaml')

    kpi_type = input("Enter the KPI type ('Latency' or 'UL'): ")

    ### insert file containing the data to be predicted(timestamp, latitude, longitude)
    csv_file_path = 'data/difference_prediction/radio_map_train_latency.csv'

    # Process the request (input from CSV file and add predictions to the same file)
    server.handle_request(kpi_type, csv_file_path)

    # # Plots
    # server.plot_error_over_location(csv_file_path, k_value=server.k, data_type=kpi_type)
    # server.plot_prediction_difference(csv_file_path, k_value=server.k, data_type=kpi_type)
    server.plot_true_vs_pred(csv_file_path, k_value=server.k, data_type=kpi_type)
    server.plot_latency_true_vs_pred(csv_file_path, k_value=server.k, data_type=kpi_type)
    server.plot_prediction_error_distribution(csv_file_path, k_value=server.k, data_type=kpi_type)
    server.calculate_and_print_metrics(csv_file_path, k_value=server.k, data_type=kpi_type)
    server.plot_classification_comparison(csv_file_path, k_value=server.k, data_type=kpi_type)
    server.plot_combined_true_vs_pred(csv_file_path, k_value=server.k, data_type=kpi_type)

    ### Hyperparameter tuning
    # Define the range of hyperparameters to test
    k_range = [4, 6, 8, 10, 12, 14, 16, 18, 20]  
    #k_range = [20]
    #radius_range = [100]
    radius_range = [10]
    x_range = [1]
    w1_range = [0.4, 0.6, 0.8, 1.0]
    # w1_range = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    w2_range = [0, 0.01, 0.2, 0.4, 0.6, 0.8]
    # w2_range = [0.50, 0.55, 0.60, 0.65, 0.70]
    w3_range = [0, 0.01, 0.2, 0.4, 0.6, 0.8]
    # w3_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    # w1_range = [1]
    # w2_range = [0.7]
    # w3_range = [0]
    #server.grid_search_optimization(csv_file_path, k_range, radius_range, x_range, w1_range, w2_range, w3_range, data_type=kpi_type, output_file='/Users/mahmoudashri/Desktop/MA_1/MA/6_csv_file/grid_results.csv')
