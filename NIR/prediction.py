import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tabulate import tabulate

# Example Data
dates = pd.date_range(start="2017-01-01 00:00:00", end="2017-01-07 23:55:00", freq="5min")
nodes = [f"{400000 + i}" for i in range(1, 326)]  # 325 nodes
data = pd.DataFrame(data=np.random.rand(len(dates), len(nodes)) * 100, index=dates, columns=nodes)

# Train/Test Split
train_data = data.iloc[:2016]
test_indices = [100, 200]  # Example test nodes

# Scaling Data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)

# Historical Average (HA)
def historical_average(train_data, test_indices):
    avg_speeds = train_data.mean(axis=0)
    predictions = avg_speeds.iloc[test_indices]
    return predictions

# ARIMA
def arima_forecast(train_data, test_indices, order=(1, 0, 0)):
    predictions = []
    for idx in test_indices:
        series = train_data.iloc[:, idx]
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predictions.append(forecast[0])
    return np.array(predictions)

# Persistence Model
def persistence_model(train_data, test_indices):
    last_values = train_data.iloc[-1, :]
    predictions = last_values.iloc[test_indices]
    return predictions

# Linear Regression
def linear_regression_forecast(train_data, test_indices, lookback=12):
    predictions = []
    for idx in test_indices:
        series = train_data.iloc[:, idx].values
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i-lookback:i])
            y.append(series[i])
        X, y = np.array(X), np.array(y)
        
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([series[-lookback:]])
        predictions.append(prediction[0])
    return np.array(predictions)

# Exponential Smoothing
def exponential_smoothing(train_data, test_indices):
    predictions = []
    for idx in test_indices:
        series = train_data.iloc[:, idx]
        model = SimpleExpSmoothing(series)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        predictions.append(forecast[0])
    return np.array(predictions)

# Predictions
ha_predictions = historical_average(train_data, test_indices)
arima_predictions = arima_forecast(train_data, test_indices)
persistence_predictions = persistence_model(train_data, test_indices)
linear_predictions = linear_regression_forecast(train_data, test_indices)
exp_smooth_predictions = exponential_smoothing(train_data, test_indices)

# True values (example, replace with actual test set values if available)
true_values = [15, 0]  # Example for test nodes

# Calculate MSE
mse_ha = mean_squared_error(true_values, ha_predictions)
mse_arima = mean_squared_error(true_values, arima_predictions)
mse_persistence = mean_squared_error(true_values, persistence_predictions)
mse_linear = mean_squared_error(true_values, linear_predictions)
mse_exp = mean_squared_error(true_values, exp_smooth_predictions)

# Results Table
results = [
    ["HA", mse_ha],
    ["ARIMA", mse_arima],
    ["Persistence", mse_persistence],
    ["Linear Regression", mse_linear],
    ["Exponential Smoothing", mse_exp],
]

print(tabulate(results, headers=["Model", "MSE"], floatfmt=".4f"))
