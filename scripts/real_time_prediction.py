import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Use absolute paths to the model and scaler files
model_path = os.path.abspath('models/stock_price_predictor_lr.pkl')
scaler_path = os.path.abspath('models/scaler.pkl')

# Load the saved model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialize lists to store actual and predicted values
actual_values = []
predicted_values = []
timestamps = []

# Function to fetch the last hour of minute-by-minute data
def fetch_last_hour_data(ticker):
    # Fetch the last 60 minutes of stock data
    stock_data = yf.download(ticker, period='1d', interval='1m')
    
    # Ensure there is enough data
    if len(stock_data) < 60:
        raise ValueError("Not enough data fetched. Ensure the market is open or the ticker is correct.")
    
    # Get the last 60 minutes of data
    last_hour_data = stock_data.iloc[-60:]
    
    return last_hour_data

# Function to fetch and preprocess the latest minute data
def fetch_and_preprocess_data(ticker):
    # Fetch the latest minute-by-minute stock data
    stock_data = yf.download(ticker, period='1d', interval='1m')
    
    # Ensure there is enough data to make predictions
    if stock_data.empty:
        raise ValueError("No data fetched. Ensure the market is open or the ticker is correct.")
    
    # Get the actual latest close price
    actual_close = stock_data.iloc[-1]['Close']
    actual_time = stock_data.index[-1]
    
    # Feature engineering
    stock_data['50MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['100MA'] = stock_data['Close'].rolling(window=100).mean()
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()
    stock_data = stock_data.dropna()
    
    # Select the latest row
    latest_data = stock_data.iloc[-1]

    # Prepare the features
    features = np.array([[
        latest_data['Close'],
        latest_data['50MA'],
        latest_data['100MA'],
        latest_data['Return'],
        latest_data['Volatility']
    ]])
    
    # Scale the features
    features_scaled = scaler.transform(features)

    return features_scaled, actual_close, actual_time

# Function to make predictions
def make_prediction(ticker):
    # Fetch and preprocess the latest data
    features_scaled, actual_close, actual_time = fetch_and_preprocess_data(ticker)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return prediction[0], actual_close, actual_time

# Initialize the plot with Seaborn and customize the style
sns.set(style='darkgrid')
plt.ion()
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
line1, = ax.plot([], [], color='red', label='Predicted Next Minute')
line2, = ax.plot([], [], color='blue', label='Actual')
ax.legend(facecolor='black', framealpha=1, edgecolor='white', fontsize='large')
plt.xlabel('Time', color='white')
plt.ylabel('Price', color='white')
plt.title('Real-Time Stock Price Prediction', color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

def update_plot(predicted, actual, timestamp):
    # Append new values to the lists
    predicted_values.append(predicted)
    actual_values.append(actual)
    timestamps.append(timestamp)
    
    # Keep only the last 60 values (past hour, with one value per minute)
    if len(actual_values) > 60:
        actual_values.pop(0)
        predicted_values.pop(0)
        timestamps.pop(0)
    
    # Update the data of the plot
    line1.set_xdata(range(len(predicted_values)))
    line1.set_ydata(predicted_values)
    line2.set_xdata(range(len(actual_values)))
    line2.set_ydata(actual_values)
    
    # Adjust plot limits
    ax.relim()
    ax.autoscale_view()
    
    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

# Main loop to fetch, preprocess, and predict next minute's closing prices
if __name__ == "__main__":
    ticker = 'TSLA'
    
    # Fetch and plot the last hour of data initially
    try:
        last_hour_data = fetch_last_hour_data(ticker)
        actual_values = last_hour_data['Close'].tolist()
        predicted_values = [None] * len(actual_values)  # Initialize predicted values with None
        timestamps = last_hour_data.index.tolist()
        
        line2.set_xdata(range(len(actual_values)))
        line2.set_ydata(actual_values)
        
        ax.relim()
        ax.autoscale_view()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    except ValueError as e:
        print(e)
    
    while True:
        try:
            prediction, actual_close, actual_time = make_prediction(ticker)
            print(f"Predicted closing price for next minute for {ticker}: ${prediction:.2f}, Actual closing price now: ${actual_close:.2f}")
            
            # Update the plot with new data
            update_plot(prediction, actual_close, actual_time)
        except ValueError as e:
            print(e)
        
        time.sleep(60)  # Wait for 1 minute before the next prediction
