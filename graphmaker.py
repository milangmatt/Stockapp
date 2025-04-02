import numpy as np
import pandas as pd
import yfinance as yf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define the tickers
tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
    'MARUTI.NS', 'TATAMOTORS.NS', 'TCS.NS',
    'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'RELIANCE.NS',
    'NTPC.NS', 'TATAPOWER.NS', 'TITAN.NS'
]

# Fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

# Define a custom Dataset for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Train LSTM function
def train_lstm(data, target_stock, correlation_threshold, max_lag):
    # Create features and target
    features = create_features(data, target_stock, correlation_threshold, max_lag)
    y = data[target_stock].shift(-1).dropna()
    X = features[:-1].dropna()
    
    # Prepare dataset
    dataset = TimeSeriesDataset(X.values, y.values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = LSTMModel(input_dim=X.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.unsqueeze(1).to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_y = batch_y.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X.values).unsqueeze(1).to('cuda' if torch.cuda.is_available() else 'cpu')
        predictions = model(X_tensor).cpu().numpy().flatten()
    
    return {
        'actual_values': y.values,
        'predictions': predictions
    }

# Create features function (to be defined based on your requirements)
def create_features(data, target_stock, correlation_threshold, max_lag):
    # This function should create lagged features based on correlation
    # For simplicity, let's just create lagged features for the target stock
    features = pd.DataFrame(index=data.index)
    for lag in range(1, max_lag + 1):
        features[f'{target_stock}_lag_{lag}'] = data[target_stock].shift(lag)
    return features.fillna(method='bfill')

# Train XGBoost function (placeholder)
def train_xgboost(data, target_stock, correlation_threshold, max_lag):
    # Implement the XGBoost training logic here
    # This is a placeholder function
    return {
        'actual_values': np.random.rand(100),  # Replace with actual values
        'predictions': np.random.rand(100)      # Replace with predictions
    }

# Define a function to analyze performance
def analyze_performance(data, model_type, correlation_threshold, max_lag):
    results = []
    
    for ticker in data.columns:
        if model_type == 'LSTM':
            start_time = time.time()
            lstm_results = train_lstm(data, ticker, correlation_threshold, max_lag)
            end_time = time.time()
            rmse = root_mean_squared_error(lstm_results['actual_values'], lstm_results['predictions'])
            mape = mean_absolute_percentage_error(lstm_results['actual_values'], lstm_results['predictions'])
            results.append((ticker, rmse, mape, end_time - start_time))
        elif model_type == 'XGBoost':
            start_time = time.time()
            xgb_results = train_xgboost(data, ticker, correlation_threshold, max_lag)
            end_time = time.time()
            rmse = root_mean_squared_error(xgb_results['actual_values'], xgb_results['predictions'])
            mape = mean_absolute_percentage_error(xgb_results['actual_values'], xgb_results['predictions'])
            results.append((ticker, rmse, mape, end_time - start_time))
    
    return results

# Main analysis function
def main_analysis():
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = fetch_stock_data(tickers, start_date, end_date)
    
    print("Starting LSTM Analysis...")
    lstm_results = analyze_performance(data, 'LSTM', correlation_threshold=0.75, max_lag=10)
    
    print("Starting XGBoost Analysis...")
    xgb_results = analyze_performance(data, 'XGBoost', correlation_threshold=0.75, max_lag=10)
    
    print("LSTM Results:")
    for result in lstm_results:
        print(f"Ticker: {result[0]}, RMSE: {result[1]:.2f}, MAPE: {result[2]:.2%}, Time Taken: {result[3]:.2f} seconds")
    
    print("XGBoost Results:")
    for result in xgb_results:
        print(f"Ticker: {result[0]}, RMSE: {result[1]:.2f}, MAPE: {result[2]:.2%}, Time Taken: {result[3]:.2f} seconds")
    
    # Generate graphs
    plt.figure(figsize=(12, 6))
    
    # LSTM results
    lstm_rmse = [result[1] for result in lstm_results]
    lstm_mape = [result[2] for result in lstm_results]
    lstm_time = [result[3] for result in lstm_results]
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_time, lstm_rmse, label='LSTM RMSE', color='blue')
    plt.plot(lstm_time, lstm_mape, label='LSTM MAPE', color='orange')
    plt.title('LSTM Performance')
    plt.xlabel('Time Taken (seconds)')
    plt.ylabel('Error Metrics')
    plt.legend()
    
    # XGBoost results
    xgb_rmse = [result[1] for result in xgb_results]
    xgb_mape = [result[2] for result in xgb_results]
    xgb_time = [result[3] for result in xgb_results]
    
    plt.subplot(1, 2, 2)
    plt.plot(xgb_time, xgb_rmse, label='XGBoost RMSE', color='green')
    plt.plot(xgb_time, xgb_mape, label='XGBoost MAPE', color='red')
    plt.title('XGBoost Performance')
    plt.xlabel('Time Taken (seconds)')
    plt.ylabel('Error Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    main_analysis()