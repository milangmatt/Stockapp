import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import time
import random
import math
from datetime import datetime, timedelta

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def fetch_stock_data_with_retry(tickers, start_date, end_date, max_retries=3, batch_size=5):
    """
    Fetches stock data with retry logic and rate limiting
    """
    all_data = pd.DataFrame()
    
    # Split tickers into smaller batches
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    
    print("Downloading stock data (this may take a few minutes)...")
    
    for batch_idx, ticker_batch in enumerate(ticker_batches):
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Add random delay between batches (1-3 seconds)
                time.sleep(random.uniform(1, 3))
                
                # Download data for current batch
                batch_data = yf.download(
                    ticker_batch,
                    start=start_date,
                    end=end_date,
                    progress=False
                )['Close']
                
                # If single ticker, reshape the data
                if len(ticker_batch) == 1:
                    batch_data = pd.DataFrame(batch_data)
                    batch_data.columns = ticker_batch
                
                # Merge with existing data
                if all_data.empty:
                    all_data = batch_data
                else:
                    all_data = pd.concat([all_data, batch_data], axis=1)
                
                # Success - break retry loop
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed to download data for batch {ticker_batch} after {max_retries} attempts")
                    print(f"Error: {str(e)}")
                    return None
                
                # Wait longer between retries (3-7 seconds)
                time.sleep(random.uniform(3, 7))
    
    return all_data

def laggedCorr(a, b):
    try:
        a = np.array(a)
        b = np.array(b)
        
        # Remove any NaN values
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]
        
        if len(a) < 2 or len(b) < 2:
            return 0, 0
            
        n = len(a)
        max_lag = min(n // 4, 20)
        
        correlations = []
        lags = []
        
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag < 0:
                    if len(a[-lag:]) < 2 or len(b[:lag]) < 2:
                        correlations.append(0)
                    else:
                        corr = stats.pearsonr(a[-lag:], b[:lag])[0]
                        correlations.append(corr)
                elif lag > 0:
                    if len(a[:-lag]) < 2 or len(b[lag:]) < 2:
                        correlations.append(0)
                    else:
                        corr = stats.pearsonr(a[:-lag], b[lag:])[0]
                        correlations.append(corr)
                else:
                    corr = stats.pearsonr(a, b)[0]
                    correlations.append(corr)
                lags.append(lag)
            except Exception as e:
                print(f"Warning: Error calculating correlation for lag {lag}: {str(e)}")
                correlations.append(0)
                lags.append(lag)
        
        if not correlations:
            return 0, 0
            
        max_corr = max(correlations, key=abs)
        max_lag_index = correlations.index(max_corr)
        
        return max_corr, lags[max_lag_index]
    except Exception as e:
        print(f"Error in laggedCorr: {str(e)}")
        return 0, 0

class TimeSeriesDataset(Dataset):
    """Custom Dataset for PyTorch"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """PyTorch LSTM Model"""
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(25, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LSTMStockPredictor:
    def __init__(self, correlation_threshold=0.5, max_lag=5):
        self.correlation_threshold = correlation_threshold
        self.max_lag = max_lag
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = max_lag
        self.feature_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_features(self, data, target_stock):
        try:
            # Find highly correlated stocks
            correlations = correlation_matrix[target_stock].abs()
            correlated_stocks = correlations[correlations > self.correlation_threshold].index.tolist()
            
            features = pd.DataFrame(index=data.index)
            
            # Add lagged values for target stock
            for lag in range(1, self.max_lag + 1):
                features[f'{target_stock}_lag_{lag}'] = data[target_stock].shift(lag)
                
            # Add lagged values for correlated stocks
            for stock in correlated_stocks:
                if stock != target_stock:
                    lag_value = int(lag_matrix.loc[target_stock, stock])
                    features[f'{stock}_lag_{lag_value}'] = data[stock].shift(lag_value)
                    
            # Add technical indicators
            features[f'{target_stock}_5d_ma'] = data[target_stock].rolling(window=5).mean()
            features[f'{target_stock}_20d_ma'] = data[target_stock].rolling(window=20).mean()
            features[f'{target_stock}_5d_std'] = data[target_stock].rolling(window=5).std()
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            return features
        except Exception as e:
            print(f"Error in create_features: {str(e)}")
            return None
            
    def prepare_data(self, data, target_stock):
        try:
            features = self.create_features(data, target_stock)
            if features is None:
                return None, None
                
            # Create target variable (next day's closing price)
            y = data[target_stock].shift(-1).loc[features.index]
            y = y[:-1]  # Remove last row since we don't have next day's price
            X = features[:-1]  # Remove last row to match y
            
            # Store feature names for later reference
            self.feature_names = X.columns.tolist()
            
            # Handle missing values in target
            mask = ~y.isna()
            y = y[mask]
            X = X[mask]
            
            return X, y
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            return None, None
    
    def create_sequences(self, X, y):
        """Transform data into sequences for LSTM input"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
        
    def train(self, data, target_stock):
        try:
            X, y = self.prepare_data(data, target_stock)
            if X is None or y is None:
                return None
                
            # Scale the features and target
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores_mse = []
            scores_mape = []
            
            # For storing feature importance
            feature_importance = {feature: 1e-6 for feature in X.columns}
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
            
            # Initialize the final model that will be used for predictions
            self.model = LSTMModel(input_dim=X_seq.shape[2]).to(self.device)
            
            for train_idx, val_idx in tscv.split(X_seq):
                X_train, X_val = X_seq[train_idx], X_seq[val_idx]
                y_train, y_val = y_seq[train_idx], y_seq[val_idx]
                
                # Create datasets and dataloaders
                train_dataset = TimeSeriesDataset(X_train, y_train)
                val_dataset = TimeSeriesDataset(X_val, y_val)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32)
                
                # Create a fresh model for each fold
                fold_model = LSTMModel(input_dim=X_train.shape[2]).to(self.device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001)
                
                best_val_loss = float('inf')
                patience = 10
                patience_counter = 0
                
                # Training loop
                for epoch in range(100):
                    fold_model.train()  # Set to training mode
                    train_loss = 0
                    
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = fold_model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validation phase
                    fold_model.eval()  # Set to evaluation mode
                    val_loss = 0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            outputs = fold_model(batch_X)
                            val_loss += criterion(outputs.squeeze(), batch_y).item()
                    
                    val_loss /= len(val_loader)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model weights for this fold
                        best_state = fold_model.state_dict()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                
                # Evaluate the fold
                fold_model.eval()
                with torch.no_grad():
                    val_predictions = fold_model(torch.FloatTensor(X_val).to(self.device))
                    y_pred_scaled = val_predictions.cpu().numpy().flatten()
                    y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    y_true = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                scores_mse.append(root_mean_squared_error(y_true, y_pred))
                scores_mape.append(mean_absolute_percentage_error(y_true, y_pred))
                
                # Update feature importance based on validation performance
                val_loss = best_val_loss if 'best_val_loss' in locals() else float('inf')
                importance_weight = 1.0 / (1.0 + val_loss)
                for feature in X.columns:
                    feature_importance[feature] += importance_weight
            
            # Train final model on all data
            dataset = TimeSeriesDataset(X_seq, y_seq)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Final training
            for epoch in range(100):
                self.model.train()
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
                predictions = self.model(X_tensor)
                y_pred_scaled = predictions.cpu().numpy().flatten()
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_true = self.target_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()
            
            # Calculate overall metrics on all data
            overall_rmse = root_mean_squared_error(y_true, y_pred)
            overall_mape = mean_absolute_percentage_error(y_true, y_pred)
            
            # Normalize feature importance with safety check
            total = sum(feature_importance.values())
            if total > 0:
                feature_importance = {k: v/total for k, v in feature_importance.items()}
            else:
                # If total is still 0, assign equal importance
                n_features = len(feature_importance)
                feature_importance = {k: 1.0/n_features for k in feature_importance.keys()}
            
            return {
                'avg_rmse': np.mean(scores_mse),
                'avg_mape': np.mean(scores_mape),
                'overall_rmse': overall_rmse,
                'overall_mape': overall_mape,
                'feature_importance': feature_importance,
                'actual_values': y[self.sequence_length:],
                'predictions': y_pred,
                'n_features': X.shape[1],
                'n_samples': X.shape[0] - self.sequence_length
            }
        except Exception as e:
            print(f"Error in LSTM training: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

class XGBoostStockPredictor:
    def __init__(self, correlation_threshold=0.5, max_lag=5):
        self.correlation_threshold = correlation_threshold
        self.max_lag = max_lag
        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
    def create_features(self, data, target_stock):
        try:
            # Find highly correlated stocks
            correlations = correlation_matrix[target_stock].abs()
            correlated_stocks = correlations[correlations > self.correlation_threshold].index.tolist()
            
            features = pd.DataFrame(index=data.index)
            
            # Add lagged values for target stock
            for lag in range(1, self.max_lag + 1):
                features[f'{target_stock}_lag_{lag}'] = data[target_stock].shift(lag)
                
            # Add lagged values for correlated stocks
            for stock in correlated_stocks:
                if stock != target_stock:
                    lag_value = int(lag_matrix.loc[target_stock, stock])
                    features[f'{stock}_lag_{lag_value}'] = data[stock].shift(lag_value)
                    
            # Add technical indicators
            features[f'{target_stock}_5d_ma'] = data[target_stock].rolling(window=5).mean()
            features[f'{target_stock}_20d_ma'] = data[target_stock].rolling(window=20).mean()
            features[f'{target_stock}_5d_std'] = data[target_stock].rolling(window=5).std()
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            return features
        except Exception as e:
            print(f"Error in create_features: {str(e)}")
            return None
            
    def prepare_data(self, data, target_stock):
        try:
            features = self.create_features(data, target_stock)
            if features is None:
                return None, None
                
            # Create target variable (next day's closing price)
            y = data[target_stock].shift(-1).loc[features.index]
            y = y[:-1]  # Remove last row since we don't have next day's price
            X = features[:-1]  # Remove last row to match y
            
            # Handle missing values in target
            mask = ~y.isna()
            y = y[mask]
            X = X[mask]
            
            return X, y
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            return None, None
        
    def train(self, data, target_stock):
        try:
            X, y = self.prepare_data(data, target_stock)
            if X is None or y is None:
                return None
                
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores_mse = []
            scores_mape = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                self.model.fit(X_train, y_train)
                predictions = self.model.predict(X_val)
                scores_mse.append(root_mean_squared_error(y_val, predictions))
                scores_mape.append(mean_absolute_percentage_error(y_val, predictions))
            
            # Train final model on all data
            self.model.fit(X_scaled, y)
            
            # Calculate overall RMSE and MAPE on all data
            predictions = self.model.predict(X_scaled)
            overall_rmse = root_mean_squared_error(y, predictions)
            overall_mape = mean_absolute_percentage_error(y, predictions)
            
            return {
                'avg_rmse': np.mean(scores_mse),
                'avg_mape': np.mean(scores_mape),
                'overall_rmse': overall_rmse,
                'overall_mape': overall_mape,
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
                'actual_values': y,
                'predictions': predictions,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }
        except Exception as e:
            print(f"Error in XGBoost training: {str(e)}")
            return None

def save_comparison_plots(results_by_ticker):
    """Create and save separate comparative plots for each metric"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    tickers = list(results_by_ticker.keys())
    metrics = {
        'Overall_RMSE': {'lstm': [], 'xgb': [], 'unit': '₹'},
        'Overall_MAPE': {'lstm': [], 'xgb': [], 'unit': '%'},
        'Average_CV_RMSE': {'lstm': [], 'xgb': [], 'unit': '₹'},
        'Average_CV_MAPE': {'lstm': [], 'xgb': [], 'unit': '%'},
        'Training_Time': {'lstm': [], 'xgb': [], 'unit': 's'}
    }
    
    # Extract metrics and print summary for each ticker
    print("\nDetailed Results Summary")
    print("="*80)
    print(f"{'Stock':<12} {'Model':<8} {'Overall RMSE':<15} {'Overall MAPE':<15} {'Avg CV RMSE':<15} {'Avg CV MAPE':<15} {'Time (s)':<10}")
    print("-"*80)
    
    for ticker in tickers:
        # LSTM results
        lstm_res = results_by_ticker[ticker]['lstm']
        metrics['Overall_RMSE']['lstm'].append(lstm_res['overall_rmse'])
        metrics['Overall_MAPE']['lstm'].append(lstm_res['overall_mape'] * 100)
        metrics['Average_CV_RMSE']['lstm'].append(lstm_res['avg_rmse'])
        metrics['Average_CV_MAPE']['lstm'].append(lstm_res['avg_mape'] * 100)
        metrics['Training_Time']['lstm'].append(lstm_res['training_time'])
        
        # XGBoost results
        xgb_res = results_by_ticker[ticker]['xgb']
        metrics['Overall_RMSE']['xgb'].append(xgb_res['overall_rmse'])
        metrics['Overall_MAPE']['xgb'].append(xgb_res['overall_mape'] * 100)
        metrics['Average_CV_RMSE']['xgb'].append(xgb_res['avg_rmse'])
        metrics['Average_CV_MAPE']['xgb'].append(xgb_res['avg_mape'] * 100)
        metrics['Training_Time']['xgb'].append(xgb_res['training_time'])
        
        # Print results
        print(f"{ticker:<12} {'LSTM':<8} {lstm_res['overall_rmse']:<15.4f} {lstm_res['overall_mape']*100:<15.4f} "
              f"{lstm_res['avg_rmse']:<15.4f} {lstm_res['avg_mape']*100:<15.4f} {lstm_res['training_time']:<10.2f}")
        print(f"{'':<12} {'XGBoost':<8} {xgb_res['overall_rmse']:<15.4f} {xgb_res['overall_mape']*100:<15.4f} "
              f"{xgb_res['avg_rmse']:<15.4f} {xgb_res['avg_mape']*100:<15.4f} {xgb_res['training_time']:<10.2f}")
        print("-"*80)
    
    # Calculate and print average performance
    print("\nOverall Model Performance")
    print("="*50)
    for model in ['lstm', 'xgb']:
        print(f"\n{model.upper()} Average Metrics:")
        for metric, values in metrics.items():
            avg_value = np.mean(values[model])
            std_value = np.std(values[model])
            print(f"{metric.replace('_', ' '):<20}: {avg_value:.4f} ± {std_value:.4f} {values['unit']}")
    
    # Create and save individual plots
    for metric, values in metrics.items():
        # Create figure with specific size
        plt.figure(figsize=(12, 6))
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Ticker': tickers + tickers,
            'Model': ['LSTM']*len(tickers) + ['XGBoost']*len(tickers),
            'Value': values['lstm'] + values['xgb']
        })
        
        # Create plot
        ax = plt.gca()
        sns.lineplot(data=plot_data, x='Ticker', y='Value', hue='Model', marker='o', ax=ax)
        
        # Customize plot
        plt.title(f'{metric.replace("_", " ")} Comparison')
        plt.xlabel('Stock Tickers')
        plt.ylabel(f'{metric.replace("_", " ")} ({values["unit"]})')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'comparison_{metric.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print saved files
    print("\nPlots have been saved as separate files:")
    for metric in metrics.keys():
        print(f"- comparison_{metric.lower()}.png")

def run_all_stocks_analysis():
    """Run the original analysis for all stocks"""
    # ... existing main function code ...
    print("\nPhase 1: Analyzing all stocks...")
    # ==========================================
    # PARAMETERS - MODIFY THESE VALUES AS NEEDED
    # ==========================================
    
    # Date range parameters
    # Format: 'YYYY-MM-DD'
    # Example: '2023-04-01'
    start_date = '2024-04-01'  # Replace with your desired start date
    end_date = '2025-04-01'    # Replace with your desired end date
    
    # Model parameters
    correlation_threshold = 0.75  # Value between 0.0 and 1.0
    max_lag = 10                  # Integer value (recommended: 1-20)
    
    # List of stock tickers
    tickers = [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'TATAMOTORS.NS', 'TCS.NS',
        'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'HINDUNILVR.NS',
        'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'RELIANCE.NS',
        'NTPC.NS', 'TATAPOWER.NS', 'TITAN.NS'
    ]
    
    # Target stock to predict
    target_stock = 'HDFCBANK.NS'  # Replace with your desired stock from the list above
    
    print("\nRunning analysis with the following parameters:")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Correlation Threshold: {correlation_threshold}")
    print(f"Maximum Lag: {max_lag}")
    print(f"Target Stock: {target_stock}")
    print("-" * 50)
    
    try:
        # Download data once for all tickers
        print("Downloading stock data (this may take a few minutes)...")
        data = fetch_stock_data_with_retry(tickers, start_date, end_date)
        if data is None or data.empty:
            print("Failed to download stock data. Please try again later.")
            return None
            
        # Calculate correlations once
        print("Calculating correlations between stocks...")
        results = []
        for i in range(len(data.columns)):
            for j in range(i + 1, len(data.columns)):
                stock_a = data.iloc[:, i]
                stock_b = data.iloc[:, j]
                corr, lag = laggedCorr(stock_a.values, stock_b.values)
                results.append((data.columns[i], data.columns[j], corr, lag))
        
        # Create correlation and lag matrices
        global correlation_matrix, lag_matrix
        correlation_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        lag_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
        
        for stock_a, stock_b, corr, lag in results:
            correlation_matrix.loc[stock_a, stock_b] = corr
            correlation_matrix.loc[stock_b, stock_a] = corr
            lag_matrix.loc[stock_a, stock_b] = lag
            lag_matrix.loc[stock_b, stock_a] = lag
        
        correlation_matrix = correlation_matrix.astype(float)
        lag_matrix = lag_matrix.astype(float)
        
        # Store results for all tickers
        results_by_ticker = {}
        
        # Process each ticker
        for target_stock in tickers:
            print(f"\n{'='*50}")
            print(f"Processing {target_stock}")
            print('='*50)
            
            # Train LSTM
            print("\nTraining LSTM model...")
            lstm_start_time = time.time()
            lstm_predictor = LSTMStockPredictor(
                correlation_threshold=correlation_threshold,
                max_lag=max_lag
            )
            lstm_results = lstm_predictor.train(data, target_stock)
            lstm_training_time = time.time() - lstm_start_time
            
            if lstm_results:
                lstm_results['training_time'] = lstm_training_time
            
            # Train XGBoost
            print("\nTraining XGBoost model...")
            xgb_start_time = time.time()
            xgb_predictor = XGBoostStockPredictor(
                correlation_threshold=correlation_threshold,
                max_lag=max_lag
            )
            xgb_results = xgb_predictor.train(data, target_stock)
            xgb_training_time = time.time() - xgb_start_time
            
            if xgb_results:
                xgb_results['training_time'] = xgb_training_time
            
            # Store results for this ticker
            if lstm_results and xgb_results:
                results_by_ticker[target_stock] = {
                    'lstm': lstm_results,
                    'xgb': xgb_results
                }
        
        # Create and save comparison plots
        if results_by_ticker:
            print("\nGenerating comparison plots and summary...")
            save_comparison_plots(results_by_ticker)
        
        return data  # Return the downloaded data
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def run_correlation_analysis(data):
    """Analyze ICICIBANK.NS with varying correlation thresholds"""
    if data is None:
        print("No data available for correlation analysis")
        return
        
    print("\nPhase 2: Running correlation analysis for ICICIBANK.NS...")
    target_stock = 'ICICIBANK.NS'
    fixed_lag = 10
    correlation_values = np.arange(0.05, 1.00, 0.05)
    
    # Create results dictionary for different correlation values
    correlation_results = {}
    
    for corr_threshold in correlation_values:
        print(f"\nTesting correlation threshold: {corr_threshold:.2f}")
        
        # Train LSTM with timing
        lstm_start_time = time.time()
        lstm_predictor = LSTMStockPredictor(
            correlation_threshold=corr_threshold,
            max_lag=fixed_lag
        )
        lstm_results = lstm_predictor.train(data, target_stock)
        lstm_training_time = time.time() - lstm_start_time
        
        # Train XGBoost with timing
        xgb_start_time = time.time()
        xgb_predictor = XGBoostStockPredictor(
            correlation_threshold=corr_threshold,
            max_lag=fixed_lag
        )
        xgb_results = xgb_predictor.train(data, target_stock)
        xgb_training_time = time.time() - xgb_start_time
        
        # Add training times to results
        if lstm_results:
            lstm_results['training_time'] = lstm_training_time
        if xgb_results:
            xgb_results['training_time'] = xgb_training_time
        
        # Store results
        correlation_results[corr_threshold] = {
            'lstm': lstm_results,
            'xgb': xgb_results
        }
    
    # Create correlation analysis plots
    create_correlation_analysis_plots(correlation_results)

def run_lag_analysis(data):
    """Analyze ICICIBANK.NS with varying lag values"""
    if data is None:
        print("No data available for lag analysis")
        return
        
    print("\nPhase 3: Running lag analysis for ICICIBANK.NS...")
    target_stock = 'ICICIBANK.NS'
    fixed_correlation = 0.75
    lag_values = range(1, 21)
    
    # Create results dictionary for different lag values
    lag_results = {}
    
    for lag in lag_values:
        print(f"\nTesting lag value: {lag}")
        
        # Train LSTM with timing
        lstm_start_time = time.time()
        lstm_predictor = LSTMStockPredictor(
            correlation_threshold=fixed_correlation,
            max_lag=lag
        )
        lstm_results = lstm_predictor.train(data, target_stock)
        lstm_training_time = time.time() - lstm_start_time
        
        # Train XGBoost with timing
        xgb_start_time = time.time()
        xgb_predictor = XGBoostStockPredictor(
            correlation_threshold=fixed_correlation,
            max_lag=lag
        )
        xgb_results = xgb_predictor.train(data, target_stock)
        xgb_training_time = time.time() - xgb_start_time
        
        # Add training times to results
        if lstm_results:
            lstm_results['training_time'] = lstm_training_time
        if xgb_results:
            xgb_results['training_time'] = xgb_training_time
        
        # Store results
        lag_results[lag] = {
            'lstm': lstm_results,
            'xgb': xgb_results
        }
        
        # Print progress
        print(f"Completed lag {lag}: LSTM time = {lstm_training_time:.2f}s, XGBoost time = {xgb_training_time:.2f}s")
    
    # Create lag analysis plots
    create_lag_analysis_plots(lag_results)

def create_correlation_analysis_plots(results):
    """Create plots for correlation analysis"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Define metrics with their units and labels
    metric_info = {
        'overall_rmse': {'unit': '₹', 'label': 'Overall RMSE'},
        'overall_mape': {'unit': '%', 'label': 'Overall MAPE'},
        'avg_rmse': {'unit': '₹', 'label': 'Average CV RMSE'},
        'avg_mape': {'unit': '%', 'label': 'Average CV MAPE'},
        'training_time': {'unit': 's', 'label': 'Training Time'}
    }
    
    for metric, info in metric_info.items():
        plt.figure(figsize=(12, 6))
        
        # Extract values with error handling
        correlations = list(results.keys())
        lstm_values = []
        xgb_values = []
        
        for c in correlations:
            # Handle potential missing results
            if results[c]['lstm'] and results[c]['xgb']:
                if metric in ['overall_mape', 'avg_mape']:
                    # Convert MAPE to percentage
                    lstm_values.append(results[c]['lstm'][metric] * 100)
                    xgb_values.append(results[c]['xgb'][metric] * 100)
                else:
                    lstm_values.append(results[c]['lstm'][metric])
                    xgb_values.append(results[c]['xgb'][metric])
            else:
                lstm_values.append(np.nan)
                xgb_values.append(np.nan)
        
        # Create plot with better styling
        sns.set_style("whitegrid")
        plt.plot(correlations, lstm_values, 'b-o', label='LSTM', linewidth=2, markersize=8)
        plt.plot(correlations, xgb_values, 'r-o', label='XGBoost', linewidth=2, markersize=8)
        
        plt.title(f'Effect of Correlation Threshold on {info["label"]}', fontsize=14)
        plt.xlabel('Correlation Threshold', fontsize=12)
        plt.ylabel(f'{info["label"]} ({info["unit"]})', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'correlation_analysis_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_lag_analysis_plots(results):
    """Create plots for lag analysis"""
    import matplotlib.pyplot as plt
    import seaborn as sns  # For better plot styling
    
    metrics = ['overall_rmse', 'overall_mape', 'avg_rmse', 'avg_mape', 'training_time']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Extract values
        lags = list(results.keys())
        lstm_values = [results[lag]['lstm'][metric] for lag in lags]
        xgb_values = [results[lag]['xgb'][metric] for lag in lags]
        
        # Create plot with better styling
        sns.set_style("whitegrid")
        plt.plot(lags, lstm_values, 'b-o', label='LSTM', linewidth=2, markersize=8)
        plt.plot(lags, xgb_values, 'r-o', label='XGBoost', linewidth=2, markersize=8)
        
        plt.title(f'Effect of Lag Value on {metric.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Lag (days)', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'lag_analysis_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Phase 1: Original analysis with all stocks
    data = run_all_stocks_analysis()
    
    if data is not None:
        # Phase 2: Correlation analysis for ICICIBANK.NS
        run_correlation_analysis(data)
        
        # Phase 3: Lag analysis for ICICIBANK.NS
        run_lag_analysis(data)
    else:
        print("Analysis terminated due to data loading failure")

if __name__ == "__main__":
    main()
