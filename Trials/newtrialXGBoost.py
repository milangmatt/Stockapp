import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import time
import random
from datetime import datetime, timedelta

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

class StockPricePredictor:
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
            print(f"Error in training: {str(e)}")
            return None

def main():
    # ==========================================
    # PARAMETERS - MODIFY THESE VALUES AS NEEDED
    # ==========================================
    
    # Date range parameters
    # Format: 'YYYY-MM-DD'
    # Example: '2023-04-01'
    # ==========================================
    # FILL IN YOUR START DATE HERE
    start_date = '2024-04-01'  # Replace with your desired start date
    
    # FILL IN YOUR END DATE HERE
    end_date = '2025-04-01'    # Replace with your desired end date
    # ==========================================
    
    # Model parameters
    # ==========================================
    # FILL IN YOUR CORRELATION THRESHOLD HERE
    correlation_threshold = 0.75  # Value between 0.0 and 1.0
    
    # FILL IN YOUR MAXIMUM LAG HERE
    max_lag = 10                  # Integer value (recommended: 1-20)
    # ==========================================
    
    # List of stock tickers
    tickers = [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'TATAMOTORS.NS', 'TCS.NS',
        'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'HINDUNILVR.NS',
        'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'RELIANCE.NS',
        'NTPC.NS', 'TATAPOWER.NS', 'TITAN.NS'
    ]
    
    # Target stock to predict
    # ==========================================
    # FILL IN YOUR TARGET STOCK HERE
    target_stock = 'HDFCBANK.NS'  # Replace with your desired stock from the list above
    # ==========================================
    
    print("\nRunning analysis with the following parameters:")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Correlation Threshold: {correlation_threshold}")
    print(f"Maximum Lag: {max_lag}")
    print(f"Target Stock: {target_stock}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Download data
        data = fetch_stock_data_with_retry(tickers, start_date, end_date)
        if data is None or data.empty:
            print("Failed to download stock data. Please try again later.")
            return
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            print("Some stocks have missing values. They will be handled automatically.")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        download_time = time.time()
        
        # Calculate correlations
        print("Downloading stock data (this may take a few minutes)...")
        results = []
        total_pairs = len(data.columns) * (len(data.columns) - 1) // 2
        current_pair = 0
        
        for i in range(len(data.columns)):
            for j in range(i + 1, len(data.columns)):
                stock_a = data.iloc[:, i]
                stock_b = data.iloc[:, j]
                corr, lag = laggedCorr(stock_a.values, stock_b.values)
                results.append((data.columns[i], data.columns[j], corr, lag))
                current_pair += 1
        
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
        
        correlation_time = time.time()
        
        # Train model and make predictions
        predictor = StockPricePredictor(
            correlation_threshold=correlation_threshold,
            max_lag=max_lag
        )
        results = predictor.train(data, target_stock)
        
        if results is None:
            print("Error during model training. Please try different parameters.")
            return
        
        model_time = time.time()
        
        # Display final metrics
        print("\n" + "="*50)
        print(f"SUMMARY FOR {target_stock}")
        print("="*50)
        print(f"Cross-validation (Time Series Split):")
        print(f"Average RMSE: ₹{results['avg_rmse']:.4f}")
        print(f"Average MAPE: {results['avg_mape']:.4%}")
        print("\nFull Dataset Metrics:")
        print(f"Overall RMSE: ₹{results['overall_rmse']:.4f}")
        print(f"Overall MAPE: {results['overall_mape']:.4%}")
        print("\nModel Information:")
        print(f"Number of features used: {results['n_features']}")
        print(f"Number of samples: {results['n_samples']}")
        print("\nTop 5 most important features:")
        sorted_features = dict(sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        for feature, importance in sorted_features.items():
            print(f"- {feature}: {importance:.4f}")
        
        print("\nTime Analysis:")
        print(f"Correlation calculation time: {correlation_time - download_time:.2f} seconds")
        print(f"Model training time: {model_time - correlation_time:.2f} seconds")
        print(f"Total execution time: {model_time - download_time:.2f} seconds")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Please try again with different parameters.")

if __name__ == "__main__":
    main()