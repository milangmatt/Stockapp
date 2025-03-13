import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

def fetch_stock_data_with_retry(tickers, start_date, end_date, max_retries=3, batch_size=5):
    """
    Fetches stock data with retry logic and rate limiting
    """
    all_data = pd.DataFrame()
    
    # Split tickers into smaller batches
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    
    with st.spinner("Downloading stock data (this may take a few minutes)..."):
        progress_bar = st.progress(0)
        
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
                    
                    # Update progress
                    progress = (batch_idx + 1) / len(ticker_batches)
                    progress_bar.progress(progress)
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        st.error(f"Failed to download data for batch {ticker_batch} after {max_retries} attempts")
                        st.error(f"Error: {str(e)}")
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
                st.warning(f"Error calculating correlation for lag {lag}: {str(e)}")
                correlations.append(0)
                lags.append(lag)
        
        if not correlations:
            return 0, 0
            
        max_corr = max(correlations, key=abs)
        max_lag_index = correlations.index(max_corr)
        
        return max_corr, lags[max_lag_index]
    except Exception as e:
        st.error(f"Error in laggedCorr: {str(e)}")
        return 0, 0

def validate_predictions(actual_prices, predicted_prices, stock_name):
    """
    Validates prediction accuracy and returns detailed metrics
    """
    # Calculate basic metrics
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
    rmse = root_mean_squared_error(actual_prices, predicted_prices)
    
    # Calculate directional accuracy
    actual_direction = np.diff(actual_prices) > 0
    predicted_direction = np.diff(predicted_prices) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction)
    
    # Calculate error distribution
    errors = predicted_prices - actual_prices
    error_percentiles = np.percentile(errors, [5, 25, 50, 75, 95])
    
    # Calculate prediction bounds
    std_error = np.std(errors)
    upper_bound = predicted_prices + (2 * std_error)
    lower_bound = predicted_prices - (2 * std_error)
    within_bounds = np.mean((actual_prices >= lower_bound) & 
                           (actual_prices <= upper_bound))
    
    # Visualization
    fig = go.Figure()
    
    # Actual vs Predicted prices
    fig.add_trace(go.Scatter(y=actual_prices, name="Actual Price",
                            line=dict(color='yellow')))
    fig.add_trace(go.Scatter(y=predicted_prices, name="Predicted Price",
                            line=dict(color='red')))
    
    # Prediction bounds
    fig.add_trace(go.Scatter(y=upper_bound, name="Upper Bound",
                            line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(y=lower_bound, name="Lower Bound",
                            line=dict(color='gray', dash='dash')))
    
    fig.update_layout(title=f"Price Prediction Validation for {stock_name}",
                     xaxis_title="Time",
                     yaxis_title="Price (₹)",
                     height=500)
    
    # Error distribution plot
    error_fig = go.Figure()
    error_fig.add_trace(go.Histogram(x=errors, nbinsx=50,
                                    name="Prediction Errors"))
    error_fig.update_layout(title="Distribution of Prediction Errors",
                           xaxis_title="Error Amount (₹)",
                           yaxis_title="Frequency",
                           height=400)
    
    # Return all metrics and plots
    metrics = {
        "MAPE": f"{mape:.2%}",
        "RMSE": f"₹{rmse:.2f}",
        "Directional Accuracy": f"{directional_accuracy:.2%}",
        "Within Bounds": f"{within_bounds:.2%}",
        "Error Percentiles": {
            "5th": f"₹{error_percentiles[0]:.2f}",
            "25th": f"₹{error_percentiles[1]:.2f}",
            "Median": f"₹{error_percentiles[2]:.2f}",
            "75th": f"₹{error_percentiles[3]:.2f}",
            "95th": f"₹{error_percentiles[4]:.2f}"
        }
    }
    
    return metrics, fig, error_fig

def display_validation(actual_prices, predicted_prices, stock_name):
    metrics, price_plot, error_plot = validate_predictions(
        actual_prices, predicted_prices, stock_name
    )
    
    # Display metrics
    st.subheader("Prediction Accuracy Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MAPE", metrics["MAPE"])
        st.metric("RMSE", metrics["RMSE"])
    
    with col2:
        st.metric("Directional Accuracy", metrics["Directional Accuracy"])
        st.metric("Within Bounds", metrics["Within Bounds"])
    
    with col3:
        st.subheader("Error Percentiles")
        for name, value in metrics["Error Percentiles"].items():
            st.text(f"{name}: {value}")
    
    # Display plots
    st.plotly_chart(price_plot)
    st.plotly_chart(error_plot)
    
    # Additional analysis
    if float(metrics["MAPE"].strip('%')) < 5:
        st.success("Model predictions are highly accurate (MAPE < 5%)")
    elif float(metrics["MAPE"].strip('%')) < 10:
        st.info("Model predictions are reasonably accurate (MAPE < 10%)")
    else:
        st.warning("Model predictions have high error rates (MAPE > 10%)")
    
    if float(metrics["Directional Accuracy"].strip('%')) > 60:
        st.success("Model is good at predicting price direction (>60% accuracy)")
    else:
        st.warning("Model struggles with directional prediction (<60% accuracy)")

def backtest_strategy(data, predictions, actual_prices, target_stock, investment=100000):
    """
    Simulates a simple trading strategy based on predictions
    """
    portfolio_value = [investment]
    holdings = 0
    cash = investment
    
    # Keep track of trades for reporting
    trades = []
    
    for i in range(1, len(predictions)):
        current_price = actual_prices[i-1]
        next_price = actual_prices[i]
        predicted_direction = predictions[i] > current_price
        
        # Simple strategy: Buy if predicted to go up, sell if predicted to go down
        if predicted_direction and cash > 0:
            # Buy as many shares as possible
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                holdings += shares_to_buy
                cash -= shares_to_buy * current_price
                trades.append(f"Day {i}: Bought {shares_to_buy} shares at ₹{current_price:.2f}")
        elif not predicted_direction and holdings > 0:
            # Sell all holdings
            trade_value = holdings * current_price
            trades.append(f"Day {i}: Sold {holdings} shares at ₹{current_price:.2f}")
            cash += trade_value
            holdings = 0
        
        # Calculate portfolio value
        portfolio_value.append(cash + holdings * next_price)
    
    # Calculate metrics
    returns = (portfolio_value[-1] - investment) / investment
    buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0]
    
    return {
        "portfolio_value": portfolio_value,
        "returns": returns,
        "buy_hold_return": buy_hold_return,
        "final_value": portfolio_value[-1],
        "trades": trades
    }

def display_backtest_results(backtest_results, investment, target_stock):
    """
    Displays the results of the backtesting in the Streamlit UI
    """
    st.subheader("Trading Strategy Backtest")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Strategy Return", 
            f"{backtest_results['returns']:.2%}",
            f"{backtest_results['returns'] - backtest_results['buy_hold_return']:.2%}"
        )
    
    with col2:
        st.metric(
            "Buy & Hold Return", 
            f"{backtest_results['buy_hold_return']:.2%}"
        )
    
    with col3:
        st.metric(
            "Final Portfolio Value", 
            f"₹{backtest_results['final_value']:,.2f}",
            f"₹{backtest_results['final_value'] - investment:,.2f}"
        )
    
    # Plot portfolio value over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=backtest_results['portfolio_value'],
        name="Portfolio Value",
        line=dict(color='green')
    ))
    
    # Add buy & hold performance
    buy_hold_values = [
        investment * (1 + backtest_results['buy_hold_return'] * i / (len(backtest_results['portfolio_value'])-1))
        for i in range(len(backtest_results['portfolio_value']))
    ]
    
    fig.add_trace(go.Scatter(
        y=buy_hold_values,
        name="Buy & Hold",
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Backtested Performance: {target_stock}",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value (₹)",
        height=400
    )
    
    st.plotly_chart(fig)
    
    # Show sample of trades
    with st.expander("View Trading Activity"):
        if len(backtest_results['trades']) > 0:
            for trade in backtest_results['trades'][:10]:  # Show first 10 trades
                st.text(trade)
            
            if len(backtest_results['trades']) > 10:
                st.text(f"... and {len(backtest_results['trades']) - 10} more trades")
        else:
            st.text("No trades were executed during the backtest period")

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
            st.error(f"Error in create_features: {str(e)}")
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
            st.error(f"Error in prepare_data: {str(e)}")
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
            
            return {
                'avg_rmse': np.mean(scores_mse),
                'avg_mape': np.mean(scores_mape),
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
                'actual_values': y,
                'predictions': self.model.predict(X_scaled)
            }
        except Exception as e:
            st.error(f"Error in training: {str(e)}")
            return None
    
    def predict(self, data, target_stock):
        try:
            X, _ = self.prepare_data(data, target_stock)
            if X is None:
                return None
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

# Set page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title and description
st.title("Stock Price Prediction using Lagged Correlations")
st.markdown("""
This application predicts stock prices using machine learning and lagged correlations between different stocks.
Select your parameters and target stock to start the prediction process.
""")

# Sidebar for parameters
st.sidebar.header("Parameters")

# Date range selector
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=365)
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.now()
)

# Model parameters
correlation_threshold = st.sidebar.slider(
    "Correlation Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

max_lag = st.sidebar.slider(
    "Maximum Lag (days)",
    min_value=1,
    max_value=20,
    value=5
)

# Backtesting parameters
initial_investment = st.sidebar.number_input(
    "Initial Investment (₹)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=10000
)

# List of stocks
tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
    'MARUTI.NS', 'TATAMOTORS.NS', 'HEROMOTOCO.NS', 'TCS.NS',
    'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'RELIANCE.NS',
    'NTPC.NS', 'TATAPOWER.NS', 'TITAN.NS'
]

# Stock selector
target_stock = st.selectbox("Select Target Stock", tickers)

if st.button("Run Analysis"):
    try:
        # Download data with retry logic
        data = fetch_stock_data_with_retry(tickers, start_date, end_date)
        if data is None or data.empty:
            st.error("Failed to download stock data. Please try again later.")
            st.stop()
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                st.warning("Some stocks have missing values. They will be handled automatically.")
                data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate correlations
        with st.spinner("Calculating correlations..."):
            results = []
            progress_bar = st.progress(0)
            total_pairs = len(data.columns) * (len(data.columns) - 1) // 2
            current_pair = 0
            
            for i in range(len(data.columns)):
                for j in range(i + 1, len(data.columns)):
                    stock_a = data.iloc[:, i]
                    stock_b = data.iloc[:, j]
                    corr, lag = laggedCorr(stock_a.values, stock_b.values)
                    results.append((data.columns[i], data.columns[j], corr, lag))
                    
                    current_pair += 1
                    progress_bar.progress(current_pair / total_pairs)
            
            correlation_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
            lag_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
            
            for stock_a, stock_b, corr, lag in results:
                correlation_matrix.loc[stock_a, stock_b] = corr
                correlation_matrix.loc[stock_b, stock_a] = corr
                lag_matrix.loc[stock_a, stock_b] = lag
                lag_matrix.loc[stock_b, stock_a] = lag
            
            correlation_matrix = correlation_matrix.astype(float)
            lag_matrix = lag_matrix.astype(float)
        
        # Train model and make predictions
        with st.spinner("Training model and making predictions..."):
            predictor = StockPricePredictor(
                correlation_threshold=correlation_threshold,
                max_lag=max_lag
            )
            results = predictor.train(data, target_stock)
            
            if results is None:
                st.error("Error during model training. Please try different parameters.")
                st.stop()
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.metric("Average RMSE", f"₹{results['avg_rmse']:.2f}")
            st.metric("Average MAPE", f"{results['avg_mape']:.2%}")
            
            st.subheader("Feature Importance")
            sorted_features = dict(sorted(
                results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sorted_features.values()),
                    y=list(sorted_features.keys()),
                    orientation='h'
                )
            ])
            fig.update_layout(
                title="Top 5 Most Important Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Price Prediction")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['actual_values'],
                name="Actual Price",
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=results['predictions'],
                name="Predicted Price",
                line=dict(color='red')
            ))
            fig.update_layout(
                title=f"{target_stock} Price Prediction",
                xaxis_title="Time",
                yaxis_title="Price (₹)",
                height=400
            )
            st.plotly_chart(fig)
        
        # Correlation heatmap
        st.subheader("Stock Correlations")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            ax=ax
        )
        plt.title('Correlation Heatmap between Stocks')
        st.pyplot(fig)
        
        # Run backtest on training data
        with st.spinner("Running trading strategy backtest..."):
            backtest_results = backtest_strategy(
                data=data, 
                predictions=results['predictions'],
                actual_prices=results['actual_values'].values,
                target_stock=target_stock,
                investment=initial_investment
            )
            
            display_backtest_results(
                backtest_results=backtest_results,
                investment=initial_investment,
                target_stock=target_stock
            )
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please try again with different parameters or contact support if the issue persists.")