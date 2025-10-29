# --- ENHANCED QUANTITATIVE PORTFOLIO ANALYSIS WITH DEEP LEARNING ---
# Based on research showing DNN superiority in stock return prediction

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math
import tenacity
import logging
import requests
import time
import random
from scipy.stats import linregress
from scipy.linalg import solve_toeplitz
import cvxpy as cp
from arch import arch_model
from sklearn.covariance import LedoitWolf
from functools import lru_cache
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas_datareader.data as web
import plotly.graph_objects as go
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
import statsmodels.tsa.api as sm
from plotly.subplots import make_subplots
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Dict, List, Tuple

# --- Basic Configuration ---
logging.basicConfig(filename='stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="AI-Enhanced Quantitative Portfolio Analysis", layout="wide")

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    st.sidebar.success(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("💻 Running on CPU")

# --- Data Structures and Mappings ---
sector_etf_map = {
    'Technology': 'XLK', 'Consumer Cyclical': 'XLY', 'Communication Services': 'XLC',
    'Financial Services': 'XLF', 'Industrials': 'XLI', 'Basic Materials': 'XLB',
    'Energy': 'XLE', 'Real Estate': 'XLRE', 'Healthcare': 'XLV',
    'Consumer Defensive': 'XLP', 'Utilities': 'XLU',
    'Information Technology': 'XLK' # Add mapping for different naming convention
}
factor_etfs = ['QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'IVE', 'IVW', 'MDY', 'MTUM', 'RSP', 'SPY', 'QUAL', 'SIZE', 'USMV']
etf_list = list(set(list(sector_etf_map.values()) + factor_etfs))

# Default weights for traditional scoring
default_weights = {
    'Return_5d': 0.05, 'Return_10d': 0.05, 'Return_21d': 0.1, 'Return_63d': 0.1, 'Return_126d': 0.1, 'Return_252d': 0.1,
    'Volatility_21d': -0.05, 'Volatility_63d': -0.05, 'Volatility_252d': -0.05,
    'Sharpe_252d': 0.15, 'Sortino_252d': 0.05,
    'SPY_Alpha': 0.05, 'Sector_Alpha': 0.05
}

################################################################################
# SECTION 1: DEEP NEURAL NETWORK MODELS
################################################################################

class DeepStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3):
        super(DeepStockPredictor, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        self.feature_extractor = nn.Sequential(*layers)
        self.predictor = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.predictor(features)

class EnsemblePredictor:
    def __init__(self, feature_cols, target_col='Return_252d', device='cpu'):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = device
        self.dnn = None
        self.svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        self.weights = {'dnn': 0.5, 'svr': 0.25, 'rf': 0.25}

    def prepare_features(self, df):
        valid_features = [col for col in self.feature_cols if col in df.columns]
        X = df[valid_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        return X

    def train_dnn(self, X_train, y_train, epochs=100, batch_size=32):
        input_size = X_train.shape[1]
        self.dnn = DeepStockPredictor(input_size=input_size).to(self.device)
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.dnn.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        self.dnn.train()
        progress_bar = st.progress(0)
        status_text = st.empty()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.dnn(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            status_text.text(f"DNN Training - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"DNN Training Complete. Final Loss: {avg_loss:.6f}")

    def fit(self, df, validation_split=0.2):
        X = self.prepare_features(df)
        y = df[self.target_col].values
        valid_idx = ~np.isnan(y)
        X = X.loc[valid_idx]
        y = y[valid_idx]
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        st.info("Training Deep Neural Network...")
        self.train_dnn(X_train, y_train)
        st.info("Training Support Vector Regression...")
        self.svr.fit(X_train, y_train)
        st.info("Training Random Forest...")
        self.rf.fit(X_train, y_train)
        self.optimize_weights(X_val, y_val)

    def optimize_weights(self, X_val, y_val):
        self.dnn.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            dnn_preds = self.dnn(X_tensor).cpu().numpy().ravel()
            svr_preds = self.svr.predict(X_val)
            rf_preds = self.rf.predict(X_val)
        predictions = np.column_stack([dnn_preds, svr_preds, rf_preds])
        try:
            weights, _, _, _ = np.linalg.lstsq(predictions, y_val, rcond=None)
            weights = np.maximum(weights, 0)
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.array([0.34, 0.33, 0.33])
        except:
             weights = np.array([0.34, 0.33, 0.33])
        self.weights = {'dnn': weights[0], 'svr': weights[1], 'rf': weights[2]}
        st.success(f"Optimized Ensemble Weights - DNN: {weights[0]:.2%}, SVR: {weights[1]:.2%}, RF: {weights[2]:.2%}")

    def predict(self, df):
        X = self.prepare_features(df)
        X_scaled = self.feature_scaler.transform(X)
        self.dnn.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            dnn_preds_scaled = self.dnn(X_tensor).cpu().numpy().ravel()
        svr_preds_scaled = self.svr.predict(X_scaled)
        rf_preds_scaled = self.rf.predict(X_scaled)
        dnn_preds = self.target_scaler.inverse_transform(dnn_preds_scaled.reshape(-1, 1)).ravel()
        svr_preds = self.target_scaler.inverse_transform(svr_preds_scaled.reshape(-1, 1)).ravel()
        rf_preds = self.target_scaler.inverse_transform(rf_preds_scaled.reshape(-1, 1)).ravel()
        ensemble_preds = (self.weights['dnn'] * dnn_preds + self.weights['svr'] * svr_preds + self.weights['rf'] * rf_preds)
        return {'ensemble': ensemble_preds, 'dnn': dnn_preds, 'svr': svr_preds, 'rf': rf_preds}

################################################################################
# SECTION 2: PORTFOLIO OPTIMIZATION
################################################################################

class DNNPortfolioOptimizer:
    def __init__(self, historical_returns, risk_model='ledoit-wolf'):
        self.risk_model = risk_model
        self.historical_returns = historical_returns

    def construct_long_short_portfolio(self, df, predictions, long_pct=0.3, short_pct=0.3, net_exposure=0.0, max_position_size=0.1):
        n_stocks = len(df)
        n_long = max(1, int(n_stocks * long_pct))
        n_short = max(1, int(n_stocks * short_pct))
        df['DNN_Prediction'] = predictions['ensemble']
        df['Prediction_Rank'] = df['DNN_Prediction'].rank(ascending=False, method='first')
        long_candidates = df[df['Prediction_Rank'] <= n_long]
        short_candidates = df[df['Prediction_Rank'] > (n_stocks - n_short)]
        long_weights_dict = self._optimize_positions(long_candidates, max_position=max_position_size)
        short_weights_dict = self._optimize_positions(short_candidates, max_position=max_position_size)
        total_long_weight = sum(long_weights_dict.values())
        total_short_weight = sum(short_weights_dict.values())
        target_long = (1.0 + net_exposure) / 2.0
        target_short = (1.0 - net_exposure) / 2.0
        long_scale = target_long / total_long_weight if total_long_weight > 0 else 0
        short_scale = target_short / total_short_weight if total_short_weight > 0 else 0
        portfolio_data = []
        for ticker, weight in long_weights_dict.items():
            pred = long_candidates.loc[long_candidates['Ticker'] == ticker, 'DNN_Prediction'].iloc[0]
            portfolio_data.append({'Ticker': ticker, 'Weight': weight * long_scale, 'Side': 'Long', 'DNN_Prediction': pred})
        for ticker, weight in short_weights_dict.items():
            pred = short_candidates.loc[short_candidates['Ticker'] == ticker, 'DNN_Prediction'].iloc[0]
            portfolio_data.append({'Ticker': ticker, 'Weight': -weight * short_scale, 'Side': 'Short', 'DNN_Prediction': pred})
        return pd.DataFrame(portfolio_data)

    def _optimize_positions(self, stocks_df, max_position=0.1):
        tickers = stocks_df['Ticker'].tolist()
        n = len(tickers)
        if n == 0: return {}
        expected_returns = stocks_df['DNN_Prediction'].values
        returns_matrix = self.historical_returns[tickers]
        if self.risk_model == 'ledoit-wolf':
            cov_matrix = LedoitWolf().fit(returns_matrix).covariance_
        else:
            cov_matrix = returns_matrix.cov().values
        weights = cp.Variable(n)
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        risk_aversion = 2.5
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        constraints = [cp.sum(weights) == 1.0, weights >= 0, weights <= max_position]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS)
            optimized_weights = weights.value if prob.status in ["optimal", "optimal_inaccurate"] else np.ones(n) / n
        except:
            optimized_weights = np.ones(n) / n
        return dict(zip(tickers, optimized_weights))

################################################################################
# SECTION 3: DATA FETCHING AND FEATURE ENGINEERING
################################################################################

################################################################################
# SECTION 3: DATA FETCHING AND FEATURE ENGINEERING (REVISED AND FIXED)
################################################################################

@lru_cache(maxsize=None)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=12))
def fetch_ticker_data(ticker, start_date, end_date, session):
    """
    Fetches historical data for a single ticker with retry logic and a session object.
    The session helps manage connections and headers, making requests more robust.
    """
    return yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,  # Automatically adjusts for splits and dividends
        session=session      # Use the shared session object
    )

def calculate_metrics(ticker, history, etf_histories, sector):
    """Calculates all financial metrics for a given stock. (No changes here, but included for completeness)"""
    if history.empty or len(history) < 252:
        return None
    returns = history['Close'].pct_change().dropna()
    if returns.empty:
        return None
    metrics = {'Ticker': ticker, 'Returns': returns}
    for d in [5, 10, 21, 63, 126, 252]:
        if len(returns) >= d:
            metrics[f'Return_{d}d'] = (1 + returns.tail(d)).prod() - 1
            metrics[f'Volatility_{d}d'] = returns.tail(d).std() * np.sqrt(d)
    if 'Return_252d' not in metrics:
        return None
    risk_free_rate = 0.02
    annual_return = metrics.get('Return_252d', 0)
    annual_vol = metrics.get('Volatility_252d', 1)
    metrics['Sharpe_252d'] = (annual_return - risk_free_rate) / annual_vol
    downside_returns = returns.tail(252)[returns.tail(252) < 0]
    if not downside_returns.empty:
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics['Sortino_252d'] = (annual_return - risk_free_rate) / downside_std
    spy_returns = etf_histories.get('SPY')
    if spy_returns is not None:
        common_idx = returns.index.intersection(spy_returns.index)
        if len(common_idx) > 20:
            beta, alpha, _, _, _ = linregress(spy_returns[common_idx], returns[common_idx])
            metrics['SPY_Alpha'] = alpha * 252
            metrics['SPY_Beta'] = beta
    sector_etf = sector_etf_map.get(sector)
    if sector_etf and sector_etf in etf_histories:
        sector_returns = etf_histories[sector_etf]
        common_idx = returns.index.intersection(sector_returns.index)
        if len(common_idx) > 20:
            beta, alpha, _, _, _ = linregress(sector_returns[common_idx], returns[common_idx])
            metrics['Sector_Alpha'] = alpha * 252
            metrics['Sector_Beta'] = beta
    return metrics

# --- REVISED AND BULLETPROOF TICKER PROCESSOR ---
@st.cache_data(ttl=3600)
def process_tickers(tickers_df, _etf_histories, _sector_etf_map, start_date, end_date):
    """
    Processes a list of tickers sequentially (one by one) to be extremely robust
    against Yahoo Finance rate limiting. This will be slower but much more reliable.
    """
    results, failed, returns_dict = [], {}, {}

    # We create one persistent session for all downloads.
    session = requests.Session()
    # This header is crucial to mimic a real browser.
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    total_tickers = len(tickers_df)
    progress_bar = st.progress(0, text=f"Starting sequential download for {total_tickers} stocks...")

    # --- NO MORE ThreadPoolExecutor. A simple, reliable for loop. ---
    for i, row in tickers_df.iterrows():
        ticker, name, sector = row['Symbol'], row['Name'], row['Sector']
        
        progress_text = f"({i+1}/{total_tickers}) Downloading {ticker}... Please be patient."
        progress_bar.progress((i + 1) / total_tickers, text=progress_text)
        
        try:
            # We will use the tenacious retry decorator on our fetch function
            history = fetch_ticker_data(ticker, start_date, end_date, session)
            
            # After trying, check if the download actually succeeded.
            if history.empty:
                # yfinance returns an empty dataframe for failed downloads (e.g., 404, invalid ticker)
                raise ValueError("Dataframe is empty (likely delisted or invalid ticker).")

            if len(history) < 252:
                failed[ticker] = "Insufficient historical data (less than 1 year)"
                continue # Skip to the next ticker

            metrics = calculate_metrics(ticker, history, _etf_histories, sector)
            if metrics:
                metrics.update({'Name': name, 'Sector': sector})
                results.append(metrics)
                returns_dict[ticker] = metrics['Returns']
            else:
                failed[ticker] = "Metric calculation failed"

        except Exception as e:
            # This catch block will now handle the final error after all retries have failed.
            if isinstance(e, tenacity.RetryError):
                failed[ticker] = "All download attempts failed. Last error: YFDataException"
            else:
                failed[ticker] = f"An unexpected error occurred: {str(e)}"
        
        # --- THE MOST IMPORTANT PART ---
        # A mandatory, longer, and random delay between EACH request to appear human.
        time.sleep(random.uniform(0.7, 1.5))

    progress_bar.empty() # Clean up the progress bar after completion

    # The rest of the function remains the same
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df, failed, pd.DataFrame()

    for col in results_df.select_dtypes(include=np.number).columns:
        results_df[f'{col}_Rank'] = results_df[col].rank(pct=True)

    results_df['Score'] = sum(
        results_df[f'{metric}_Rank'] * weight
        for metric, weight in default_weights.items()
        if f'{metric}_Rank' in results_df
    )

    return results_df, failed, pd.DataFrame(returns_dict)
################################################################################
# SECTION 4: PERFORMANCE ANALYSIS AND PORTFOLIO OPTIMIZATION
################################################################################

def calculate_ml_portfolio_metrics(portfolio_df, historical_returns, benchmark_returns):
    """
    Calculate performance metrics for ML-based portfolio.
    """
    if portfolio_df.empty or historical_returns.empty:
        return {}, pd.Series(dtype=float)

    metrics = {}
    portfolio_returns = pd.Series(0.0, index=historical_returns.index)

    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        if ticker in historical_returns.columns:
            portfolio_returns += weight * historical_returns[ticker]

    if portfolio_returns.std() == 0: return {}, portfolio_returns

    metrics['Annual_Return'] = portfolio_returns.mean() * 252
    metrics['Volatility'] = portfolio_returns.std() * np.sqrt(252)
    metrics['Sharpe_Ratio'] = metrics['Annual_Return'] / metrics['Volatility'] if metrics['Volatility'] > 0 else 0

    if benchmark_returns is not None:
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        active_returns = portfolio_returns[common_idx] - benchmark_returns[common_idx]
        tracking_error = active_returns.std() * np.sqrt(252)
        if tracking_error > 0:
            metrics['Information_Ratio'] = (active_returns.mean() * 252) / tracking_error

    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['Max_Drawdown'] = drawdown.min()

    if metrics['Max_Drawdown'] != 0:
        metrics['Calmar_Ratio'] = metrics['Annual_Return'] / abs(metrics['Max_Drawdown'])
        
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if not downside_returns.empty:
        downside_vol = downside_returns.std() * np.sqrt(252)
        if downside_vol > 0:
            metrics['Sortino_Ratio'] = metrics['Annual_Return'] / downside_vol
    
    return metrics, portfolio_returns

def enhanced_portfolio_optimization(top_df, historical_returns):
    """Performs Mean-Variance Optimization on the selected stocks."""
    tickers = top_df['Ticker'].tolist()
    returns_matrix = historical_returns[tickers]
    
    mu = returns_matrix.mean() * 252
    Sigma = LedoitWolf().fit(returns_matrix).covariance_ * 252
    
    weights = cp.Variable(len(tickers))
    ret = mu.values @ weights
    risk = cp.quad_form(weights, Sigma)
    
    prob = cp.Problem(cp.Maximize(ret - 2.5 * risk), 
                      [cp.sum(weights) == 1, weights >= 0, weights <= 0.2]) # Max 20% weight per stock
    prob.solve(solver=cp.ECOS)
    
    # Handle solver failure
    if prob.status != 'optimal':
        return pd.DataFrame({'Ticker': tickers, 'Weight': 1/len(tickers)})
        
    return pd.DataFrame({'Ticker': tickers, 'Weight': weights.value})

################################################################################
# SECTION 5: VISUALIZATION AND UI HELPERS
################################################################################

@st.cache_data
def train_ml_models(_results_df, feature_cols, target_col='Return_252d'):
    """
    Train ML models on historical data. Cached for efficiency.
    """
    train_df = _results_df.dropna(subset=[target_col]).copy()
    if len(train_df) < 50:
        st.warning("Insufficient data for ML training. Need at least 50 data points.")
        return None
    ensemble = EnsemblePredictor(feature_cols=feature_cols, target_col=target_col, device=DEVICE)
    ensemble.fit(train_df)
    return ensemble

def display_ml_insights(portfolio_df, predictions, metrics):
    """
    Display ML model insights and predictions.
    """
    st.header("🤖 Deep Learning Insights")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        st.subheader("Model Confidence")
        corr_matrix = np.corrcoef([predictions['dnn'], predictions['svr'], predictions['rf']])
        avg_corr = (corr_matrix[0,1] + corr_matrix[0,2] + corr_matrix[1,2]) / 3
        st.metric("Model Agreement (Correlation)", f"{avg_corr:.2%}")
        pred_std = np.std([predictions['dnn'], predictions['svr'], predictions['rf']], axis=0).mean()
        st.metric("Prediction Dispersion (Std Dev)", f"{pred_std:.4f}")
    with col2:
        st.subheader("Ensemble Weights")
        st.metric("DNN Weight", f"{predictions['weights']['dnn']:.2%}")
        st.metric("SVR Weight", f"{predictions['weights']['svr']:.2%}")
        st.metric("RF Weight", f"{predictions['weights']['rf']:.2%}")
    with col3:
        st.subheader("L/S Portfolio Performance")
        for metric, value in metrics.items():
            st.metric(metric.replace('_', ' '), f"{value:.2%}" if "Return" in metric or "Drawdown" in metric else f"{value:.2f}")

def plot_prediction_distribution(predictions, portfolio_df):
    """
    Plot the distribution of ML predictions.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Ensemble Prediction Distribution', 'Long vs. Short Predictions'])
    fig.add_trace(go.Histogram(x=predictions['ensemble'], name='Ensemble', nbinsx=50, marker_color='#636EFA'), row=1, col=1)
    if portfolio_df is not None and not portfolio_df.empty:
        long_preds = portfolio_df[portfolio_df['Side'] == 'Long']['DNN_Prediction']
        short_preds = portfolio_df[portfolio_df['Side'] == 'Short']['DNN_Prediction']
        fig.add_trace(go.Box(y=long_preds, name='Long Picks', marker_color='green'), row=1, col=2)
        fig.add_trace(go.Box(y=short_preds, name='Short Picks', marker_color='red'), row=1, col=2)
    fig.update_layout(title_text="ML Model Prediction Analysis", showlegend=False, height=400, template='plotly_dark')
    return fig

################################################################################
# MAIN APP FUNCTION
################################################################################
@st.cache_data(ttl=3600)
def fetch_all_etf_histories(_etf_list, start_date, end_date):
    """
    Fetches historical data for all ETFs concurrently using a session object for robustness.
    """
    etf_histories = {}
    
    # --- FIX IMPLEMENTATION ---
    # Create a session object with headers, just like in process_tickers
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    # Use a conservative number of workers
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Pass the session object to each fetch_ticker_data call
        future_to_etf = {
            executor.submit(fetch_ticker_data, etf, start_date, end_date, session): etf 
            for etf in _etf_list
        }
        # --- END OF FIX ---

        for future in as_completed(future_to_etf):
            etf = future_to_etf[future]
            try:
                data = future.result()
                if not data.empty:
                    etf_histories[etf] = data['Close'].pct_change().dropna()
            except Exception as e:
                logging.error(f"Failed to fetch data for ETF {etf}: {e}")
                
    return etf_histories
def main():
    st.title("🧠 AI-Enhanced Quantitative Portfolio Analysis")
    st.caption("Leveraging Deep Neural Networks for Superior Return Prediction")

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Universe & Timeframe")

    @st.cache_data(ttl=86400)
    def load_sp500_tickers():
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            sp500_tickers = pd.read_html(response.text)[0]
            sp500_tickers['Symbol'] = sp500_tickers['Symbol'].str.replace('.', '-', regex=False)
            sp500_tickers = sp500_tickers[['Symbol', 'Security', 'GICS Sector']].rename(columns={'Security': 'Name', 'GICS Sector': 'Sector'})
            return sp500_tickers
        except Exception as e:
            st.error(f"Could not load S&P 500 tickers. Using a default list. Error: {e}")
            return pd.DataFrame([{'Symbol': s, 'Name': n, 'Sector': c} for s, n, c in [('AAPL', 'Apple Inc.', 'Information Technology'), ('MSFT', 'Microsoft Corp.', 'Information Technology'), ('GOOGL', 'Alphabet Inc.', 'Communication Services')]])

    sp500_tickers = load_sp500_tickers()
    selected_sectors = st.sidebar.multiselect("Filter by Sector", options=sp500_tickers['Sector'].unique(), default=sp500_tickers['Sector'].unique())
    tickers_df = sp500_tickers[sp500_tickers['Sector'].isin(selected_sectors)] if selected_sectors else sp500_tickers
    num_stocks = st.sidebar.slider("Number of Stocks to Analyze", 20, 500, 100, 10)
    tickers_df = tickers_df.head(num_stocks)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=st.sidebar.slider("Historical Lookback (days)", 365, 2000, 1000))

    st.sidebar.header("🤖 ML Settings")
    use_ml = st.sidebar.checkbox("Enable Deep Learning Models", value=True)
    ml_weight_in_final = st.sidebar.slider("ML Signal Weight (%)", 0, 100, 70) if use_ml else 0

    st.sidebar.header("📈 Portfolio Construction")
    is_long_short = st.sidebar.checkbox("Enable Long-Short Portfolio (ML Required)", value=True, disabled=not use_ml)
    if is_long_short and use_ml:
        long_pct = st.sidebar.slider("Long Allocation (%)", 10, 50, 20) / 100.0
        short_pct = st.sidebar.slider("Short Allocation (%)", 10, 50, 20) / 100.0
        net_exposure = st.sidebar.slider("Net Exposure (%)", -50, 50, 0) / 100.0
        weighting_method = "ML-Enhanced Long/Short"
    else:
        weighting_method = st.sidebar.selectbox("Weighting Method (Long-Only)", ["Enhanced Portfolio Optimization (EPO)", "Equal Weight", "Inverse Volatility"])

    # --- DATA PROCESSING ---
    if st.sidebar.button("Run Analysis", type="primary"):
        st.header("1. Data Collection & Feature Engineering")
        with st.spinner("Fetching market data and calculating factors..."):
            etf_histories = fetch_all_etf_histories(etf_list, start_date, end_date)
            results_df, failed, returns_df = process_tickers(tickers_df, etf_histories, sector_etf_map, start_date, end_date)
        if results_df.empty:
            st.error("No data could be processed. Check your ticker list and date range.")
            if failed: st.warning("The following tickers failed to process:"); st.json(failed)
            st.stop()
        st.success(f"Successfully processed {len(results_df)} stocks.")
        if failed:
            with st.expander(f"See {len(failed)} failed tickers"): st.json(failed)

        # --- ML MODELING ---
        ml_portfolio = None; ml_metrics = {}; portfolio_returns_ts = pd.Series(dtype=float)
        if use_ml:
            st.header("2. Deep Learning Model Training & Prediction")
            feature_cols = list(default_weights.keys())
            ensemble_model = train_ml_models(results_df, feature_cols)
            if ensemble_model:
                with st.spinner("Generating predictions..."):
                    ml_predictions = ensemble_model.predict(results_df)
                    ml_predictions['weights'] = ensemble_model.weights
                    results_df['ML_Score'] = ml_predictions['ensemble']
                st.success("ML predictions generated.")
                results_df['Combined_Score'] = (ml_weight_in_final / 100.0 * results_df['ML_Score'].rank(pct=True) + (1 - ml_weight_in_final / 100.0) * results_df['Score'])
                if is_long_short:
                    optimizer = DNNPortfolioOptimizer(returns_df)
                    ml_portfolio = optimizer.construct_long_short_portfolio(results_df.copy(), ml_predictions, long_pct, short_pct, net_exposure)
                    ml_metrics, portfolio_returns_ts = calculate_ml_portfolio_metrics(ml_portfolio, returns_df, etf_histories.get('SPY'))
                    display_ml_insights(ml_portfolio, ml_predictions, ml_metrics)
                    st.plotly_chart(plot_prediction_distribution(ml_predictions, ml_portfolio), use_container_width=True)

        # --- PORTFOLIO CONSTRUCTION ---
        st.header("3. Portfolio Allocation")
        if weighting_method == "ML-Enhanced Long/Short":
            st.subheader("ML-Based Long-Short Portfolio")
            if ml_portfolio is not None and not ml_portfolio.empty:
                col1, col2 = st.columns(2)
                long_df = ml_portfolio[ml_portfolio['Side'] == 'Long']
                short_df = ml_portfolio[ml_portfolio['Side'] == 'Short']
                with col1:
                    st.metric("Total Long Exposure", f"{long_df['Weight'].sum():.2%}")
                    st.dataframe(long_df.sort_values('Weight', ascending=False), use_container_width=True)
                with col2:
                    st.metric("Total Short Exposure", f"{short_df['Weight'].sum():.2%}")
                    st.dataframe(short_df.sort_values('Weight', ascending=True), use_container_width=True)
                st.metric("Net Exposure", f"{(long_df['Weight'].sum() + short_df['Weight'].sum()):.2%}", "A positive value indicates a long bias.")
            else:
                st.warning("Could not generate the Long-Short portfolio. Ensure ML models were trained successfully.")
        else:
            st.subheader("Long-Only Portfolio")
            score_col = 'Combined_Score' if use_ml and 'Combined_Score' in results_df.columns else 'Score'
            top_15_df = results_df.nlargest(15, score_col)
            if weighting_method == "Enhanced Portfolio Optimization (EPO)":
                final_portfolio_df = enhanced_portfolio_optimization(top_15_df, returns_df)
            elif weighting_method == "Inverse Volatility":
                inv_vol = 1 / top_15_df['Volatility_252d']
                final_portfolio_df = pd.DataFrame({'Ticker': top_15_df['Ticker'], 'Weight': inv_vol / inv_vol.sum()})
            else: # Equal Weight
                final_portfolio_df = pd.DataFrame({'Ticker': top_15_df['Ticker'], 'Weight': 1 / 15})
            st.dataframe(final_portfolio_df.sort_values('Weight', ascending=False), use_container_width=True)
            _, portfolio_returns_ts = calculate_ml_portfolio_metrics(final_portfolio_df, returns_df, etf_histories.get('SPY'))

        # --- PERFORMANCE VISUALIZATION ---
        if not portfolio_returns_ts.empty:
            st.header("4. Backtested Performance")
            cumulative_returns = (1 + portfolio_returns_ts).cumprod()
            benchmark_returns = etf_histories.get('SPY')
            if benchmark_returns is not None:
                common_index = cumulative_returns.index.intersection(benchmark_returns.index)
                cumulative_benchmark = (1 + benchmark_returns[common_index]).cumprod()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Portfolio'))
                fig.add_trace(go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark, mode='lines', name='SPY Benchmark'))
                fig.update_layout(title="Cumulative Returns vs. Benchmark (SPY)", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
