# --- ENHANCED QUANTITATIVE PORTFOLIO ANALYSIS WITH DEEP LEARNING ---
# Based on research showing DNN superiority in stock return prediction

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas_datareader.data as web
import plotly.graph_objects as go
import plotly.express as px
from hmmlearn.hmm import GaussianHMM
import statsmodels.tsa.api as sm
import statsmodels.tsa.stattools as smt
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

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
    'Consumer Defensive': 'XLP', 'Utilities': 'XLU'
}
factor_etfs = ['QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'IVE', 'IVW', 'MDY', 'MTUM', 'RSP', 'SPY', 'QUAL', 'SIZE', 'USMV']
etf_list = list(set(list(sector_etf_map.values()) + factor_etfs))

# Mapping for display names
REVERSE_METRIC_NAME_MAP = {
    'Return_5d': '5-Day Return', 'Return_10d': '10-Day Return', 'Return_21d': '1-Month Return',
    'Return_63d': '3-Month Return', 'Return_126d': '6-Month Return', 'Return_252d': '1-Year Return',
    'Volatility_21d': '1-Month Volatility', 'Volatility_63d': '3-Month Volatility', 'Volatility_252d': '1-Year Volatility',
    'Sharpe_252d': '1-Year Sharpe Ratio', 'Sortino_252d': '1-Year Sortino Ratio',
    'SPY_Alpha': 'Alpha vs SPY', 'SPY_Beta': 'Beta vs SPY',
    'Sector_Alpha': 'Alpha vs Sector', 'Sector_Beta': 'Beta vs Sector',
    'RSI_14d': '14-Day RSI', 'MACD_Signal_Line_Crossover': 'MACD Crossover',
    'GARCH_Vol': 'GARCH Volatility', 'HMM_Regime': 'HMM Regime'
}

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

class AttentionLayer(nn.Module):
    """Self-attention layer for capturing temporal dependencies"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted = attention_weights * lstm_output
        return weighted.sum(dim=1)

class DeepStockPredictor(nn.Module):
    """
    Deep Neural Network for stock return prediction.
    Based on research showing DNN superiority over shallow networks.
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3, use_attention=True):
        super(DeepStockPredictor, self).__init__()
        
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Build deep architecture
        layers = []
        prev_size = input_size
        
        # Add multiple hidden layers (deep architecture)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # LSTM for temporal patterns (if sequence data available)
        self.lstm = nn.LSTM(prev_size, prev_size//2, 2, batch_first=True, dropout=dropout_rate)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(prev_size//2)
        
        # Final prediction layers
        final_size = prev_size//2 if use_attention else prev_size
        self.predictor = nn.Sequential(
            nn.Linear(final_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x, temporal_features=None):
        # Extract features through deep layers
        features = self.feature_extractor(x)
        
        if temporal_features is not None:
            # This part is a placeholder for sequence-based models, but our current features are static per stock
            # To use LSTM, we would need to reshape features into sequences (e.g., (batch_size, seq_len, features))
            # For now, we pass the features directly to the predictor
            pass

        # For this implementation, we bypass LSTM/Attention if data is not structured as a sequence
        # Final prediction
        return self.predictor(features)

class EnsemblePredictor:
    """
    Ensemble model combining DNN, SVR, and Random Forest predictions.
    Research shows ensemble models achieve higher correlation than single models.
    """
    def __init__(self, feature_cols, target_col='Return_252d', device='cpu'):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = device
        
        # Initialize models
        self.dnn = None
        self.svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        # Scalers for normalization
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Model weights (learned through validation)
        self.weights = {'dnn': 0.5, 'svr': 0.25, 'rf': 0.25}
        
    def prepare_features(self, df):
        """Prepare and scale features for model input"""
        # Select valid features
        valid_features = [col for col in self.feature_cols if col in df.columns]
        X = df[valid_features].copy()
        
        # Handle potential inf/nan values from feature engineering
        X = X.replace([np.inf, -np.inf], np.nan)
        # Impute with column median
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def train_dnn(self, X_train, y_train, epochs=100, batch_size=32):
        """Train the Deep Neural Network"""
        input_size = X_train.shape[1]
        
        # Initialize DNN with deep architecture
        self.dnn = DeepStockPredictor(
            input_size=input_size,
            hidden_sizes=[256, 128, 64, 32],  # Deep architecture
            dropout_rate=0.3,
            use_attention=True # Although not fully utilized without sequence data, it's part of the architecture
        ).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = optim.AdamW(self.dnn.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        # Training loop
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
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            status_text.text(f"DNN Training - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            progress_bar.progress((epoch + 1) / epochs)
            
        status_text.text(f"DNN Training Complete. Final Loss: {avg_loss:.6f}")
    
    def fit(self, df, validation_split=0.2):
        """Train all models in the ensemble"""
        # Prepare features
        X = self.prepare_features(df)
        y = df[self.target_col].values
        
        # Remove NaN values from target
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features and target
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Time series split for validation
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Train DNN
        st.info("Training Deep Neural Network...")
        self.train_dnn(X_train, y_train)
        
        # Train SVR
        st.info("Training Support Vector Regression...")
        self.svr.fit(X_train, y_train)
        
        # Train Random Forest
        st.info("Training Random Forest...")
        self.rf.fit(X_train, y_train)
        
        # Optimize ensemble weights using validation set
        self.optimize_weights(X_val, y_val)
        
    def optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        self.dnn.eval()
        
        with torch.no_grad():
            # Get predictions from each model
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            dnn_preds = self.dnn(X_tensor).cpu().numpy().ravel()
            svr_preds = self.svr.predict(X_val)
            rf_preds = self.rf.predict(X_val)
        
        # Stack predictions
        predictions = np.column_stack([dnn_preds, svr_preds, rf_preds])
        
        # Optimize weights using least squares
        try:
            weights = np.linalg.lstsq(predictions, y_val, rcond=None)[0]
            weights = np.maximum(weights, 0)  # Non-negative weights
            if weights.sum() > 0:
                weights = weights / weights.sum()  # Normalize
            else:
                weights = np.array([0.34, 0.33, 0.33]) # Fallback
        except:
             weights = np.array([0.34, 0.33, 0.33]) # Fallback
        
        self.weights = {'dnn': weights[0], 'svr': weights[1], 'rf': weights[2]}
        st.success(f"Optimized Ensemble Weights - DNN: {weights[0]:.2%}, SVR: {weights[1]:.2%}, RF: {weights[2]:.2%}")
    
    def predict(self, df):
        """Generate ensemble predictions"""
        X = self.prepare_features(df)
        X_scaled = self.feature_scaler.transform(X)
        
        self.dnn.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            dnn_preds_scaled = self.dnn(X_tensor).cpu().numpy().ravel()
        
        svr_preds_scaled = self.svr.predict(X_scaled)
        rf_preds_scaled = self.rf.predict(X_scaled)
        
        # Inverse transform predictions to their original scale
        dnn_preds = self.target_scaler.inverse_transform(dnn_preds_scaled.reshape(-1, 1)).ravel()
        svr_preds = self.target_scaler.inverse_transform(svr_preds_scaled.reshape(-1, 1)).ravel()
        rf_preds = self.target_scaler.inverse_transform(rf_preds_scaled.reshape(-1, 1)).ravel()
        
        # Apply weighted ensemble
        ensemble_preds = (
            self.weights['dnn'] * dnn_preds +
            self.weights['svr'] * svr_preds +
            self.weights['rf'] * rf_preds
        )
        
        return {
            'ensemble': ensemble_preds,
            'dnn': dnn_preds,
            'svr': svr_preds,
            'rf': rf_preds
        }

################################################################################
# SECTION 2: LONG-SHORT PORTFOLIO CONSTRUCTION USING DNN
################################################################################

class DNNPortfolioOptimizer:
    """
    Portfolio optimizer using DNN predictions for long-short strategy.
    Research shows this achieves highest risk-adjusted returns.
    """
    def __init__(self, ensemble_predictor, historical_returns, risk_model='ledoit-wolf'):
        self.predictor = ensemble_predictor
        self.risk_model = risk_model
        self.historical_returns = historical_returns

    def construct_long_short_portfolio(self, df, predictions, long_pct=0.3, short_pct=0.3,
                                       net_exposure=0.0, max_position_size=0.1):
        """
        Construct long-short portfolio based on DNN predictions.
        """
        n_stocks = len(df)
        n_long = max(1, int(n_stocks * long_pct))
        n_short = max(1, int(n_stocks * short_pct))
        
        # Get ensemble predictions
        df['DNN_Prediction'] = predictions['ensemble']
        df['Prediction_Rank'] = df['DNN_Prediction'].rank(ascending=False, method='first')
        
        # Select long and short candidates
        long_candidates = df[df['Prediction_Rank'] <= n_long]
        short_candidates = df[df['Prediction_Rank'] > (n_stocks - n_short)]

        # Optimize positions
        long_weights_dict = self._optimize_positions(long_candidates, side='long', max_position=max_position_size)
        short_weights_dict = self._optimize_positions(short_candidates, side='short', max_position=max_position_size)

        # Scale weights to meet net exposure targets
        total_long_weight = sum(long_weights_dict.values())
        total_short_weight = sum(short_weights_dict.values())

        # Target gross exposure is typically 200% for a fully leveraged L/S portfolio
        # Here we target 100% long, 100% short unless net exposure is specified
        target_long = (1.0 + net_exposure) / 2.0
        target_short = (1.0 - net_exposure) / 2.0

        long_scale = target_long / total_long_weight if total_long_weight > 0 else 0
        short_scale = target_short / total_short_weight if total_short_weight > 0 else 0

        # Create final portfolio DataFrame
        portfolio_data = []
        for ticker, weight in long_weights_dict.items():
            pred = long_candidates.loc[long_candidates['Ticker'] == ticker, 'DNN_Prediction'].iloc[0]
            portfolio_data.append({'Ticker': ticker, 'Weight': weight * long_scale, 'Side': 'Long', 'DNN_Prediction': pred})
        
        for ticker, weight in short_weights_dict.items():
            pred = short_candidates.loc[short_candidates['Ticker'] == ticker, 'DNN_Prediction'].iloc[0]
            portfolio_data.append({'Ticker': ticker, 'Weight': -weight * short_scale, 'Side': 'Short', 'DNN_Prediction': pred})

        return pd.DataFrame(portfolio_data)

    def _optimize_positions(self, stocks_df, side='long', max_position=0.1):
        """
        Optimize position sizes using mean-variance optimization.
        """
        tickers = stocks_df['Ticker'].tolist()
        n = len(tickers)
        if n == 0:
            return {}
        
        # Use DNN predictions as expected returns
        expected_returns = stocks_df['DNN_Prediction'].values
        
        # Estimate covariance matrix from historical returns
        returns_matrix = self.historical_returns[tickers]
        
        if self.risk_model == 'ledoit-wolf':
            cov_matrix = LedoitWolf().fit(returns_matrix).covariance_
        else: # sample covariance
            cov_matrix = returns_matrix.cov().values

        # Optimization problem
        weights = cp.Variable(n)
        
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        # Objective: Maximize risk-adjusted return
        risk_aversion = 2.5
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1.0,      # Sum of weights is 1 (before scaling)
            weights >= 0,                # No shorting within the long or short book
            weights <= max_position      # Max position size
        ]
        
        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False) # ECOS is a good solver for this type of problem
            if prob.status in ["optimal", "optimal_inaccurate"]:
                optimized_weights = weights.value
            else: # Fallback
                 optimized_weights = np.ones(n) / n
        except: # Fallback
            optimized_weights = np.ones(n) / n
        
        return dict(zip(tickers, optimized_weights))

################################################################################
# SECTION 3: DATA FETCHING AND FEATURE ENGINEERING (IMPLEMENTED)
################################################################################

@lru_cache(maxsize=None)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_ticker_data(ticker, start_date, end_date):
    """Fetches historical data for a single ticker with retry logic."""
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

@st.cache_data(ttl=3600)
def fetch_all_etf_histories(_etf_list, start_date, end_date):
    """Fetches historical data for all ETFs concurrently."""
    etf_histories = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_etf = {executor.submit(fetch_ticker_data, etf, start_date, end_date): etf for etf in _etf_list}
        for future in as_completed(future_to_etf):
            etf = future_to_etf[future]
            try:
                data = future.result()
                if not data.empty:
                    etf_histories[etf] = data['Adj Close'].pct_change().dropna()
            except Exception as e:
                logging.error(f"Failed to fetch data for ETF {etf}: {e}")
    return etf_histories

def calculate_metrics(ticker, history, etf_histories, sector):
    """Calculates all financial metrics for a given stock."""
    if history.empty or len(history) < 252:
        return None
    
    returns = history['Adj Close'].pct_change().dropna()
    metrics = {'Ticker': ticker, 'Returns': returns}
    
    # Time-series metrics
    for d in [5, 10, 21, 63, 126, 252]:
        if len(returns) >= d:
            metrics[f'Return_{d}d'] = (1 + returns.tail(d)).prod() - 1
            metrics[f'Volatility_{d}d'] = returns.tail(d).std() * np.sqrt(d)

    # Risk-adjusted return
    risk_free_rate = 0.02 # Assume 2% annualized risk-free rate
    annual_return = metrics.get('Return_252d', 0)
    annual_vol = metrics.get('Volatility_252d', 0)
    if annual_vol > 0:
        metrics['Sharpe_252d'] = (annual_return - risk_free_rate) / annual_vol
        downside_returns = returns.tail(252)[returns.tail(252) < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics['Sortino_252d'] = (annual_return - risk_free_rate) / downside_std

    # Alpha and Beta
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
            
    # Technical Indicators
    if len(history) > 14:
        delta = history['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        metrics['RSI_14d'] = 100 - (100 / (1 + rs.iloc[-1]))
        
    if len(history) > 26:
        ema_26 = history['Adj Close'].ewm(span=26, adjust=False).mean()
        ema_12 = history['Adj Close'].ewm(span=12, adjust=False).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        metrics['MACD_Signal_Line_Crossover'] = 1 if macd.iloc[-1] > signal_line.iloc[-1] else 0

    # Advanced Volatility and Regime
    try:
        garch_model = arch_model(returns.tail(252) * 100, vol='Garch', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        metrics['GARCH_Vol'] = garch_fit.conditional_volatility.iloc[-1] / 100
    except:
        metrics['GARCH_Vol'] = np.nan

    try:
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        hmm_model.fit(returns.tail(252).values.reshape(-1, 1))
        if hmm_model.monitor_.converged:
            metrics['HMM_Regime'] = hmm_model.predict(returns.tail(1).values.reshape(-1, 1))[0]
    except:
        metrics['HMM_Regime'] = np.nan
        
    return metrics

@st.cache_data(ttl=3600)
def process_tickers(tickers_df, _etf_histories, _sector_etf_map, start_date, end_date):
    """Processes a list of tickers to fetch data and calculate metrics in parallel."""
    results = []
    failed = []
    returns_dict = {}
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {}
        for index, row in tickers_df.iterrows():
            ticker = row['Symbol']
            future = executor.submit(fetch_ticker_data, ticker, start_date, end_date)
            future_to_ticker[future] = (ticker, row['Name'], row['Sector'])

        progress_bar = st.progress(0)
        total_futures = len(future_to_ticker)
        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker, name, sector = future_to_ticker[future]
            try:
                history = future.result()
                if not history.empty:
                    metrics = calculate_metrics(ticker, history, _etf_histories, sector)
                    if metrics:
                        metrics.update({'Name': name, 'Sector': sector})
                        results.append(metrics)
                        returns_dict[ticker] = metrics['Returns']
                else:
                    failed.append(ticker)
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
                failed.append(ticker)
            progress_bar.progress((i + 1) / total_futures)

    results_df = pd.DataFrame(results)
    
    # Post-processing: Rank metrics and create a composite score
    numeric_cols = results_df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
         # Rank from 0 to 1
        results_df[f'{col}_Rank'] = results_df[col].rank(pct=True)

    # Calculate traditional score
    results_df['Score'] = 0
    for metric, weight in default_weights.items():
        if f'{metric}_Rank' in results_df.columns:
            results_df['Score'] += results_df[f'{metric}_Rank'] * weight
            
    return results_df, failed, pd.DataFrame(returns_dict)

################################################################################
# SECTION 4: PERFORMANCE ANALYSIS AND PORTFOLIO OPTIMIZATION
################################################################################

def calculate_ml_portfolio_metrics(portfolio_df, historical_returns, benchmark_returns):
    """
    Calculate performance metrics for ML-based portfolio.
    """
    if portfolio_df.empty or historical_returns.empty:
        return {}

    metrics = {}
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0.0, index=historical_returns.index)
    
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        if ticker in historical_returns.columns:
            portfolio_returns += weight * historical_returns[ticker]
    
    # Performance metrics
    metrics['Annual_Return'] = portfolio_returns.mean() * 252
    metrics['Volatility'] = portfolio_returns.std() * np.sqrt(252)
    metrics['Sharpe_Ratio'] = metrics['Annual_Return'] / metrics['Volatility'] if metrics['Volatility'] > 0 else 0
    
    # Information Ratio (vs. benchmark)
    if benchmark_returns is not None:
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        active_returns = portfolio_returns[common_idx] - benchmark_returns[common_idx]
        tracking_error = active_returns.std() * np.sqrt(252)
        if tracking_error > 0:
            metrics['Information_Ratio'] = (active_returns.mean() * 252) / tracking_error
    
    # Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['Max_Drawdown'] = drawdown.min()
    
    # Calmar and Sortino Ratios
    if metrics['Max_Drawdown'] != 0:
        metrics['Calmar_Ratio'] = metrics['Annual_Return'] / abs(metrics['Max_Drawdown'])
        
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    if downside_vol > 0:
        metrics['Sortino_Ratio'] = metrics['Annual_Return'] / downside_vol
    
    return metrics, portfolio_returns

def enhanced_portfolio_optimization(top_df, historical_returns, risk_aversion=2.5):
    """Performs Mean-Variance Optimization on the selected stocks."""
    tickers = top_df['Ticker'].tolist()
    returns_matrix = historical_returns[tickers]
    
    mu = returns_matrix.mean() * 252
    Sigma = LedoitWolf().fit(returns_matrix).covariance_ * 252
    
    weights = cp.Variable(len(tickers))
    ret = mu.values @ weights
    risk = cp.quad_form(weights, Sigma)
    
    prob = cp.Problem(cp.Maximize(ret - risk_aversion * risk), 
                      [cp.sum(weights) == 1, weights >= 0])
    prob.solve()
    
    return pd.DataFrame({'Ticker': tickers, 'Weight': weights.value})

################################################################################
# SECTION 5: VISUALIZATION AND UI HELPERS
################################################################################

@st.cache_data
def train_ml_models(_results_df, feature_cols, target_col='Return_252d'):
    """
    Train ML models on historical data. Cached for efficiency.
    """
    # Remove rows with missing target values
    train_df = _results_df.dropna(subset=[target_col]).copy()
    
    if len(train_df) < 50:
        st.warning("Insufficient data for ML training. Need at least 50 data points.")
        return None
    
    ensemble = EnsemblePredictor(
        feature_cols=feature_cols,
        target_col=target_col,
        device=DEVICE
    )
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
        # Calculate correlation between model predictions
        corr_matrix = np.corrcoef([predictions['dnn'], predictions['svr'], predictions['rf']])
        avg_corr = (corr_matrix[0,1] + corr_matrix[0,2] + corr_matrix[1,2]) / 3
        st.metric("Model Agreement (Correlation)", f"{avg_corr:.2%}")
        
        # Prediction dispersion
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

def main():
    st.title("🧠 AI-Enhanced Quantitative Portfolio Analysis")
    st.caption("Leveraging Deep Neural Networks for Superior Return Prediction")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Universe & Timeframe")
    
    # Load S&P 500 tickers
    try:
        sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_tickers['Symbol'] = sp500_tickers['Symbol'].str.replace('.', '-', regex=False)
        sp500_tickers = sp500_tickers[['Symbol', 'Security', 'GICS Sector']].rename(columns={'Security': 'Name', 'GICS Sector': 'Sector'})
    except Exception as e:
        st.error(f"Could not load S&P 500 tickers. Using a default list. Error: {e}")
        sp500_tickers = pd.DataFrame([
            {'Symbol': 'AAPL', 'Name': 'Apple Inc.', 'Sector': 'Technology'},
            {'Symbol': 'MSFT', 'Name': 'Microsoft Corp.', 'Sector': 'Technology'},
            {'Symbol': 'GOOGL', 'Name': 'Alphabet Inc.', 'Sector': 'Communication Services'},
             # Add more defaults if needed
        ])

    selected_sectors = st.sidebar.multiselect("Filter by Sector", options=sp500_tickers['Sector'].unique(), default=sp500_tickers['Sector'].unique())
    if not selected_sectors:
        tickers_df = sp500_tickers
    else:
        tickers_df = sp500_tickers[sp500_tickers['Sector'].isin(selected_sectors)]
    
    num_stocks = st.sidebar.slider("Number of Stocks to Analyze", 50, 500, 100)
    tickers_df = tickers_df.head(num_stocks)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=st.sidebar.slider("Historical Lookback (days)", 365, 2000, 1000))
    
    st.sidebar.header("🤖 ML Settings")
    use_ml = st.sidebar.checkbox("Enable Deep Learning Models", value=True, help="Use DNN ensemble for return prediction")
    
    if use_ml:
        ml_weight_in_final = st.sidebar.slider("ML Signal Weight (%)", 0, 100, 70, help="Weight of ML predictions vs traditional factors for long-only portfolio")

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
            st.stop()
        st.success(f"Successfully processed {len(results_df)} stocks. Failed to process {len(failed)}.")
        
        # --- ML MODELING ---
        ml_portfolio = None
        ml_metrics = {}
        portfolio_returns_ts = None

        if use_ml:
            st.header("2. Deep Learning Model Training & Prediction")
            feature_cols = list(default_weights.keys())
            
            ensemble_model = train_ml_models(results_df, feature_cols)
            
            if ensemble_model:
                with st.spinner("Generating predictions from ensemble model..."):
                    ml_predictions = ensemble_model.predict(results_df)
                    ml_predictions['weights'] = ensemble_model.weights # Store weights for display
                    results_df['ML_Score'] = ml_predictions['ensemble']
                st.success("ML predictions generated.")

                # Combine scores
                ml_weight = ml_weight_in_final / 100.0
                results_df['Combined_Score'] = (ml_weight * results_df['ML_Score'].rank(pct=True) + 
                                               (1 - ml_weight) * results_df['Score'])
                
                if is_long_short:
                    optimizer = DNNPortfolioOptimizer(ensemble_model, returns_df)
                    ml_portfolio = optimizer.construct_long_short_portfolio(
                        results_df.copy(), ml_predictions, 
                        long_pct=long_pct, short_pct=short_pct, net_exposure=net_exposure)
                    
                    # Calculate L/S portfolio metrics
                    ml_metrics, portfolio_returns_ts = calculate_ml_portfolio_metrics(
                        ml_portfolio, returns_df, etf_histories.get('SPY')
                    )
                    
                    # Display ML insights and plots
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
                st.metric("Net Exposure", f"{(long_df['Weight'].sum() + short_df['Weight'].sum()):.2%}")
            else:
                st.warning("Could not generate the Long-Short portfolio.")

        else: # Long-Only portfolios
            st.subheader("Long-Only Portfolio")
            score_col = 'Combined_Score' if use_ml else 'Score'
            top_15_df = results_df.nlargest(15, score_col)
            
            if weighting_method == "Enhanced Portfolio Optimization (EPO)":
                final_portfolio_df = enhanced_portfolio_optimization(top_15_df, returns_df)
            elif weighting_method == "Inverse Volatility":
                inv_vol = 1 / top_15_df['Volatility_252d']
                final_portfolio_df = pd.DataFrame({
                    'Ticker': top_15_df['Ticker'],
                    'Weight': inv_vol / inv_vol.sum()
                })
            else: # Equal Weight
                final_portfolio_df = pd.DataFrame({
                    'Ticker': top_15_df['Ticker'],
                    'Weight': 1 / 15
                })
            
            st.dataframe(final_portfolio_df.sort_values('Weight', ascending=False), use_container_width=True)

        # --- PERFORMANCE VISUALIZATION ---
        if portfolio_returns_ts is not None:
            st.header("4. Backtested Performance")
            cumulative_returns = (1 + portfolio_returns_ts).cumprod()
            benchmark_returns = etf_histories.get('SPY')
            cumulative_benchmark = (1 + benchmark_returns[cumulative_returns.index]).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='L/S Portfolio'))
            fig.add_trace(go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark, mode='lines', name='SPY Benchmark'))
            fig.update_layout(title="Cumulative Returns", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
