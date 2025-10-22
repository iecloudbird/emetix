"""
Time-Series Data Processor for LSTM-DCF
Extends existing YFinanceFetcher with sequential data preparation
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import time
from pathlib import Path

from src.data.fetchers import YFinanceFetcher
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class TimeSeriesProcessor:
    """
    Prepares sequential stock data for LSTM training
    
    Pattern: Maintains compatibility with existing fetchers
    """
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize time-series processor
        
        Args:
            sequence_length: Number of timesteps for LSTM sequences (default: 60 quarters)
        """
        self.fetcher = YFinanceFetcher()
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def fetch_sequential_data(
        self, 
        ticker: str, 
        period: str = "15y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and prepare sequential data for LSTM
        
        Args:
            ticker: Stock ticker symbol
            period: Historical period (default: 15 years for quarterly)
            
        Returns:
            DataFrame with time-series features or None on error
        """
        try:
            # Use existing fetcher for raw data
            stock_data = self.fetcher.fetch_stock_data(ticker)
            if stock_data is None or (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
                logger.warning(f"No stock data available for {ticker}")
                return None
            
            # Convert DataFrame to dict if needed
            if isinstance(stock_data, pd.DataFrame):
                stock_data = stock_data.to_dict('records')[0]
            
            # Fetch historical prices
            import yfinance as yf
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, interval="1d")
            
            if history.empty:
                logger.warning(f"No historical data for {ticker}")
                return None
            
            # Engineer time-series features
            df = self._engineer_features(history, stock_data)
            
            # Save to cache
            cache_dir = RAW_DATA_DIR / "timeseries"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{ticker}_timeseries.csv"
            df.to_csv(cache_path)
            logger.info(f"Cached time-series data: {cache_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Time-series fetch failed for {ticker}: {e}")
            return None
    
    def _engineer_features(
        self, 
        history: pd.DataFrame, 
        fundamentals: dict
    ) -> pd.DataFrame:
        """
        Create LSTM input features from raw data
        
        Features:
        - Price/Volume (from history)
        - Fundamental ratios (from fetcher)
        - Technical indicators (rolling)
        """
        df = pd.DataFrame()
        
        # Price features
        df['close'] = history['Close']
        df['volume'] = history['Volume']
        df['returns'] = df['close'].pct_change()
        
        # Technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['volatility_30'] = df['returns'].rolling(30).std()
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Fundamental features (broadcast static values)
        df['pe_ratio'] = fundamentals.get('pe_ratio', np.nan)
        df['beta'] = fundamentals.get('beta', 1.0)
        df['debt_equity'] = fundamentals.get('debt_to_equity', 0)
        df['eps'] = fundamentals.get('eps', 0)
        
        # Proxy FCFF (simplified - enhance with actual cash flow data)
        eps_value = fundamentals.get('eps', 0)
        if eps_value and not pd.isna(eps_value):
            df['fcff_proxy'] = df['close'] * eps_value * 0.7
        else:
            df['fcff_proxy'] = 0
        
        # Drop NaN from rolling calculations
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM input sequences (X) and targets (y)
        
        Args:
            df: Feature DataFrame
            target_col: Column to predict
            
        Returns:
            (X_sequences, y_targets) arrays
        """
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.error("No numeric columns found in DataFrame")
            return np.array([]), np.array([])
        
        # Normalize features
        feature_cols = numeric_df.columns.tolist()
        scaled_data = self.scaler.fit_transform(numeric_df[feature_cols])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i])
            if target_col in feature_cols:
                y.append(scaled_data[i, feature_cols.index(target_col)])
            else:
                logger.warning(f"Target column '{target_col}' not found, using first column")
                y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def batch_fetch_for_training(
        self, 
        tickers: List[str], 
        max_samples: int = 100
    ) -> pd.DataFrame:
        """
        Batch fetch for model training (S&P 500 sample)
        
        Pattern: Rate-limited to respect API constraints
        
        Args:
            tickers: List of ticker symbols
            max_samples: Maximum number of tickers to fetch
            
        Returns:
            Combined DataFrame with all tickers' time-series data
        """
        all_data = []
        successful_fetches = 0
        
        for i, ticker in enumerate(tickers[:max_samples]):
            logger.info(f"Fetching {ticker} ({i+1}/{min(len(tickers), max_samples)})")
            
            df = self.fetch_sequential_data(ticker)
            if df is not None and not df.empty:
                df['ticker'] = ticker
                all_data.append(df)
                successful_fetches += 1
            
            # Rate limiting (1 second between requests)
            time.sleep(1)
        
        if not all_data:
            logger.error("No data fetched for training")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully fetched {successful_fetches}/{min(len(tickers), max_samples)} tickers")
        logger.info(f"Total records: {len(combined)}")
        
        # Save combined data
        output_dir = PROCESSED_DATA_DIR / "training"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "lstm_training_data.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved combined training data: {output_path}")
        
        return combined
