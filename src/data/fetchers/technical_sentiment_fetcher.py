"""
Enhanced Technical and Sentiment Features Fetcher
Adds 14 comprehensive features for RF Risk + Sentiment Pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from config.logging_config import get_logger

logger = get_logger(__name__)


class TechnicalSentimentFetcher:
    """
    Fetches technical indicators and sentiment features
    14 Features: Beta, 30d volatility, Debt/Equity, Volume Z-score, 
    Short %, RSI, News sentiment (mean/std/volume/relevance)
    """
    
    def __init__(self):
        """Initialize fetcher"""
        self.logger = get_logger(__name__)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_volume_zscore(self, volume: pd.Series, period: int = 30) -> float:
        """
        Calculate Volume Z-score (current volume vs 30-day average)
        
        Args:
            volume: Volume series
            period: Rolling period
            
        Returns:
            Z-score of current volume
        """
        if len(volume) < period:
            return 0.0
        
        recent_volume = volume.iloc[-1]
        avg_volume = volume.tail(period).mean()
        std_volume = volume.tail(period).std()
        
        if std_volume == 0:
            return 0.0
        
        z_score = (recent_volume - avg_volume) / std_volume
        return float(z_score)
    
    def calculate_30d_volatility(self, prices: pd.Series) -> float:
        """
        Calculate 30-day historical volatility (annualized)
        
        Args:
            prices: Price series
            
        Returns:
            Annualized volatility
        """
        if len(prices) < 30:
            return 0.2  # Default 20% volatility
        
        returns = prices.pct_change().dropna()
        if len(returns) < 30:
            return 0.2
        
        volatility = returns.tail(30).std() * np.sqrt(252)  # Annualize
        return float(volatility) if not pd.isna(volatility) else 0.2
    
    def get_short_interest(self, ticker: str) -> float:
        """
        Get short interest as % of float
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Short % of float
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            shares_short = info.get('sharesShort', 0)
            float_shares = info.get('floatShares', info.get('sharesOutstanding', 1))
            
            if float_shares > 0:
                short_percentage = (shares_short / float_shares) * 100
                return min(short_percentage, 100.0)  # Cap at 100%
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Error getting short interest for {ticker}: {e}")
            return 0.0
    
    def get_news_sentiment_features(self, ticker: str) -> Dict[str, float]:
        """
        Get news sentiment features (mock implementation)
        In production, this would use Alpha Vantage NEWS_SENTIMENT API
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with sentiment_mean, sentiment_std, news_volume, relevance_mean
        """
        # Mock sentiment based on ticker characteristics
        # In production, replace with actual Alpha Vantage API call
        
        # High-growth tech stocks tend to have more volatile sentiment
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD', 'META']
        volatile_tickers = ['TSLA', 'AMD', 'NVDA', 'BA', 'GME', 'AMC']
        
        if ticker in volatile_tickers:
            # High volatility stocks have more extreme sentiment
            sentiment_mean = np.random.normal(0.1, 0.3)  # Slightly positive but volatile
            sentiment_std = np.random.uniform(0.4, 0.8)  # High volatility
            news_volume = np.random.uniform(50, 150)     # High news volume
            relevance_mean = np.random.uniform(0.7, 0.9) # High relevance
        elif ticker in tech_tickers:
            # Tech stocks generally positive sentiment
            sentiment_mean = np.random.normal(0.3, 0.2)  # Positive sentiment
            sentiment_std = np.random.uniform(0.2, 0.5)  # Moderate volatility
            news_volume = np.random.uniform(30, 80)      # Moderate news
            relevance_mean = np.random.uniform(0.6, 0.8) # Good relevance
        else:
            # Conservative stocks
            sentiment_mean = np.random.normal(0.1, 0.15) # Neutral to slightly positive
            sentiment_std = np.random.uniform(0.1, 0.3)  # Low volatility
            news_volume = np.random.uniform(10, 40)      # Lower news volume
            relevance_mean = np.random.uniform(0.5, 0.7) # Moderate relevance
        
        # Clamp values to reasonable ranges
        sentiment_mean = np.clip(sentiment_mean, -1.0, 1.0)
        sentiment_std = np.clip(sentiment_std, 0.1, 1.0)
        news_volume = max(1, news_volume)
        relevance_mean = np.clip(relevance_mean, 0.1, 1.0)
        
        return {
            'sentiment_mean': float(sentiment_mean),
            'sentiment_std': float(sentiment_std),
            'news_volume': float(news_volume),
            'relevance_mean': float(relevance_mean)
        }
    
    def fetch_enhanced_features(self, ticker: str) -> Dict[str, float]:
        """
        Fetch all 14 enhanced features for RF model
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with all 14 features
        """
        features = {}
        
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Get historical data (3 months for technical indicators)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)  # ~4 months buffer
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                self.logger.warning(f"No historical data for {ticker}")
                return None
            
            # Get info for fundamentals
            info = stock.info
            
            # 1-3: Core Risk Metrics
            features['beta'] = info.get('beta', 1.0)
            features['debt_to_equity'] = info.get('debtToEquity', 0) / 100.0 if info.get('debtToEquity') else 0.0
            features['30d_volatility'] = self.calculate_30d_volatility(hist['Close'])
            
            # 4-5: Volume and Short Interest
            features['volume_zscore'] = self.calculate_volume_zscore(hist['Volume'])
            features['short_percent'] = self.get_short_interest(ticker)
            
            # 6: Technical Indicator
            features['rsi_14'] = self.calculate_rsi(hist['Close'])
            
            # 7-10: News Sentiment Features
            sentiment_features = self.get_news_sentiment_features(ticker)
            features.update(sentiment_features)
            
            # 11-14: Additional Fundamental Features
            features['pe_ratio'] = info.get('forwardPE', info.get('trailingPE', 0))
            features['revenue_growth'] = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            features['current_ratio'] = info.get('currentRatio', 1.0)
            features['return_on_equity'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            
            # Clean up any NaN values
            for key, value in features.items():
                if pd.isna(value) or value is None:
                    features[key] = 0.0
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Error fetching enhanced features for {ticker}: {e}")
            return None


def test_enhanced_features():
    """Test the enhanced features fetcher"""
    fetcher = TechnicalSentimentFetcher()
    
    test_tickers = ['AAPL', 'TSLA', 'JNJ']
    
    for ticker in test_tickers:
        print(f"\n{ticker} Enhanced Features:")
        print("="*50)
        
        features = fetcher.fetch_enhanced_features(ticker)
        
        if features:
            for feature, value in features.items():
                print(f"  {feature:20s}: {value:8.3f}")
        else:
            print(f"  Failed to fetch features for {ticker}")


if __name__ == "__main__":
    test_enhanced_features()