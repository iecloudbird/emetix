"""
Data fetcher for Yahoo Finance
Based on Phase 2 draft code
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from config.logging_config import get_logger

logger = get_logger(__name__)


class YFinanceFetcher:
    """Fetch stock data from Yahoo Finance"""
    
    def __init__(self):
        self.logger = logger
    
    def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch stock fundamentals and price data
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
        
        Returns:
            DataFrame with stock metrics or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                self.logger.warning(f"No historical data for {ticker}")
                return None
            
            # Calculate metrics
            df = pd.DataFrame({
                'ticker': [ticker],
                'pe_ratio': [info.get('trailingPE', 0)],
                'forward_pe': [info.get('forwardPE', 0)],
                'debt_equity': [info.get('debtToEquity', 0)],
                'current_ratio': [info.get('currentRatio', 0)],
                'market_cap': [info.get('marketCap', 0)],
                'beta': [info.get('beta', 0)],
                'dividend_yield': [info.get('dividendYield', 0)],
                'eps': [info.get('trailingEps', 0)],
                'revenue_growth': [info.get('revenueGrowth', 0)],
                'volatility': [hist['Close'].pct_change().std() * 100],  # Annualized
                'current_price': [info.get('currentPrice', 0)]
            })
            
            self.logger.info(f"Successfully fetched data for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, tickers: list) -> pd.DataFrame:
        """
        Fetch data for multiple stocks
        
        Args:
            tickers: List of ticker symbols
        
        Returns:
            Combined DataFrame
        """
        data_frames = []
        
        for ticker in tickers:
            df = self.fetch_stock_data(ticker)
            if df is not None:
                data_frames.append(df)
        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def fetch_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            ticker: Stock ticker
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            self.logger.error(f"Error fetching historical prices for {ticker}: {str(e)}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    fetcher = YFinanceFetcher()
    
    # Test single stock
    data = fetcher.fetch_stock_data('AAPL')
    print(data)
    
    # Test multiple stocks
    sp500_sample = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    bulk_data = fetcher.fetch_multiple_stocks(sp500_sample)
    print(f"\nFetched {len(bulk_data)} stocks")
