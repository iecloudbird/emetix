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

    def fetch_insider_transactions(self, ticker: str) -> Optional[Dict]:
        """
        Fetch insider transaction data (buys/sells by officers/directors).

        Returns:
            Dict with insider transaction summary or None on error.
        """
        try:
            stock = yf.Ticker(ticker)
            txns = stock.insider_transactions

            if txns is None or (isinstance(txns, pd.DataFrame) and txns.empty):
                self.logger.info(f"No insider transactions for {ticker}")
                return {"ticker": ticker, "transactions": [], "summary": {"total": 0}}

            records = txns.head(20).to_dict(orient="records")
            # Summarise buy/sell counts
            buy_count = sum(
                1 for r in records
                if "purchase" in str(r.get("Text", "")).lower()
                or "buy" in str(r.get("Text", "")).lower()
            )
            sell_count = sum(
                1 for r in records
                if "sale" in str(r.get("Text", "")).lower()
                or "sell" in str(r.get("Text", "")).lower()
            )
            self.logger.info(f"Fetched {len(records)} insider txns for {ticker}")
            return {
                "ticker": ticker,
                "transactions": records,
                "summary": {
                    "total": len(records),
                    "buys": buy_count,
                    "sells": sell_count,
                },
            }
        except Exception as e:
            self.logger.error(f"Error fetching insider txns for {ticker}: {e}")
            return None

    def fetch_institutional_holders(self, ticker: str) -> Optional[Dict]:
        """
        Fetch top institutional holders (funds, banks, etc.).

        Returns:
            Dict with institutional holder details or None on error.
        """
        try:
            stock = yf.Ticker(ticker)
            inst = stock.institutional_holders

            if inst is None or (isinstance(inst, pd.DataFrame) and inst.empty):
                self.logger.info(f"No institutional holders data for {ticker}")
                return {"ticker": ticker, "holders": [], "count": 0}

            records = inst.head(15).to_dict(orient="records")
            self.logger.info(f"Fetched {len(records)} institutional holders for {ticker}")
            return {
                "ticker": ticker,
                "holders": records,
                "count": len(records),
            }
        except Exception as e:
            self.logger.error(f"Error fetching institutional holders for {ticker}: {e}")
            return None

    def fetch_major_holders(self, ticker: str) -> Optional[Dict]:
        """
        Fetch major holder breakdown (insider %, institutional %, etc.).

        Returns:
            Dict with major holder percentages or None on error.
        """
        try:
            stock = yf.Ticker(ticker)
            major = stock.major_holders

            if major is None or (isinstance(major, pd.DataFrame) and major.empty):
                self.logger.info(f"No major holders data for {ticker}")
                return {"ticker": ticker, "breakdown": {}}

            breakdown = {}
            for _, row in major.iterrows():
                label = str(row.iloc[1]).strip() if len(row) > 1 else ""
                value = row.iloc[0]
                if label:
                    breakdown[label] = value
            self.logger.info(f"Fetched major holders for {ticker}")
            return {"ticker": ticker, "breakdown": breakdown}
        except Exception as e:
            self.logger.error(f"Error fetching major holders for {ticker}: {e}")
            return None


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
