"""
Script to fetch and cache historical stock data
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.fetchers import YFinanceFetcher
from src.data.processors.time_series_processor import TimeSeriesProcessor
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_logger(__name__)


def fetch_and_save_data(tickers: list, period: str = "1y"):
    """
    Fetch historical data and save to CSV
    
    Args:
        tickers: List of stock tickers
        period: Time period (1y, 2y, 5y, etc.)
    """
    logger.info(f"Fetching data for {len(tickers)} stocks...")
    
    fetcher = YFinanceFetcher()
    
    # Fetch fundamentals
    fundamentals = fetcher.fetch_multiple_stocks(tickers)
    
    # Save fundamentals
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fundamentals_path = RAW_DATA_DIR / 'stocks' / f'fundamentals_{timestamp}.csv'
    fundamentals_path.parent.mkdir(parents=True, exist_ok=True)
    fundamentals.to_csv(fundamentals_path, index=False)
    
    logger.info(f"Saved fundamentals to {fundamentals_path}")
    
    # Fetch historical prices for each stock
    for ticker in tickers:
        logger.info(f"Fetching historical prices for {ticker}...")
        
        hist = fetcher.fetch_historical_prices(ticker, period=period)
        
        if not hist.empty:
            price_path = RAW_DATA_DIR / 'stocks' / f'{ticker}_{period}.csv'
            hist.to_csv(price_path)
            logger.info(f"Saved {ticker} prices to {price_path}")
    
    logger.info("Data fetching complete!")
    return fundamentals


def fetch_lstm_training_data(tickers: list, max_samples: int = 50):
    """
    Fetch time-series data for LSTM training
    
    Args:
        tickers: List of stock tickers
        max_samples: Maximum number of tickers to fetch
    """
    logger.info("=" * 80)
    logger.info("FETCHING TIME-SERIES DATA FOR LSTM TRAINING")
    logger.info("=" * 80)
    
    processor = TimeSeriesProcessor(sequence_length=60)
    
    # Batch fetch with rate limiting
    training_data = processor.batch_fetch_for_training(
        tickers=tickers,
        max_samples=max_samples
    )
    
    if not training_data.empty:
        logger.info(f"✓ LSTM training data prepared: {len(training_data)} records")
        logger.info(f"  Unique tickers: {training_data['ticker'].nunique()}")
        logger.info(f"  Date range: {training_data.index.min()} to {training_data.index.max()}" if hasattr(training_data.index, 'min') else "")
    else:
        logger.warning("No LSTM training data collected")
    
    return training_data


def main():
    """Main data fetching pipeline"""
    logger.info("=" * 80)
    logger.info("STARTING DATA FETCHING PIPELINE")
    logger.info("=" * 80)
    
    # S&P 500 sample tickers
    sp500_sample = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
        'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE',
        'CRM', 'NFLX', 'CMCSA', 'XOM', 'PFE', 'KO', 'PEP', 'CSCO', 'INTC',
        'NKE', 'TMO', 'ABT', 'VZ', 'ORCL', 'MRK', 'CVX', 'ACN', 'COST',
        'LLY', 'AVGO', 'TXN', 'MDT', 'NEE', 'DHR', 'BMY', 'PM', 'LIN',
        'HON', 'QCOM', 'UNP', 'RTX', 'LOW'
    ]
    
    # 1. Fetch traditional data
    logger.info("\n[1/2] Fetching traditional stock data...")
    data = fetch_and_save_data(sp500_sample, period="2y")
    logger.info(f"✓ Fetched data for {len(data)} stocks")
    
    # 2. Fetch LSTM time-series data
    logger.info("\n[2/2] Fetching time-series data for LSTM...")
    lstm_data = fetch_lstm_training_data(sp500_sample, max_samples=30)
    
    logger.info("=" * 80)
    logger.info("DATA FETCHING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✓ Traditional data: {len(data)} stocks")
    logger.info(f"✓ LSTM training data: {lstm_data['ticker'].nunique() if not lstm_data.empty else 0} stocks")
    logger.info("\nNext steps:")
    logger.info("  1. Train LSTM-DCF: python scripts/lstm/train_lstm_dcf_enhanced.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
