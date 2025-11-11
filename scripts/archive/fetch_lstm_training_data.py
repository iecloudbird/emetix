"""
Daily Batch Fetcher for LSTM-DCF Training Data
Fetches financial statements from Alpha Vantage for NYSE + S&P 500 stocks

Features:
- Smart caching (skip already-fetched stocks)
- Rate limit management (25 calls/day, 5/min)
- Progress tracking
- Resume capability
- Graceful error handling

Usage:
    python scripts/fetch_lstm_training_data.py --daily-limit 10
    python scripts/fetch_lstm_training_data.py --tickers AAPL MSFT GOOGL
    python scripts/fetch_lstm_training_data.py --resume
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd

from src.data.fetchers.alpha_vantage_financials import AlphaVantageFinancialsFetcher
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


# S&P 500 tickers (top 100 for FYP scope - expand as needed)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'KO', 'AVGO', 'COST', 'LLY', 'TMO', 'WMT', 'MCD', 'ACN', 'CSCO', 'ABT',
    'DHR', 'VZ', 'ADBE', 'NKE', 'CRM', 'TXN', 'CMCSA', 'ORCL', 'INTC', 'DIS',
    'PM', 'NEE', 'BMY', 'UPS', 'T', 'QCOM', 'RTX', 'HON', 'LOW', 'UNP',
    'INTU', 'AMD', 'SPGI', 'BA', 'COP', 'SBUX', 'AMAT', 'DE', 'GS', 'CAT',
    'AMGN', 'AXP', 'BKNG', 'LMT', 'PLD', 'GILD', 'MDLZ', 'ADI', 'TJX', 'SYK',
    'MMC', 'ADP', 'CI', 'VRTX', 'BDX', 'CVS', 'ZTS', 'TMUS', 'CB', 'SO',
    'PGR', 'ISRG', 'DUK', 'REGN', 'MO', 'NOC', 'SLB', 'ITW', 'EOG', 'MMM',
    'CL', 'BSX', 'APD', 'EQIX', 'GE', 'SCHW', 'PNC', 'USB', 'BLK', 'CME'
]

# NYSE tickers (top 50 large caps)
NYSE_TICKERS = [
    'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'USB', 'PNC', 'TFC', 'COF',
    'BK', 'STT', 'FITB', 'KEY', 'RF', 'CFG', 'HBAN', 'CMA', 'ZION', 'SIVB',
    'F', 'GM', 'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'WKHS',
    'T', 'VZ', 'TMUS', 'S', 'DISH', 'CHTR', 'CMCSA', 'LUMN', 'VOD', 'TEF',
    'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'MRO', 'DVN', 'APA', 'EOG', 'HES','ZETA','OSCR','HIMS'
]

# Combine and deduplicate
ALL_TICKERS = sorted(list(set(SP500_TICKERS + NYSE_TICKERS)))


class LSTMTrainingDataCollector:
    """Manages daily batch collection of financial statements for LSTM training"""
    
    def __init__(self):
        self.fetcher = AlphaVantageFinancialsFetcher()
        self.progress_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "fetch_progress.json"
        self.output_dir = PROCESSED_DATA_DIR / "lstm_dcf_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load progress
        self.progress = self._load_progress()
        
        logger.info("LSTM Training Data Collector initialized")
        logger.info(f"Progress file: {self.progress_file}")
        logger.info(f"Total tickers in universe: {len(ALL_TICKERS)}")
    
    def _load_progress(self) -> Dict:
        """Load fetch progress from disk"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'started': datetime.now().isoformat(),
            'last_updated': None,
            'fetched_tickers': [],
            'failed_tickers': [],
            'total_tickers': len(ALL_TICKERS),
            'total_quarters': 0,
            'total_api_calls': 0
        }
    
    def _save_progress(self):
        """Save fetch progress to disk"""
        self.progress['last_updated'] = datetime.now().isoformat()
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        
        logger.info(f"Progress saved: {len(self.progress['fetched_tickers'])}/{self.progress['total_tickers']} stocks")
    
    def get_pending_tickers(self, ticker_list: List[str] = None) -> List[str]:
        """Get list of tickers that haven't been fetched yet"""
        ticker_list = ticker_list or ALL_TICKERS
        
        fetched = set(self.progress['fetched_tickers'])
        failed = set(self.progress['failed_tickers'])
        
        # Pending = not fetched and not failed
        pending = [t for t in ticker_list if t not in fetched and t not in failed]
        
        logger.info(f"Pending tickers: {len(pending)}/{len(ticker_list)}")
        return pending
    
    def fetch_daily_batch(self, daily_limit: int = 10, specific_tickers: List[str] = None):
        """
        Fetch a daily batch of stocks
        
        Args:
            daily_limit: Maximum number of stocks to fetch (respects 25 API calls/day)
            specific_tickers: Optional list of specific tickers to fetch
        """
        logger.info("="*80)
        logger.info("DAILY BATCH FETCH - LSTM TRAINING DATA")
        logger.info("="*80)
        
        # Get pending tickers
        if specific_tickers:
            pending = [t for t in specific_tickers if t not in self.progress['fetched_tickers']]
        else:
            pending = self.get_pending_tickers()
        
        if not pending:
            logger.info("✅ All tickers have been fetched!")
            self._print_summary()
            return
        
        # Limit to daily batch size
        batch = pending[:daily_limit]
        
        logger.info(f"\n📊 Today's Batch:")
        logger.info(f"  Tickers to fetch: {len(batch)}")
        logger.info(f"  Estimated API calls: {len(batch) * 3}")
        logger.info(f"  Tickers: {', '.join(batch)}")
        logger.info(f"\n📈 Overall Progress:")
        logger.info(f"  Fetched: {len(self.progress['fetched_tickers'])}/{self.progress['total_tickers']}")
        logger.info(f"  Remaining: {len(pending)}")
        logger.info(f"  Failed: {len(self.progress['failed_tickers'])}")
        
        # Fetch batch
        success_count = 0
        failed_count = 0
        total_quarters = 0
        
        for i, ticker in enumerate(batch, 1):
            logger.info(f"\n{'─'*80}")
            logger.info(f"Processing {ticker} ({i}/{len(batch)})")
            logger.info(f"{'─'*80}")
            
            try:
                # Prepare training data
                data = self.fetcher.prepare_lstm_training_data(ticker, min_quarters=20, use_cache=True)
                
                if data is None:
                    logger.warning(f"Failed to prepare data for {ticker}")
                    self.progress['failed_tickers'].append(ticker)
                    failed_count += 1
                    continue
                
                # Success!
                self.progress['fetched_tickers'].append(ticker)
                self.progress['total_quarters'] += data['quarters']
                total_quarters += data['quarters']
                success_count += 1
                
                logger.info(f"✅ {ticker}: {data['quarters']} quarters fetched")
                
                # Save progress after each stock
                self._save_progress()
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                self.progress['failed_tickers'].append(ticker)
                failed_count += 1
                self._save_progress()
        
        # Update total API calls estimate
        self.progress['total_api_calls'] += (success_count * 3)
        self._save_progress()
        
        # Print batch summary
        logger.info("\n" + "="*80)
        logger.info("✅ BATCH COMPLETE")
        logger.info("="*80)
        logger.info(f"  Successful: {success_count}/{len(batch)}")
        logger.info(f"  Failed: {failed_count}/{len(batch)}")
        logger.info(f"  Quarters collected: {total_quarters}")
        logger.info(f"  API calls used: ~{success_count * 3}")
        logger.info(f"\n📊 Overall Progress:")
        logger.info(f"  Total fetched: {len(self.progress['fetched_tickers'])}/{self.progress['total_tickers']}")
        logger.info(f"  Completion: {len(self.progress['fetched_tickers'])/self.progress['total_tickers']*100:.1f}%")
        logger.info(f"  Total quarters: {self.progress['total_quarters']}")
        logger.info(f"  Total API calls: ~{self.progress['total_api_calls']}")
        
        # Days remaining estimate
        remaining = len(self.get_pending_tickers())
        days_remaining = remaining // daily_limit if daily_limit > 0 else 0
        logger.info(f"\n⏰ Estimated days remaining: {days_remaining} (at {daily_limit} stocks/day)")
        logger.info("="*80)
    
    def _print_status(self):
        """Print current collection status"""
        fetched = len(self.progress['fetched_tickers'])
        failed = len(self.progress['failed_tickers'])
        total = self.progress['total_tickers']
        pending = len(self.get_pending_tickers())
        completion = (fetched / total * 100) if total > 0 else 0
        
        print("\n" + "="*80)
        print("📊 COLLECTION STATUS")
        print("="*80)
        print(f"  Started: {self.progress.get('started', 'Not started')}")
        print(f"  Last updated: {self.progress.get('last_updated', 'Never')}")
        print(f"\n📈 Progress:")
        print(f"  ✅ Fetched: {fetched}/{total} stocks ({completion:.1f}%)")
        print(f"  ⏳ Pending: {pending} stocks")
        print(f"  ❌ Failed: {failed} stocks")
        print(f"\n📊 Data Collected:")
        print(f"  Total quarters: {self.progress['total_quarters']:,}")
        print(f"  API calls used: ~{self.progress['total_api_calls']}")
        
        if fetched > 0:
            print(f"\n📦 Recently Fetched:")
            recent = self.progress['fetched_tickers'][-5:]
            for ticker in recent:
                print(f"    - {ticker}")
        
        if failed > 0:
            print(f"\n⚠️ Failed Tickers (will retry):")
            for ticker in self.progress['failed_tickers'][:5]:
                print(f"    - {ticker}")
            if failed > 5:
                print(f"    ... and {failed - 5} more")
        
        if pending > 0:
            # Estimate days remaining (assume 8 stocks/day = 24 API calls)
            daily_capacity = 8
            days_remaining = (pending + daily_capacity - 1) // daily_capacity
            print(f"\n⏰ Estimated Timeline:")
            print(f"  Days remaining: ~{days_remaining} days (at {daily_capacity} stocks/day)")
            print(f"  Expected completion: ~{days_remaining} days from now")
        else:
            print(f"\n✅ Collection COMPLETE! Ready to create training dataset.")
        
        print("="*80)
    
    def _print_summary(self):
        """Print final collection summary"""
        logger.info("\n" + "="*80)
        logger.info("🎉 COLLECTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"  Started: {self.progress['started']}")
        logger.info(f"  Completed: {self.progress['last_updated']}")
        logger.info(f"  Total stocks: {self.progress['total_tickers']}")
        logger.info(f"  Fetched: {len(self.progress['fetched_tickers'])}")
        logger.info(f"  Failed: {len(self.progress['failed_tickers'])}")
        logger.info(f"  Total quarters: {self.progress['total_quarters']:,}")
        logger.info(f"  Total API calls: ~{self.progress['total_api_calls']}")
        logger.info(f"\n✅ Ready for LSTM training!")
        logger.info("="*80)
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """
        Create combined training dataset from all fetched stocks
        
        Returns:
            DataFrame with columns: [ticker, date, revenue_std, capex_std, da_std, nopat_std]
        """
        logger.info("Creating combined training dataset...")
        
        all_data = []
        
        for ticker in self.progress['fetched_tickers']:
            data = self.fetcher.prepare_lstm_training_data(ticker, use_cache=True)
            
            if data is None:
                continue
            
            # Extract standardized metrics
            std_df = data['standardized_data']
            
            training_df = pd.DataFrame({
                'ticker': ticker,
                'date': std_df['date'],
                'revenue_std': std_df['revenue_norm_std'],
                'capex_std': std_df['capex_norm_std'],
                'da_std': std_df['da_norm_std'],
                'nopat_std': std_df['nopat_norm_std']
            })
            
            all_data.append(training_df)
        
        if not all_data:
            logger.error("No data available for training!")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to disk
        output_file = self.output_dir / "lstm_growth_training_data.csv"
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Combined dataset created:")
        logger.info(f"  File: {output_file}")
        logger.info(f"  Records: {len(combined_df)}")
        logger.info(f"  Stocks: {combined_df['ticker'].nunique()}")
        logger.info(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Fetch LSTM training data from Alpha Vantage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 10 stocks per day (recommended)
  python scripts/fetch_lstm_training_data.py --daily-limit 10
  
  # Fetch specific stocks
  python scripts/fetch_lstm_training_data.py --tickers AAPL MSFT GOOGL
  
  # Resume with lower limit (if rate limited)
  python scripts/fetch_lstm_training_data.py --daily-limit 5
  
  # Create training dataset from fetched data
  python scripts/fetch_lstm_training_data.py --create-dataset
  
  # Show progress
  python scripts/fetch_lstm_training_data.py --status
        """
    )
    
    parser.add_argument(
        '--daily-limit',
        type=int,
        default=10,
        help='Number of stocks to fetch per day (default: 10, uses ~30 API calls)'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to fetch (optional)'
    )
    
    parser.add_argument(
        '--create-dataset',
        action='store_true',
        help='Create combined training dataset from fetched data'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show collection status and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = LSTMTrainingDataCollector()
    
    # Show status
    if args.status:
        collector._print_status()
        return
    
    # Create dataset
    if args.create_dataset:
        df = collector.create_combined_dataset()
        if not df.empty:
            print(f"\n✅ Training dataset ready: {len(df)} records")
        return
    
    # Fetch batch
    collector.fetch_daily_batch(
        daily_limit=args.daily_limit,
        specific_tickers=args.tickers
    )


if __name__ == "__main__":
    main()
