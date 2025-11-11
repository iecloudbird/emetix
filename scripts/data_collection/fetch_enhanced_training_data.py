"""
Enhanced LSTM Training Data Collection Script
Uses unified fetcher (Alpha Vantage + Finnhub) for maximum data coverage

Improvements over original:
- No more 25 calls/day limit bottleneck
- Can fetch 100+ stocks in one run
- Automatic fallback between data sources
- Better data quality through redundancy

Usage:
    # Fetch using both APIs (recommended)
    python scripts/fetch_enhanced_training_data.py --batch-size 50
    
    # Prefer Finnhub (faster, no limit)
    python scripts/fetch_enhanced_training_data.py --prefer-finnhub --batch-size 100
    
    # Prefer Alpha Vantage (better quality, but limited)
    python scripts/fetch_enhanced_training_data.py --prefer-alpha-vantage --batch-size 25
    
    # Create dataset from fetched data
    python scripts/fetch_enhanced_training_data.py --create-dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from typing import List
import pandas as pd

from src.data.fetchers.unified_financials_fetcher import UnifiedFinancialsFetcher
from config.settings import PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


# Expanded ticker universe (200+ tickers)
EXPANDED_TICKERS = [
    # S&P 500 Top 100
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'KO', 'AVGO', 'COST', 'LLY', 'TMO', 'WMT', 'MCD', 'ACN', 'CSCO', 'ABT',
    'DHR', 'VZ', 'ADBE', 'NKE', 'CRM', 'TXN', 'CMCSA', 'ORCL', 'INTC', 'DIS',
    'PM', 'NEE', 'BMY', 'UPS', 'T', 'QCOM', 'RTX', 'HON', 'LOW', 'UNP',
    'INTU', 'AMD', 'SPGI', 'BA', 'COP', 'SBUX', 'AMAT', 'DE', 'GS', 'CAT',
    'AMGN', 'AXP', 'BKNG', 'LMT', 'PLD', 'GILD', 'MDLZ', 'ADI', 'TJX', 'SYK',
    'MMC', 'ADP', 'CI', 'VRTX', 'BDX', 'CVS', 'ZTS', 'TMUS', 'CB', 'SO',
    'PGR', 'ISRG', 'DUK', 'REGN', 'MO', 'NOC', 'SLB', 'ITW', 'EOG', 'MMM',
    'CL', 'BSX', 'APD', 'EQIX', 'GE', 'SCHW', 'PNC', 'USB', 'BLK', 'CME',
    
    # NYSE Top 50
    'BAC', 'WFC', 'C', 'MS', 'COF', 'BK', 'STT', 'FITB', 'KEY', 'RF',
    'CFG', 'HBAN', 'CMA', 'ZION', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV',
    'LI', 'S', 'DISH', 'CHTR', 'LUMN', 'VOD', 'TEF', 'HAL', 'MRO', 'DVN',
    'APA', 'HES', 'OXY', 'FANG', 'MPC', 'VLO', 'PSX', 'PBF', 'DINO', 'HFC',
    
    # Tech Growth
    'SNOW', 'PLTR', 'RBLX', 'U', 'DASH', 'ABNB', 'COIN', 'DDOG', 'NET', 'CRWD',
    'ZS', 'OKTA', 'MDB', 'TWLO', 'DOCU', 'ZM', 'SHOP', 'SQ', 'PYPL', 'ROKU',
    
    # Healthcare
    'PFE', 'UNH', 'JNJ', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'REGN',
    'VRTX', 'BIIB', 'MRNA', 'ILMN', 'ISRG', 'EW', 'BDX', 'SYK', 'BSX', 'ZBH',
    
    # Financials
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'BLK', 'AXP', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'KEY', 'RF', 'CFG', 'HBAN',
    
    # Consumer
    'AMZN', 'WMT', 'HD', 'TGT', 'LOW', 'COST', 'TJX', 'DG', 'DLTR', 'ROST',
    'EBAY', 'ETSY', 'W', 'CHWY', 'BBBY', 'BBY', 'GPS', 'ANF', 'AEO', 'URBN'
]


class EnhancedDataCollector:
    """Enhanced collector using unified fetcher"""
    
    def __init__(self):
        self.fetcher = UnifiedFinancialsFetcher()
        self.progress_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "enhanced_fetch_progress.json"
        self.output_dir = PROCESSED_DATA_DIR / "lstm_dcf_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load progress
        self.progress = self._load_progress()
        
        logger.info("Enhanced Data Collector initialized")
        logger.info(f"Total ticker universe: {len(EXPANDED_TICKERS)}")
    
    def _load_progress(self):
        """Load fetch progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'started': datetime.now().isoformat(),
            'last_updated': None,
            'fetched_tickers': [],
            'failed_tickers': [],
            'source_stats': {'Alpha Vantage': 0, 'Finnhub': 0},
            'total_tickers': len(EXPANDED_TICKERS),
            'total_quarters': 0
        }
    
    def _save_progress(self):
        """Save progress"""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_pending_tickers(self) -> List[str]:
        """Get tickers that haven't been fetched"""
        fetched = set(self.progress['fetched_tickers'])
        return [t for t in EXPANDED_TICKERS if t not in fetched]
    
    def fetch_batch(
        self,
        batch_size: int = 50,
        prefer_source: str = 'alpha_vantage',
        alpha_vantage_limit: int = 25
    ):
        """
        Fetch a batch of stocks using unified fetcher
        
        Args:
            batch_size: Number of stocks to fetch
            prefer_source: 'alpha_vantage' or 'finnhub'
            alpha_vantage_limit: Max Alpha Vantage calls (default 25/day)
        """
        logger.info("="*80)
        logger.info("ENHANCED BATCH FETCH - MULTI-SOURCE")
        logger.info("="*80)
        
        pending = self.get_pending_tickers()
        
        if not pending:
            logger.info("✅ All tickers have been fetched!")
            self._print_summary()
            return
        
        batch = pending[:batch_size]
        
        logger.info(f"\n📊 Batch Configuration:")
        logger.info(f"  Batch size: {len(batch)} tickers")
        logger.info(f"  Prefer source: {prefer_source}")
        logger.info(f"  Alpha Vantage limit: {alpha_vantage_limit} calls/day")
        logger.info(f"  Finnhub limit: unlimited (60/min)")
        logger.info(f"\n📈 Overall Progress:")
        logger.info(f"  Fetched: {len(self.progress['fetched_tickers'])}/{self.progress['total_tickers']}")
        logger.info(f"  Remaining: {len(pending)}")
        
        # Fetch batch with smart source selection
        results = self.fetcher.fetch_batch_smart(
            batch,
            min_quarters=20,
            use_cache=True,
            alpha_vantage_limit=alpha_vantage_limit
        )
        
        # Update progress
        success_count = 0
        total_quarters = 0
        
        for ticker, data in results.items():
            self.progress['fetched_tickers'].append(ticker)
            self.progress['total_quarters'] += data['quarters']
            total_quarters += data['quarters']
            success_count += 1
            
            # Track source usage
            source = data.get('source', 'Unknown')
            self.progress['source_stats'][source] = self.progress['source_stats'].get(source, 0) + 1
        
        # Track failures
        failed_tickers = set(batch) - set(results.keys())
        self.progress['failed_tickers'].extend(list(failed_tickers))
        
        self._save_progress()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("✅ BATCH COMPLETE")
        logger.info("="*80)
        logger.info(f"  Successful: {success_count}/{len(batch)}")
        logger.info(f"  Failed: {len(failed_tickers)}/{len(batch)}")
        logger.info(f"  Quarters collected: {total_quarters:,}")
        logger.info(f"\n📊 Overall Progress:")
        logger.info(f"  Total fetched: {len(self.progress['fetched_tickers'])}/{self.progress['total_tickers']}")
        logger.info(f"  Completion: {len(self.progress['fetched_tickers'])/self.progress['total_tickers']*100:.1f}%")
        logger.info(f"  Total quarters: {self.progress['total_quarters']:,}")
        logger.info(f"\n📈 Data Sources:")
        for source, count in self.progress['source_stats'].items():
            logger.info(f"  {source}: {count} stocks")
        logger.info("="*80)
    
    def _print_summary(self):
        """Print collection summary"""
        logger.info("\n" + "="*80)
        logger.info("🎉 COLLECTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"  Total stocks: {len(self.progress['fetched_tickers'])}")
        logger.info(f"  Total quarters: {self.progress['total_quarters']:,}")
        logger.info(f"  Failed: {len(self.progress['failed_tickers'])}")
        logger.info(f"\n📈 Data Sources:")
        for source, count in self.progress['source_stats'].items():
            logger.info(f"  {source}: {count} stocks")
        logger.info("="*80)
    
    def create_dataset(self):
        """Create combined training dataset"""
        logger.info("Creating enhanced training dataset...")
        
        df = self.fetcher.create_combined_dataset(
            self.progress['fetched_tickers'],
            min_quarters=20
        )
        
        if df.empty:
            logger.error("No data available!")
            return
        
        # Save dataset
        output_file = self.output_dir / "lstm_growth_training_data_enhanced.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"\n✅ Enhanced dataset created:")
        logger.info(f"  File: {output_file}")
        logger.info(f"  Records: {len(df):,}")
        logger.info(f"  Stocks: {df['ticker'].nunique()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced LSTM training data fetcher (Alpha Vantage + Finnhub)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of stocks to fetch (default: 50, can be 100+ with Finnhub)'
    )
    
    parser.add_argument(
        '--prefer-finnhub',
        action='store_true',
        help='Prefer Finnhub (faster, no daily limit)'
    )
    
    parser.add_argument(
        '--prefer-alpha-vantage',
        action='store_true',
        help='Prefer Alpha Vantage (better quality, 25/day limit)'
    )
    
    parser.add_argument(
        '--alpha-vantage-limit',
        type=int,
        default=25,
        help='Alpha Vantage daily call limit (default: 25)'
    )
    
    parser.add_argument(
        '--create-dataset',
        action='store_true',
        help='Create combined training dataset from fetched data'
    )
    
    args = parser.parse_args()
    
    collector = EnhancedDataCollector()
    
    if args.create_dataset:
        collector.create_dataset()
        return
    
    # Determine preferred source
    prefer_source = 'alpha_vantage'
    if args.prefer_finnhub:
        prefer_source = 'finnhub'
    
    # Fetch batch
    collector.fetch_batch(
        batch_size=args.batch_size,
        prefer_source=prefer_source,
        alpha_vantage_limit=args.alpha_vantage_limit
    )


if __name__ == "__main__":
    main()
