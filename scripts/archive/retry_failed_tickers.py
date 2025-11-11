"""
Retry fetching failed tickers from Alpha Vantage
This script clears the failed_tickers list and retries them
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from scripts.fetch_lstm_training_data import LSTMTrainingDataCollector
from config.logging_config import get_logger

logger = get_logger(__name__)


def clear_failed_tickers():
    """Clear the failed tickers list to allow retry"""
    collector = LSTMTrainingDataCollector()
    
    failed_count = len(collector.progress['failed_tickers'])
    failed_list = collector.progress['failed_tickers'].copy()
    
    logger.info(f"Found {failed_count} failed tickers to retry")
    logger.info(f"Failed tickers: {', '.join(failed_list[:10])}")
    if failed_count > 10:
        logger.info(f"... and {failed_count - 10} more")
    
    # Clear failed list
    collector.progress['failed_tickers'] = []
    collector._save_progress()
    
    logger.info(f"✅ Cleared {failed_count} failed tickers - they will be retried on next fetch")
    
    return failed_list


def retry_batch(batch_size: int = 10):
    """Clear failed list and immediately retry a batch"""
    logger.info("="*80)
    logger.info("RETRYING FAILED TICKERS")
    logger.info("="*80)
    
    # Clear failed tickers
    failed_list = clear_failed_tickers()
    
    # Now fetch with the batch
    collector = LSTMTrainingDataCollector()
    
    logger.info(f"\nRetrying batch of {batch_size} tickers...")
    collector.fetch_daily_batch(daily_limit=batch_size)
    
    logger.info("\n" + "="*80)
    logger.info("RETRY COMPLETE")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Retry failed ticker fetches')
    parser.add_argument(
        '--clear-only',
        action='store_true',
        help='Only clear the failed list without fetching'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of tickers to retry (default: 10)'
    )
    
    args = parser.parse_args()
    
    if args.clear_only:
        clear_failed_tickers()
    else:
        retry_batch(args.batch_size)


if __name__ == "__main__":
    main()
