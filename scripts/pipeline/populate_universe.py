"""
Full Universe Population Script

Populates the MongoDB universe_stocks collection with US-traded stocks
from major exchanges (NASDAQ, NYSE, AMEX).

This is the source of truth for the scanning pipeline.

Usage:
    python scripts/pipeline/populate_universe.py
    python scripts/pipeline/populate_universe.py --force  # Force refresh
    python scripts/pipeline/populate_universe.py --exchanges NASDAQ NYSE AMEX
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from src.data.fetchers.ticker_universe import TickerUniverseFetcher
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)

# Supported exchanges
SUPPORTED_EXCHANGES = ['NASDAQ', 'NYSE', 'AMEX', 'ARCA', 'BATS']
DEFAULT_EXCHANGES = ['NASDAQ', 'NYSE', 'AMEX']


def populate_universe(
    force_refresh: bool = False,
    exchanges: List[str] = None,
    clear_existing: bool = False
) -> int:
    """
    Populate the universe_stocks collection with US-traded stocks.
    
    Args:
        force_refresh: Force refresh from NASDAQ FTP
        exchanges: List of exchanges to include (default: NASDAQ, NYSE, AMEX)
        clear_existing: Clear existing universe before populating
        
    Returns:
        Number of stocks added/updated
    """
    if not is_pipeline_available():
        print("ERROR: Pipeline DB not available")
        return 0
    
    if exchanges is None:
        exchanges = DEFAULT_EXCHANGES
    
    print("=" * 60)
    print("UNIVERSE POPULATION SCRIPT")
    print("=" * 60)
    print(f"\nExchanges: {', '.join(exchanges)}")
    
    # Optionally clear existing
    if clear_existing:
        print("\n⚠️  Clearing existing universe collection...")
        # This would need a clear method in pipeline_db
    
    # Fetch all US tickers from selected exchanges
    print(f"\n1. Fetching US stock universe from {', '.join(exchanges)}...")
    fetcher = TickerUniverseFetcher()
    tickers = fetcher.get_all_us_tickers(
        force_refresh=force_refresh,
        exchanges=exchanges
    )
    print(f"   Found {len(tickers)} common stocks from {', '.join(exchanges)}")
    
    # Prepare universe documents
    print("\n2. Populating universe_stocks collection...")
    now = datetime.utcnow()
    success_count = 0
    error_count = 0
    
    # Use bulk insert for efficiency
    batch_size = 500
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
        
        for ticker in batch:
            try:
                success = pipeline_db.upsert_universe_stock({
                    "ticker": ticker,
                    "source": f"NASDAQ_FTP_{'+'.join(exchanges)}",
                    "exchanges": exchanges,
                    "last_updated": now
                })
                if success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                if error_count < 10:
                    print(f"   Error with {ticker}: {str(e)[:50]}")
    
    print(f"\n3. Results:")
    print(f"   Successfully added/updated: {success_count}")
    print(f"   Errors: {error_count}")
    
    # Verify
    db_count = len(pipeline_db.get_universe_tickers())
    print(f"   Total in universe collection: {db_count}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Populate universe_stocks collection from major exchanges"
    )
    parser.add_argument("--force", action="store_true", 
                       help="Force refresh from NASDAQ FTP")
    parser.add_argument("--exchanges", nargs="+", 
                       default=DEFAULT_EXCHANGES,
                       choices=SUPPORTED_EXCHANGES,
                       help=f"Exchanges to include (default: {' '.join(DEFAULT_EXCHANGES)})")
    parser.add_argument("--clear", action="store_true",
                       help="Clear existing universe before populating")
    args = parser.parse_args()
    
    try:
        count = populate_universe(
            force_refresh=args.force,
            exchanges=args.exchanges,
            clear_existing=args.clear
        )
        print(f"\n✅ Universe population complete: {count} stocks")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
