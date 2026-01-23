"""
Quick Start Script for Enhanced Data Collection
One command to validate, fetch, and prepare data for LSTM training

This script:
1. Validates Finnhub data format compatibility
2. Fetches initial batch (100 stocks) for quick start
3. Creates backtest dataset
4. Validates data quality
5. Prints next steps

Usage:
    # Quick start (100 stocks)
    python scripts/quick_start_data_collection.py

    # Full dataset (600 stocks, requires multiple runs over days)
    python scripts/quick_start_data_collection.py --full

    # Dry run (validation only)
    python scripts/quick_start_data_collection.py --dry-run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
from config.logging_config import get_logger

logger = get_logger(__name__)


def run_command(command: str, description: str):
    """Run a PowerShell command and handle errors"""
    logger.info(f"\n{'='*80}")
    logger.info(f"[STEP] {description}")
    logger.info(f"{'='*80}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors with '?'
        )
        
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning(f"Warnings:\n{result.stderr}")
        
        logger.info(f"[OK] {description} - SUCCESS")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAILED] {description} - FAILED")
        logger.error(f"Error: {e.stderr}")
        return False


def quick_start(batch_size: int = 100):
    """Quick start: Fetch 100 stocks for immediate LSTM training"""
    logger.info("="*80)
    logger.info("QUICK START - ENHANCED DATA COLLECTION")
    logger.info("="*80)
    logger.info(f"\nBatch size: {batch_size} stocks")
    logger.info(f"Estimated time: ~20 minutes")
    logger.info(f"Output: Time-series + Fundamentals + Backtest dataset")
    
    venv_python = r".\venv\Scripts\python.exe"
    
    # Step 1: Validate Finnhub format
    success = run_command(
        f"{venv_python} scripts\\validate_finnhub_format.py --ticker AAPL",
        "Step 1/5: Validating Finnhub data format"
    )
    
    if not success:
        logger.error("\n[FAILED] Finnhub validation failed! Check FINNHUB_API_KEY in .env")
        return False
    
    # Step 2: Fetch time-series
    success = run_command(
        f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode timeseries --batch-size {batch_size}",
        f"Step 2/5: Fetching time-series data ({batch_size} stocks)"
    )
    
    if not success:
        logger.error("\n[FAILED] Time-series fetch failed!")
        return False
    
    # Step 3: Fetch fundamentals
    fundamentals_batch = min(batch_size, 50)  # Cap at 50 for Alpha Vantage
    success = run_command(
        f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode fundamentals --batch-size {fundamentals_batch}",
        f"Step 3/5: Fetching quarterly fundamentals ({fundamentals_batch} stocks)"
    )
    
    if not success:
        logger.warning("\n[WARN] Fundamentals fetch incomplete (may hit Alpha Vantage limit)")
        logger.info("   Run again tomorrow to continue with Finnhub fallback")
    
    # Step 4: Create backtest dataset
    success = run_command(
        f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode backtest --lookback-years 5",
        "Step 4/5: Creating backtest dataset (5 years)"
    )
    
    if not success:
        logger.warning("\n[WARN] Backtest creation incomplete (needs both timeseries + fundamentals)")
    
    # Step 5: Validate
    run_command(
        f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode validate",
        "Step 5/5: Validating data quality"
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("[SUCCESS] QUICK START COMPLETE!")
    logger.info("="*80)
    logger.info("\n[DATA] Data Location:")
    logger.info("  Time-series: data/raw/timeseries/")
    logger.info("  Fundamentals: data/raw/fundamentals/")
    logger.info("  Backtest: data/processed/backtesting/lstm_dcf_backtest_data.csv")
    
    logger.info("\n[NEXT] Next Steps:")
    logger.info("  1. Check data quality:")
    logger.info(f"     {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode validate")
    logger.info("\n  2. Continue fetching (if needed):")
    logger.info(f"     {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode timeseries --batch-size 100")
    logger.info(f"     {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode fundamentals --batch-size 50")
    logger.info("\n  3. Train LSTM with new data:")
    logger.info(f"     {venv_python} scripts\\train_lstm_dcf.py --data comprehensive")
    logger.info("\n  4. Backtest model performance:")
    logger.info(f"     {venv_python} scripts\\backtest_lstm_dcf.py --lookback-years 5")
    logger.info("="*80)
    
    return True


def full_dataset():
    """Full dataset: Fetch all 600 stocks (requires multiple runs over days)"""
    logger.info("="*80)
    logger.info("FULL DATASET - 600 STOCKS")
    logger.info("="*80)
    logger.info("\n[WARN] NOTE: This requires multiple runs over 1-2 weeks")
    logger.info("   Alpha Vantage limit: 25 calls/day")
    logger.info("   Finnhub handles overflow")
    logger.info("\nRecommended workflow:")
    logger.info("  Day 1-6: Fetch all timeseries (6 batches × 100 stocks)")
    logger.info("  Day 1-12: Fetch all fundamentals (12 batches × 50 stocks)")
    logger.info("  Day 14: Create backtest dataset")
    
    venv_python = r".\venv\Scripts\python.exe"
    
    # Fetch timeseries (can do all in one day)
    for batch_num in range(1, 7):
        logger.info(f"\n{'='*80}")
        logger.info(f"📌 Timeseries Batch {batch_num}/6")
        logger.info(f"{'='*80}")
        
        success = run_command(
            f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode timeseries --batch-size 100",
            f"Fetching timeseries batch {batch_num}"
        )
        
        if not success:
            logger.error(f"\n❌ Batch {batch_num} failed!")
            break
    
    # Fetch fundamentals (respects Alpha Vantage limit)
    logger.info("\n" + "="*80)
    logger.info("[STEP] Starting Fundamentals Fetch")
    logger.info("="*80)
    logger.info("\n[WARN] This will hit Alpha Vantage limit after 25 stocks")
    logger.info("   Finnhub will handle remaining 25 stocks")
    logger.info("   Run this command daily to continue:")
    logger.info(f"   {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode fundamentals --batch-size 50")
    
    run_command(
        f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode fundamentals --batch-size 50",
        "Fetching fundamentals batch 1"
    )
    
    logger.info("\n" + "="*80)
    logger.info("[NEXT] NEXT STEPS FOR FULL DATASET")
    logger.info("="*80)
    logger.info("\nRun daily until complete:")
    logger.info(f"  {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode fundamentals --batch-size 50")
    logger.info("\nCheck progress:")
    logger.info(f"  {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode validate")
    logger.info("\nOnce complete, create backtest:")
    logger.info(f"  {venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode backtest --lookback-years 5")


def dry_run():
    """Dry run: Only validate, don't fetch"""
    logger.info("="*80)
    logger.info("DRY RUN - VALIDATION ONLY")
    logger.info("="*80)
    
    venv_python = r".\venv\Scripts\python.exe"
    
    # Step 1: Validate Finnhub (single stock)
    run_command(
        f"{venv_python} scripts\\validate_finnhub_format.py --ticker AAPL",
        "Validating Finnhub format (AAPL)"
    )
    
    # Step 2: Validate Finnhub (multiple stocks)
    run_command(
        f"{venv_python} scripts\\validate_finnhub_format.py --multi",
        "Validating Finnhub format (5 stocks)"
    )
    
    # Step 3: Check current progress
    run_command(
        f"{venv_python} scripts\\data_collection\\fetch_comprehensive_training_data.py --mode validate",
        "Checking current data quality"
    )
    
    logger.info("\n" + "="*80)
    logger.info("[SUCCESS] DRY RUN COMPLETE")
    logger.info("="*80)
    logger.info("\nIf validation passed, run:")
    logger.info(f"  {venv_python} scripts\\quick_start_data_collection.py")


def main():
    parser = argparse.ArgumentParser(
        description='Quick start for enhanced data collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start (100 stocks, ~20 mins)
  python scripts/quick_start_data_collection.py

  # Full dataset (600 stocks, requires multiple runs)
  python scripts/quick_start_data_collection.py --full

  # Validation only (no fetching)
  python scripts/quick_start_data_collection.py --dry-run

  # Custom batch size
  python scripts/quick_start_data_collection.py --batch-size 50
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Fetch full dataset (600 stocks, requires multiple runs over days)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validation only, no fetching'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for quick start (default: 100)'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run()
    elif args.full:
        full_dataset()
    else:
        quick_start(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
