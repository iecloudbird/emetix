"""
Weekly Attention Scan - Stage 1 of Quality Screening Pipeline (v2.2)

Scans the entire universe (~5800 stocks) for trigger conditions:
- Trigger A: Significant Drop (≥-40% from 52w high + FCF positive)
- Trigger B: Quality Growth Gate (4 paths: rev growth + FCF ROIC)
- Trigger C: Deep Value (MoS ≥30% AND FCF Yield ≥5%) - BOTH required
- Trigger D: Consistent Growth (3yr CAGR ≥20% + Gross Margin ≥30%)

Stocks that fire ANY trigger are added to the attention list.
Run weekly via scheduled task or manually.

Note: For full universe scan with rate limiting, use full_universe_scan.py
      This script is lighter-weight for quick scans or testing.

Usage:
    python scripts/pipeline/weekly_attention_scan.py
    python scripts/pipeline/weekly_attention_scan.py --tickers AAPL,MSFT,NVDA  # Test mode
    python scripts/pipeline/weekly_attention_scan.py --sector Technology
    python scripts/pipeline/weekly_attention_scan.py --limit 100  # Limit for testing
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from src.analysis.stock_screener import StockScreener
from src.analysis.quality_growth_gate import AttentionTriggers, QualityGrowthGate
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for MongoDB serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class WeeklyAttentionScanner:
    """
    Stage 1 Scanner: Identifies stocks that meet attention criteria.
    
    Processes the stock universe and evaluates each against 3 triggers.
    Stocks matching ANY trigger are added to attention_stocks collection.
    """
    
    def __init__(self):
        self.screener = StockScreener()
        self.gate = QualityGrowthGate()
        self.stats = {
            "scanned": 0,
            "triggered": 0,
            "errors": 0,
            "skipped": 0
        }
        self.errors = []
    
    def get_universe(
        self,
        tickers: Optional[List[str]] = None,
        sector: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get the list of tickers to scan.
        
        Priority:
        1. Explicit tickers list (for testing)
        2. From MongoDB universe
        3. Fallback to SP500 from screener
        """
        if tickers:
            return tickers
        
        # Try MongoDB universe first
        if is_pipeline_available():
            db_tickers = pipeline_db.get_universe_tickers(sector=sector)
            if db_tickers:
                logger.info(f"Loaded {len(db_tickers)} tickers from universe collection")
                if limit:
                    return db_tickers[:limit]
                return db_tickers
        
        # Fallback to SP500 from screener
        logger.warning("Universe collection empty, using SP500 fallback")
        sp500 = list(self.screener.SP500_TICKERS)
        
        if limit:
            return sp500[:limit]
        return sp500
    
    def evaluate_stock(self, ticker: str) -> Optional[Dict]:
        """
        Evaluate a single stock against all triggers (v2.2).
        
        Returns:
            Dict with ticker and triggers if any fired, None otherwise
        """
        try:
            # Fetch stock data using existing screener
            data = self.screener._fetch_stock_data(ticker)
            
            if not data:
                self.stats["skipped"] += 1
                return None
            
            # Evaluate all triggers (pass ticker for CAGR calculation in Trigger D)
            trigger_result = AttentionTriggers.evaluate_all_triggers(data, ticker=ticker)
            
            if trigger_result["any_triggered"]:
                return {
                    "ticker": ticker,
                    "triggers": trigger_result["triggers"],
                    "trigger_count": trigger_result["trigger_count"],
                    "data": {
                        "company_name": data.get("company_name"),
                        "sector": data.get("sector"),
                        "current_price": data.get("current_price"),
                        "market_cap": data.get("market_cap"),
                        "pct_from_52w_high": data.get("pct_from_52w_high"),
                        "revenue_growth": data.get("revenue_growth"),
                        "fcf_roic": data.get("fcf_roic"),
                        "margin_of_safety": data.get("margin_of_safety"),
                        "fcf_yield": data.get("fcf_yield"),
                    }
                }
            
            return None
            
        except Exception as e:
            self.stats["errors"] += 1
            self.errors.append(f"{ticker}: {str(e)}")
            logger.debug(f"Error evaluating {ticker}: {e}")
            return None
    
    def run_scan(
        self,
        tickers: Optional[List[str]] = None,
        sector: Optional[str] = None,
        limit: Optional[int] = None,
        save_to_db: bool = True,
        progress_callback: callable = None
    ) -> Dict:
        """
        Run the weekly attention scan.
        
        Args:
            tickers: Optional list of specific tickers to scan
            sector: Filter universe by sector
            limit: Limit number of stocks (for testing)
            save_to_db: Whether to save results to MongoDB
            progress_callback: Optional callback(current, total, ticker)
            
        Returns:
            Scan results summary
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("WEEKLY ATTENTION SCAN - Stage 1")
        logger.info("=" * 60)
        
        # Get universe
        universe = self.get_universe(tickers, sector, limit)
        total = len(universe)
        logger.info(f"Scanning {total} stocks...")
        
        # Log scan start
        scan_id = None
        if save_to_db and is_pipeline_available():
            scan_id = pipeline_db.log_scan_start("weekly_attention", total)
        
        # Process each stock
        triggered_stocks = []
        
        for i, ticker in enumerate(universe):
            self.stats["scanned"] += 1
            
            if progress_callback:
                progress_callback(i + 1, total, ticker)
            elif (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{total} ({(i+1)/total*100:.1f}%)")
            
            result = self.evaluate_stock(ticker)
            
            if result:
                self.stats["triggered"] += 1
                
                # Convert numpy types before saving
                result = convert_numpy_types(result)
                triggered_stocks.append(result)
                
                # Save to DB immediately for long-running scans
                if save_to_db and is_pipeline_available():
                    pipeline_db.upsert_attention_stock(
                        ticker=result["ticker"],
                        triggers=result["triggers"],
                        status="active"
                    )
            
            # Rate limiting - be nice to yfinance
            time.sleep(0.3)  # ~200 stocks per minute
        
        # Expire old attention stocks
        expired_count = 0
        if save_to_db and is_pipeline_available():
            expired_count = pipeline_db.bulk_expire_old_attention(days_old=30)
        
        # Log scan completion
        if scan_id and is_pipeline_available():
            pipeline_db.log_scan_complete(
                scan_id=scan_id,
                new_attention=self.stats["triggered"],
                expired=expired_count,
                errors=self.errors[:10]  # First 10 errors
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Build summary
        summary = {
            "scan_type": "weekly_attention",
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": round(duration, 1),
            "universe_size": total,
            "stats": self.stats,
            "new_attention": len(triggered_stocks),
            "expired": expired_count,
            "triggered_stocks": triggered_stocks,
            "error_sample": self.errors[:5]
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Scanned: {self.stats['scanned']}")
        logger.info(f"Triggered: {self.stats['triggered']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if triggered_stocks:
            logger.info("\nTriggered Stocks:")
            for stock in triggered_stocks[:20]:  # Show first 20
                triggers = [t.get("type", "?") for t in stock["triggers"]]
                logger.info(f"  {stock['ticker']}: {', '.join(triggers)}")
            
            if len(triggered_stocks) > 20:
                logger.info(f"  ... and {len(triggered_stocks) - 20} more")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Weekly Attention Scan - Stage 1 Pipeline"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to scan (test mode)"
    )
    parser.add_argument(
        "--sector",
        type=str,
        help="Filter by sector (e.g., Technology, Healthcare)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of stocks to scan (for testing)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to MongoDB (dry run)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-stock progress"
    )
    
    args = parser.parse_args()
    
    # Parse tickers if provided
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    
    # Progress callback for verbose mode
    def progress(current, total, ticker):
        if args.verbose:
            print(f"\r[{current}/{total}] Scanning {ticker}...", end="", flush=True)
    
    # Run scan
    scanner = WeeklyAttentionScanner()
    
    try:
        results = scanner.run_scan(
            tickers=tickers,
            sector=args.sector,
            limit=args.limit,
            save_to_db=not args.no_save,
            progress_callback=progress if args.verbose else None
        )
        
        if args.verbose:
            print()  # Newline after progress
        
        # Print detailed results for test mode
        if tickers or args.limit:
            print("\n" + "=" * 60)
            print("DETAILED RESULTS")
            print("=" * 60)
            for stock in results["triggered_stocks"]:
                print(f"\n{stock['ticker']} - {stock['data'].get('company_name', 'N/A')}")
                print(f"  Sector: {stock['data'].get('sector', 'N/A')}")
                print(f"  Price: ${stock['data'].get('current_price', 0):.2f}")
                print(f"  52W High: {stock['data'].get('pct_from_52w_high', 0):.1f}%")
                print(f"  Revenue Growth: {stock['data'].get('revenue_growth', 0):.1f}%")
                print(f"  FCF ROIC: {stock['data'].get('fcf_roic', 0):.1f}%")
                print(f"  MoS: {stock['data'].get('margin_of_safety', 0):.1f}%")
                print(f"  Triggers: {[t['type'] for t in stock['triggers']]}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
