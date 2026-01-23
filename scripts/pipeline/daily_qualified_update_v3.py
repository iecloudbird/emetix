"""
Daily Qualified Update v3.0 - ULTRA RELIABLE with Session Rotation

Designed to handle yfinance rate limiting with:
- Fresh session per batch (rotates cookies/crumbs)
- Ultra-conservative rate limiting (1-2 requests/second)
- Aggressive throttle detection and backoff
- Sequential processing (most reliable)
- Auto-resume from any interruption

Why v3?
- v2 hit rate limits with 4+ parallel workers
- yfinance 401 errors caused 100% skip rate
- This version prioritizes reliability over speed

Expected Performance:
- ~1-2 stocks per second
- ~4000 stocks in 1-2 hours (not 15 min, but actually works)
- Can run overnight unattended

Usage:
    python scripts/pipeline/daily_qualified_update_v3.py --force-all
    python scripts/pipeline/daily_qualified_update_v3.py --resume
    python scripts/pipeline/daily_qualified_update_v3.py --ticker AAPL  # Single test
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.settings import CACHE_DIR
from src.analysis.pillar_scorer import PillarScorer
from src.analysis.stock_screener import StockScreener
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)

# Progress file for resume capability
PROGRESS_FILE = CACHE_DIR / "qualified_update_v3_progress.json"

# Ultra-conservative settings
DEFAULT_DELAY_MS = 800  # 800ms between stocks (~1.25 requests/sec)
BATCH_SIZE = 25  # Small batches
BATCH_PAUSE_SEC = 10  # 10 second pause between batches
ERROR_PAUSE_SEC = 30  # 30 second pause after error
THROTTLE_PAUSE_SEC = 120  # 2 minute pause when throttled
MAX_CONSECUTIVE_ERRORS = 5  # Trigger throttle pause after this many errors


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


def fresh_yfinance_session():
    """Create a fresh yfinance session to avoid stale cookies."""
    import yfinance as yf
    # Clear any cached session data
    try:
        yf.utils._USER_AGENT_HEADERS = None
    except:
        pass
    return yf


class ReliableQualifiedUpdater:
    """
    Ultra-reliable Stage 2 Updater: Prioritizes completion over speed.
    
    Designed to handle yfinance rate limiting gracefully.
    """
    
    # Qualification paths
    MIN_QUALIFIED_SCORE = 50
    PILLAR_EXCELLENCE_THRESHOLD = 65
    PILLAR_EXCELLENCE_COUNT = 2
    
    # Classification thresholds
    BUY_MOS_THRESHOLD = 20
    BUY_SCORE_THRESHOLD = 65
    HOLD_MOS_MIN = -10
    HOLD_MOS_MAX = 20
    HOLD_SCORE_THRESHOLD = 55
    
    def __init__(self, delay_ms: int = DEFAULT_DELAY_MS):
        self.delay_ms = delay_ms
        self.pillar_scorer = PillarScorer()
        self.stock_screener = StockScreener()  # For proper data fetching
        
        # Statistics
        self.stats = {
            "processed": 0,
            "qualified": 0,
            "skipped": 0,
            "errors": 0,
            "rate_limited": 0,
            "buy": 0,
            "hold": 0,
            "watch": 0
        }
        
        # Progress tracking
        self.processed_tickers = set()
        self.consecutive_errors = 0
        self.start_time = None
    
    def load_progress(self) -> set:
        """Load progress from checkpoint file."""
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE) as f:
                    data = json.load(f)
                    return set(data.get("processed", []))
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
        return set()
    
    def save_progress(self):
        """Save progress checkpoint."""
        try:
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({
                    "processed": list(self.processed_tickers),
                    "stats": self.stats,
                    "last_updated": datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def fetch_stock_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch stock data using StockScreener for properly formatted data.
        
        Returns dict with all fields needed by pillar_scorer.calculate_composite():
        - margin_of_safety, fcf_roic, volatility, pe_ratio, etc.
        """
        try:
            # Use StockScreener which formats data correctly for pillar_scorer
            data = self.stock_screener._fetch_stock_data(ticker)
            
            if not data:
                return None
            
            # Reset error counter on success
            self.consecutive_errors = 0
            return data
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Detect rate limiting
            if '401' in error_str or 'rate' in error_str or 'crumb' in error_str:
                self.stats["rate_limited"] += 1
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n⚠️ Rate limited! Pausing {THROTTLE_PAUSE_SEC}s...")
                    time.sleep(THROTTLE_PAUSE_SEC)
                    self.consecutive_errors = 0
                    # Create fresh session
                    fresh_yfinance_session()
                else:
                    time.sleep(ERROR_PAUSE_SEC)
            else:
                self.consecutive_errors += 1
                
            return None
    
    def calculate_mos(self, data: Dict) -> Optional[float]:
        """Calculate Margin of Safety (already in data from StockScreener)."""
        # StockScreener already calculates margin_of_safety
        return data.get('margin_of_safety')
    
    def classify_stock(self, composite: float, pillars: Dict, mos: Optional[float]) -> Dict:
        """Classify stock into Buy/Hold/Watch with sub-category."""
        mos = mos or 0
        
        # Buy classification
        if mos >= self.BUY_MOS_THRESHOLD and composite >= self.BUY_SCORE_THRESHOLD:
            return {"class": "Buy", "sub": "Strong", "emoji": "🟢"}
        elif mos >= self.BUY_MOS_THRESHOLD * 0.75 and composite >= self.BUY_SCORE_THRESHOLD * 0.9:
            return {"class": "Buy", "sub": "Moderate", "emoji": "🟢"}
        
        # Hold classification
        if self.HOLD_MOS_MIN <= mos <= self.HOLD_MOS_MAX and composite >= self.HOLD_SCORE_THRESHOLD:
            return {"class": "Hold", "sub": "Steady", "emoji": "🟡"}
        elif composite >= self.HOLD_SCORE_THRESHOLD:
            return {"class": "Hold", "sub": "Review", "emoji": "🟡"}
        
        # Watch classification with sub-categories
        if pillars.get("Growth", 0) >= 70:
            return {"class": "Watch", "sub": "Growth", "emoji": "🔵"}
        elif pillars.get("Value", 0) >= 70:
            return {"class": "Watch", "sub": "Value", "emoji": "🔵"}
        elif pillars.get("Momentum", 0) >= 70:
            return {"class": "Watch", "sub": "Momentum", "emoji": "🔵"}
        else:
            return {"class": "Watch", "sub": "Speculative", "emoji": "⚪"}
    
    def process_single_stock(self, ticker: str) -> Optional[Dict]:
        """Process a single stock through pillar scoring."""
        try:
            # Fetch data (using StockScreener for proper format)
            data = self.fetch_stock_data(ticker)
            if not data:
                self.stats["skipped"] += 1
                return None
            
            # Score with pillar scorer using calculate_composite (not score_stock!)
            result = self.pillar_scorer.calculate_composite(data)
            if not result:
                self.stats["skipped"] += 1
                return None
            
            # Get pillar scores and composite
            # calculate_composite returns: {pillars: {value: {score: X}, quality: {score: X}, ...}, composite_score: X}
            pillars_raw = result.get("pillars", {})
            pillars = {name: p.get("score", 0) for name, p in pillars_raw.items()}
            composite = result.get("composite_score", 0)
            
            # Get MoS from data (already calculated by StockScreener)
            mos = data.get('margin_of_safety', 0)
            
            # Check qualification
            high_pillars = sum(1 for p in pillars.values() if p >= self.PILLAR_EXCELLENCE_THRESHOLD)
            is_qualified = (composite >= self.MIN_QUALIFIED_SCORE or 
                          high_pillars >= self.PILLAR_EXCELLENCE_COUNT)
            
            if not is_qualified:
                self.stats["skipped"] += 1
                return None
            
            # Classify
            classification = self.classify_stock(composite, pillars, mos)
            
            # Update stats
            self.stats["qualified"] += 1
            self.stats[classification["class"].lower()] += 1
            
            # Build result
            qualified_doc = convert_numpy_types({
                "ticker": ticker,
                "pillar_scores": pillars,
                "composite_score": composite,
                "margin_of_safety_pct": mos,
                "classification": classification["class"],
                "sub_category": classification["sub"],
                "high_pillars": high_pillars,
                "qualified_at": datetime.utcnow(),
                "last_updated": datetime.utcnow(),
                # Basic company info (from StockScreener formatted data)
                "name": data.get("company_name", ""),
                "sector": data.get("sector", ""),
                "industry": data.get("industry", ""),
                "price": data.get("current_price"),
                "market_cap": data.get("market_cap"),
            })
            
            return qualified_doc
            
        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            self.stats["errors"] += 1
            return None
    
    def print_progress(self, current: int, total: int):
        """Print progress bar."""
        elapsed = time.time() - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        eta_min = int(eta / 60)
        
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"\r[{bar}] {current}/{total} | "
              f"Q:{self.stats['qualified']} B:{self.stats['buy']} H:{self.stats['hold']} W:{self.stats['watch']} | "
              f"Skip:{self.stats['skipped']} Err:{self.stats['errors']} | "
              f"ETA:{eta_min}m", end="", flush=True)
    
    def run(self,
            tickers: List[str] = None,
            force_all: bool = False,
            save_to_db: bool = True,
            resume: bool = False) -> Dict:
        """
        Run the reliable qualified update.
        
        Args:
            tickers: Specific tickers to process
            force_all: Process all attention stocks regardless of status
            save_to_db: Save results to MongoDB
            resume: Resume from checkpoint
        """
        print("=" * 60)
        print("DAILY QUALIFIED UPDATE v3.0 (ULTRA RELIABLE)")
        print("=" * 60)
        print(f"\nSettings:")
        print(f"  • Delay: {self.delay_ms}ms between stocks")
        print(f"  • Batch Size: {BATCH_SIZE} stocks")
        print(f"  • Batch Pause: {BATCH_PAUSE_SEC}s between batches")
        print(f"  • Throttle Pause: {THROTTLE_PAUSE_SEC}s on rate limit")
        
        # Get attention list
        if tickers:
            attention_list = [{"ticker": t} for t in tickers]
            print(f"\n📊 Processing {len(tickers)} specified stocks")
        else:
            if not is_pipeline_available():
                print("❌ Pipeline DB not available")
                return {"error": "Pipeline DB not available"}
            
            if force_all:
                attention_list = pipeline_db.get_attention_stocks(status=None, limit=10000)
            else:
                attention_list = pipeline_db.get_attention_stocks(status="active", limit=5000)
            
            print(f"\n📊 Found {len(attention_list)} attention stocks")
        
        if not attention_list:
            print("❌ No attention stocks to process")
            return {"error": "No attention stocks"}
        
        # Resume from checkpoint
        if resume:
            self.processed_tickers = self.load_progress()
            if self.processed_tickers:
                print(f"📂 Resuming: {len(self.processed_tickers)} already processed")
        
        # Filter out already processed
        all_tickers = [s.get("ticker") for s in attention_list if s.get("ticker")]
        remaining = [t for t in all_tickers if t not in self.processed_tickers]
        
        total = len(remaining)
        if total == 0:
            print("✅ All stocks already processed!")
            return self.stats
        
        # Estimate time
        est_seconds = total * (self.delay_ms / 1000) + (total // BATCH_SIZE) * BATCH_PAUSE_SEC
        est_minutes = int(est_seconds / 60)
        print(f"⏱️  Estimated time: ~{est_minutes} minutes")
        
        qualified_stocks = []
        self.start_time = time.time()
        
        print("\nProcessing...\n")
        
        try:
            for i, ticker in enumerate(remaining, 1):
                # Process stock
                result = self.process_single_stock(ticker)
                
                if result:
                    qualified_stocks.append(result)
                    
                    # Save to DB periodically
                    if save_to_db and len(qualified_stocks) % 10 == 0:
                        for doc in qualified_stocks[-10:]:
                            try:
                                pipeline_db.upsert_qualified_stock(doc)
                            except Exception as e:
                                logger.debug(f"DB save error: {e}")
                
                # Mark as processed
                self.processed_tickers.add(ticker)
                self.stats["processed"] += 1
                
                # Progress update
                if i % 10 == 0 or i == total:
                    self.print_progress(i, total)
                
                # Save checkpoint every batch
                if i % BATCH_SIZE == 0:
                    self.save_progress()
                    print(f"\n💾 Checkpoint saved. Pausing {BATCH_PAUSE_SEC}s...")
                    time.sleep(BATCH_PAUSE_SEC)
                    # Fresh session after each batch
                    fresh_yfinance_session()
                
                # Normal delay
                time.sleep(self.delay_ms / 1000)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted! Progress saved. Use --resume to continue.")
            self.save_progress()
            return self.stats
        
        # Final save
        if save_to_db and qualified_stocks:
            print("\n\n💾 Saving final results to database...")
            saved_count = 0
            for doc in qualified_stocks:
                try:
                    if pipeline_db.upsert_qualified_stock(doc):
                        saved_count += 1
                except Exception as e:
                    logger.debug(f"DB save error: {e}")
            print(f"   ✅ Saved {saved_count}/{len(qualified_stocks)} qualified stocks to MongoDB")
        
        # Clean up progress file
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        
        # Summary
        elapsed = time.time() - self.start_time
        print("\n")
        print("=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print(f"\n📊 Results:")
        print(f"  • Processed: {self.stats['processed']}")
        print(f"  • Qualified: {self.stats['qualified']}")
        print(f"  • Skipped: {self.stats['skipped']}")
        print(f"  • Errors: {self.stats['errors']}")
        print(f"  • Rate Limited: {self.stats['rate_limited']}")
        print(f"\n📈 Classification:")
        print(f"  • 🟢 Buy: {self.stats['buy']}")
        print(f"  • 🟡 Hold: {self.stats['hold']}")
        print(f"  • 🔵 Watch: {self.stats['watch']}")
        print(f"\n⏱️  Time: {int(elapsed/60)}m {int(elapsed%60)}s")
        
        return self.stats


def test_single_stock(ticker: str):
    """Test with a single stock to verify yfinance works."""
    print(f"Testing single stock: {ticker}")
    print("-" * 40)
    
    updater = ReliableQualifiedUpdater(delay_ms=0)
    result = updater.process_single_stock(ticker)
    
    if result:
        print(f"✅ SUCCESS: {ticker}")
        print(f"   Composite: {result['composite_score']:.1f}")
        print(f"   Classification: {result['classification']} ({result['sub_category']})")
        print(f"   Pillars: {result['pillar_scores']}")
        if result.get('margin_of_safety_pct'):
            print(f"   MoS: {result['margin_of_safety_pct']:.1f}%")
    else:
        print(f"❌ FAILED: Could not process {ticker}")
        print("   (yfinance may still be rate limited)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Daily Qualified Update v3.0 (Ultra Reliable)")
    parser.add_argument("--ticker", "-t", help="Process single ticker (test mode)")
    parser.add_argument("--force-all", action="store_true", help="Process all attention stocks")
    parser.add_argument("--fresh", action="store_true", help="Clear existing qualified stocks before running")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY_MS, help=f"Delay between stocks in ms (default: {DEFAULT_DELAY_MS})")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to MongoDB")
    
    args = parser.parse_args()
    
    # Single stock test mode
    if args.ticker:
        test_single_stock(args.ticker)
        return
    
    # Fresh start: clear existing qualified stocks
    if args.fresh:
        print("🗑️  Clearing existing qualified stocks for fresh start...")
        deleted = pipeline_db.clear_qualified_stocks()
        print(f"   Deleted {deleted} existing qualified stocks\n")
    
    # Run full update
    updater = ReliableQualifiedUpdater(delay_ms=args.delay)
    updater.run(
        force_all=args.force_all,
        save_to_db=not args.dry_run,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
