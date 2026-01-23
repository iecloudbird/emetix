"""
Full Universe Scan - Stage 1 Complete Scan (v3.0)

Scans the entire US stock universe (~5800+ stocks) for attention triggers.
Optimized for throughput with batch processing and progress tracking.

Key Features (v3.0):
- NO market cap/volume filtering - scans ALL stocks (diverse coverage)
- Stock categorization (Large Cap / Mid Cap / Small Cap / Micro Cap)
- Rate limiting protection (delays between batches to avoid yfinance throttle)
- ATH (All-Time High) drop trigger (-40%)
- Resume capability for interrupted scans
- RED FLAG VETO: Beneish M-Score detection for earnings manipulation
- SOLVENCY GUARDRAIL: Altman Z-Score + Interest Coverage for Trigger A
- DYNAMIC YIELD: Treasury-adjusted FCF yield threshold for Trigger C

Triggers (Selective - 5 triggers v3.0):
- A: Significant Drop (≥-40% from 52-week high + FCF positive + SOLVENCY OK)
- B: Quality Growth Gate (4 paths based on revenue growth + FCF ROIC)
- C: Deep Value (MoS ≥30% AND FCF Yield ≥ Treasury + 2%)
- D: Consistent Growth (3-year Revenue CAGR ≥20% + Gross Margin ≥30%)
- E: Moat Strength (Recurring Revenue > 70% OR Market Share Momentum > 5%) [NEW]

Red Flag Detection:
- Beneish M-Score > -2.22 = VETO (potential earnings manipulation)

Rate Limiting (yfinance protection):
- Max ~2000-2500 requests/hour recommended
- Default 200ms delay between batches
- Default 8 workers (conservative parallelism)

Note: Trigger D (CAGR) requires additional API calls for historical data,
      so it's only evaluated if other triggers don't fire (to save API calls).

Usage:
    python scripts/pipeline/full_universe_scan.py
    python scripts/pipeline/full_universe_scan.py --workers 8
    python scripts/pipeline/full_universe_scan.py --delay 300  # 300ms between batches
    python scripts/pipeline/full_universe_scan.py --resume  # Continue from last position
    python scripts/pipeline/full_universe_scan.py --limit 500  # Test with 500 stocks
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.settings import CACHE_DIR
from src.analysis.stock_screener import StockScreener
from src.analysis.quality_growth_gate import QualityGrowthGate, AttentionTriggers
from src.analysis.red_flag_detector import RedFlagDetector
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)

PROGRESS_FILE = CACHE_DIR / "full_scan_progress.json"

# Market Cap Categories (for labeling, not filtering!)
CAP_CATEGORIES = {
    "large_cap": 10_000_000_000,   # $10B+
    "mid_cap": 2_000_000_000,      # $2B - $10B
    "small_cap": 300_000_000,      # $300M - $2B
    "micro_cap": 0                  # < $300M
}

# Rate limiting settings (to avoid yfinance throttle)
# In practice: ~2000-2500 requests/hour from single IP
DEFAULT_BATCH_DELAY_MS = 200  # Delay between batches in milliseconds
DEFAULT_MAX_WORKERS = 8       # Conservative parallelism
MAX_REQUESTS_PER_HOUR = 2000  # yfinance practical limit


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


def categorize_market_cap(market_cap: float) -> str:
    """Categorize stock by market cap (no filtering, just labeling)."""
    if market_cap >= CAP_CATEGORIES["large_cap"]:
        return "large_cap"
    elif market_cap >= CAP_CATEGORIES["mid_cap"]:
        return "mid_cap"
    elif market_cap >= CAP_CATEGORIES["small_cap"]:
        return "small_cap"
    else:
        return "micro_cap"


def calculate_revenue_cagr(ticker: str, years: int = 3) -> Optional[Dict]:
    """
    Calculate multi-year revenue CAGR from yfinance annual financials.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years for CAGR calculation (default 3)
        
    Returns:
        Dict with CAGR info or None if insufficient data
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        
        # Get annual income statement (has revenue history)
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return None
        
        # Look for revenue row (different names in yfinance)
        revenue_row = None
        for row_name in ['Total Revenue', 'Revenue', 'Operating Revenue']:
            if row_name in income_stmt.index:
                revenue_row = income_stmt.loc[row_name]
                break
        
        if revenue_row is None:
            return None
        
        # Get revenue values sorted by date (oldest first)
        revenues = revenue_row.dropna().sort_index()
        
        if len(revenues) < 2:
            return None
        
        # Get latest and oldest (within years range)
        latest_revenue = revenues.iloc[-1]
        
        # Find revenue from ~years ago
        target_idx = max(0, len(revenues) - years - 1)
        oldest_revenue = revenues.iloc[target_idx]
        
        actual_years = len(revenues) - 1 - target_idx
        if actual_years < 2 or oldest_revenue <= 0 or latest_revenue <= 0:
            return None
        
        # Calculate CAGR: (ending/beginning)^(1/years) - 1
        cagr = ((latest_revenue / oldest_revenue) ** (1 / actual_years) - 1) * 100
        
        return {
            "cagr": round(cagr, 1),
            "years": actual_years,
            "latest_revenue": float(latest_revenue),
            "oldest_revenue": float(oldest_revenue)
        }
        
    except Exception as e:
        logger.debug(f"{ticker}: CAGR calculation failed: {e}")
        return None


class FullUniverseScanner:
    """
    Full universe scanner v3.1 with:
    - PRE-FILTER: Market cap >= $500M (quality filter, reduces micro-caps)
    - Stock categorization by market cap (large/mid/small)
    - Rate limiting protection
    - ATH drop trigger with solvency guardrail (tightened)
    - Trigger E: Moat Strength detection (no SaaS auto-assumption)
    - Red flag veto (Beneish M-Score)
    - Resume capability
    
    v3.1 Changes:
    - Added MIN_MARKET_CAP = $500M pre-filter (reduces ~70% of micro-caps)
    - Revenue filter removed (not available in yfinance info dict)
    - Tighter trigger thresholds (synced with quality_growth_gate.py v3.1)
    """
    
    # v3.1: Pre-filter to reduce volume and focus on quality
    MIN_MARKET_CAP = 500_000_000    # $500M minimum (was no filter)
    
    def __init__(
        self, 
        batch_size: int = 50, 
        max_workers: int = DEFAULT_MAX_WORKERS,
        batch_delay_ms: int = DEFAULT_BATCH_DELAY_MS,
        treasury_rate: float = 4.5,  # Current 10-year Treasury rate
        min_market_cap: int = None   # Override default pre-filter
    ):
        self.screener = StockScreener()
        self.gate = QualityGrowthGate()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.batch_delay_ms = batch_delay_ms
        self.treasury_rate = treasury_rate
        # v3.1: Configurable market cap pre-filter
        self.min_market_cap = min_market_cap if min_market_cap is not None else self.MIN_MARKET_CAP
        self.stats = {
            "scanned": 0,
            "triggered": 0,
            "vetoed": 0,        # v3.0: Red flag vetoes
            "prefiltered": 0,   # v3.1: Failed market cap pre-filter
            "skipped": 0,       # No data from yfinance
            "errors": 0,
            # Market cap breakdown
            "by_cap": {
                "large_cap": 0,
                "mid_cap": 0,
                "small_cap": 0,
                "micro_cap": 0
            },
            # Trigger breakdown (5 triggers v3.0)
            "by_trigger": {
                "significant_drop": 0,
                "quality_growth": 0,
                "deep_value": 0,
                "consistent_growth": 0,
                "moat_strength": 0  # v3.0 NEW
            }
        }
        self.stats_lock = threading.Lock()
        self.errors = []
        self.skipped_tickers = []  # Track skipped for retry
        self.vetoed_tickers = []   # v3.0: Track vetoed
        self.start_time = None
        self.request_count = 0
        self.hour_start = None
        self.consecutive_401_errors = 0  # Track throttling
        self.throttle_pause_minutes = 5  # Fixed 5-minute pause
        
    def detect_and_handle_throttle(self) -> bool:
        """
        Detect if we're being throttled and pause if needed.
        Returns True if we had to pause.
        Uses fixed 5-minute pause (no exponential backoff).
        """
        if self.consecutive_401_errors >= 10:
            print(f"\n⚠️  Detected rate limiting ({self.consecutive_401_errors} consecutive 401 errors)")
            print(f"   Pausing for {self.throttle_pause_minutes} minutes (fixed)...")
            time.sleep(self.throttle_pause_minutes * 60)
            self.consecutive_401_errors = 0
            # Fixed pause - no exponential backoff
            print("   Resuming scan...")
            return True
        return False
        
    def check_rate_limit(self):
        """Check if we're approaching rate limit and pause if needed."""
        now = datetime.utcnow()
        
        if self.hour_start is None:
            self.hour_start = now
            self.request_count = 0
        
        # Reset counter every hour
        if (now - self.hour_start).total_seconds() >= 3600:
            self.hour_start = now
            self.request_count = 0
        
        # If approaching limit, pause
        if self.request_count >= MAX_REQUESTS_PER_HOUR * 0.9:  # 90% of limit
            wait_seconds = 3600 - (now - self.hour_start).total_seconds()
            if wait_seconds > 0:
                print(f"\n⚠️  Approaching rate limit ({self.request_count} requests).")
                print(f"   Pausing for {wait_seconds/60:.0f} minutes until next hour...")
                time.sleep(wait_seconds + 60)  # Wait until next hour + buffer
                self.hour_start = datetime.utcnow()
                self.request_count = 0
                print("   Resuming scan...")
        
    def load_progress(self) -> Dict:
        """Load last scan progress for resume capability."""
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE) as f:
                    data = json.load(f)
                return data
        except:
            pass
        return {"last_index": 0, "skipped_tickers": [], "stats": None}
    
    def save_progress(self, index: int, stats: Dict, skipped_tickers: List[str] = None):
        """Save scan progress. Only saves if we've actually made progress."""
        try:
            # v3.1 FIX: Count all progress types (scanned + skipped + prefiltered)
            total_processed = (
                stats.get("scanned", 0) + 
                stats.get("skipped", 0) + 
                stats.get("prefiltered", 0)
            )
            if total_processed == 0:
                return  # Don't overwrite with empty progress
                
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, "w") as f:
                json.dump({
                    "last_index": index,
                    "timestamp": datetime.utcnow().isoformat(),
                    "stats": stats,
                    "skipped_tickers": skipped_tickers or []
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def clear_progress(self):
        """Clear progress file for fresh start."""
        try:
            if PROGRESS_FILE.exists():
                PROGRESS_FILE.unlink()
        except:
            pass
    
    def scan_stock(self, ticker: str) -> Optional[Dict]:
        """
        Scan a single stock for attention triggers (v3.1).
        Thread-safe for parallel processing.
        
        v3.1 PRE-FILTERS (applied before trigger evaluation):
        - Market cap >= $500M (configurable via min_market_cap)
        - Revenue >= $50M (configurable via min_revenue)
        
        v3.1 Enhancements:
        - Uses AttentionTriggers.evaluate_all_triggers() for unified checking
        - Red flag veto (Beneish M-Score)
        - Solvency guardrail for Trigger A (tightened: Z>2.0, ICR>4.0)
        - Trigger E: Moat Strength (no SaaS auto-assumption)
        
        Returns trigger dict if any trigger fires, None otherwise.
        Returns {"skipped": ticker} if no data available.
        Returns {"prefiltered": ticker} if fails pre-filters.
        Returns {"vetoed": ticker} if red flags detected.
        """
        try:
            # Fetch data
            data = self.screener._fetch_stock_data(ticker)
            
            with self.stats_lock:
                self.request_count += 1
            
            if not data:
                with self.stats_lock:
                    self.stats["skipped"] += 1
                    self.skipped_tickers.append(ticker)  # Track for retry
                    # Track consecutive skips (potential throttle)
                    self.consecutive_401_errors += 1
                return {"skipped": ticker}  # Return marker for skipped
            
            # Got valid data - reset throttle counter
            with self.stats_lock:
                self.consecutive_401_errors = 0
            
            # v3.1: PRE-FILTERS - Apply before trigger evaluation
            market_cap = data.get("market_cap") or 0
            
            # Check market cap filter (revenue filter disabled - not in yfinance info dict)
            if self.min_market_cap > 0 and market_cap < self.min_market_cap:
                with self.stats_lock:
                    self.stats["prefiltered"] += 1
                return {"prefiltered": ticker, "reason": f"Market cap ${market_cap/1e6:.0f}M < ${self.min_market_cap/1e6:.0f}M"}
            
            # Categorize by market cap (post pre-filter)
            cap_category = categorize_market_cap(market_cap)
            
            with self.stats_lock:
                self.stats["scanned"] += 1
                self.stats["by_cap"][cap_category] += 1
            
            # v3.0: Use unified trigger evaluation with red flag check
            trigger_result = AttentionTriggers.evaluate_all_triggers(
                stock_data=data,
                ticker=ticker,
                treasury_rate=self.treasury_rate,
                include_red_flags=True
            )
            
            # Handle red flag veto
            if trigger_result.get("vetoed"):
                with self.stats_lock:
                    self.stats["vetoed"] += 1
                    self.vetoed_tickers.append(ticker)
                return {"vetoed": ticker, "reason": trigger_result.get("veto_reason")}
            
            # Update trigger stats
            triggers_fired = trigger_result.get("triggers", [])
            for trigger in triggers_fired:
                trigger_type = trigger.get("type", "unknown")
                if trigger_type in self.stats["by_trigger"]:
                    with self.stats_lock:
                        self.stats["by_trigger"][trigger_type] += 1
            
            if triggers_fired:
                with self.stats_lock:
                    self.stats["triggered"] += 1
                return {
                    "ticker": ticker,
                    "triggers": triggers_fired,
                    "cap_category": cap_category,
                    "red_flags": trigger_result.get("red_flags", []),
                    "data_snapshot": {
                        "price": data.get("current_price"),
                        "sector": data.get("sector"),
                        "industry": data.get("industry"),
                        "market_cap": market_cap,
                        "cap_category": cap_category
                    }
                }
            
            return None
            
        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
                if len(self.errors) < 50:
                    self.errors.append({"ticker": ticker, "error": str(e)[:100]})
            return None
    
    def run(self, resume: bool = False, limit: Optional[int] = None) -> Dict:
        """
        Run the full universe scan with rate limiting protection.
        
        Args:
            resume: Resume from last position
            limit: Limit number of stocks (for testing)
            
        Returns:
            Scan results summary
        """
        if not is_pipeline_available():
            print("ERROR: Pipeline DB not available")
            return {"error": "Pipeline DB not available"}
        
        self.start_time = datetime.utcnow()
        self.hour_start = self.start_time
        
        # Get universe
        print("=" * 70)
        print("FULL UNIVERSE ATTENTION SCAN v3.1 (With Market Cap Filter + Tighter Triggers)")
        print("=" * 70)
        print(f"\nScan Settings:")
        print(f"  • PRE-FILTER (v3.1): Market Cap >= ${self.min_market_cap/1e6:.0f}M")
        print(f"  • Stocks categorized as: Large/Mid/Small Cap (micro-caps filtered)")
        print(f"  • Parallel Workers: {self.max_workers}")
        print(f"  • Batch Delay: {self.batch_delay_ms}ms (rate limit protection)")
        print(f"  • Treasury Rate: {self.treasury_rate}% (for dynamic FCF yield)")
        print(f"\nTriggers (Selective - 5 paths v3.1 TIGHTENED):")
        print(f"  A: Significant Drop (≥-50% + FCF positive + Solvency Z>2.0/ICR>4.0)")
        print(f"  B: Quality Growth Gate (4 paths: rev growth + FCF ROIC)")
        print(f"  C: Deep Value (MoS ≥40% AND FCF Yield ≥{self.treasury_rate + 2}%)")
        print(f"  D: Consistent Growth (3yr CAGR ≥20% + Gross Margin ≥30%)")
        print(f"  E: Moat Strength (Recurring Rev >75% OR Market Share +8%)")
        print(f"\n🚩 Red Flag Veto: Beneish M-Score > -2.22 = REJECT")
        
        tickers = pipeline_db.get_universe_tickers()
        if not tickers:
            print("\nERROR: Universe collection is empty. Run populate_universe.py first.")
            return {"error": "Empty universe"}
        
        total = len(tickers)
        if limit:
            tickers = tickers[:limit]
            print(f"\n⚠️  Limited to first {limit} of {total} stocks (testing mode)")
        
        print(f"\nScanning {len(tickers)} tickers from universe...")
        
        # Estimate time based on rate limiting
        requests_per_second = self.max_workers / (0.5 + self.batch_delay_ms/1000)
        est_time_mins = len(tickers) / requests_per_second / 60
        print(f"Estimated time: ~{est_time_mins:.0f} minutes (with rate limiting)")
        print()
        
        # Resume from last position if requested
        start_idx = 0
        previous_skipped = []
        if resume:
            progress = self.load_progress()
            start_idx = progress.get("last_index", 0)
            previous_skipped = progress.get("skipped_tickers", [])
            prev_stats = progress.get("stats")
            
            # v3.1 FIX: Check total processed (scanned + prefiltered + skipped)
            if prev_stats:
                total_processed = (
                    prev_stats.get("scanned", 0) + 
                    prev_stats.get("prefiltered", 0) + 
                    prev_stats.get("skipped", 0)
                )
                if total_processed > 0 and start_idx > 0 and start_idx < len(tickers):
                    print(f"📌 Resuming from position {start_idx}/{len(tickers)}")
                    print(f"   Previous: {prev_stats.get('scanned', 0)} scanned, "
                          f"{prev_stats.get('prefiltered', 0)} prefiltered, "
                          f"{prev_stats.get('triggered', 0)} triggered")
                    if previous_skipped:
                        print(f"   {len(previous_skipped)} skipped tickers will be retried at the end")
                else:
                    print(f"⚠️  Previous scan had no progress. Starting fresh.")
                    start_idx = 0
                    previous_skipped = []
                    self.clear_progress()
            else:
                start_idx = 0
                self.clear_progress()
        else:
            self.clear_progress()
        
        # Process stocks in parallel with rate limiting
        triggered_stocks = []
        remaining_tickers = tickers[start_idx:]
        processed_count = start_idx
        
        # Process in batches
        batch_size = 100
        
        for batch_start in range(0, len(remaining_tickers), batch_size):
            # Rate limit check
            self.check_rate_limit()
            
            # Check for throttling before batch
            self.detect_and_handle_throttle()
            
            batch = remaining_tickers[batch_start:batch_start + batch_size]
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self.scan_stock, ticker): ticker 
                    for ticker in batch
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        # Filter out skipped markers and None results
                        if result and isinstance(result, dict) and "ticker" in result:
                            batch_results.append(result)
                    except Exception as e:
                        with self.stats_lock:
                            self.stats["errors"] += 1
            
            # Save triggered stocks to MongoDB
            for result in batch_results:
                result = convert_numpy_types(result)
                pipeline_db.upsert_attention_stock(
                    ticker=result["ticker"],
                    triggers=result["triggers"],
                    status="active"
                )
                triggered_stocks.append(result["ticker"])
            
            # Update progress
            processed_count += len(batch)
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta_seconds = (len(tickers) - processed_count) / rate if rate > 0 else 0
            eta_mins = eta_seconds / 60
            
            # Detailed progress
            print(f"  [{processed_count:>5}/{len(tickers)}] "
                  f"Triggered: {self.stats['triggered']:>4} | "
                  f"Scanned: {self.stats['scanned']:>4} | "
                  f"Skip: {self.stats['skipped']:>3} | "
                  f"Err: {self.stats['errors']:>3} | "
                  f"Rate: {rate:.1f}/s | "
                  f"ETA: {eta_mins:.0f}m")
            
            # Save progress (with skipped tickers for later retry)
            self.save_progress(processed_count, self.stats, self.skipped_tickers)
            
            # Rate limiting delay between batches
            if self.batch_delay_ms > 0:
                time.sleep(self.batch_delay_ms / 1000)
        
        # Add any previously skipped tickers to current list for retry
        all_skipped = list(set(previous_skipped + self.skipped_tickers))
        
        # ========== RETRY PHASE ==========
        if all_skipped:
            print(f"\n{'='*70}")
            print(f"RETRY PHASE: {len(all_skipped)} skipped tickers")
            print("="*70)
            print("(Slower rate for reliability)")
            
            retry_success = 0
            retry_still_skipped = []
            
            for i, ticker in enumerate(all_skipped):
                # Slower retry: single-threaded with longer delay
                time.sleep(0.5)  # 500ms between each
                
                result = self.scan_stock(ticker)
                
                # Check if result is a triggered stock (has "ticker" key)
                if result and isinstance(result, dict) and "ticker" in result:
                    # Success - triggered stock, save to MongoDB
                    result = convert_numpy_types(result)
                    pipeline_db.upsert_attention_stock(
                        ticker=result["ticker"],
                        triggers=result["triggers"],
                        status="active"
                    )
                    triggered_stocks.append(result["ticker"])
                    retry_success += 1
                elif result and isinstance(result, dict) and ("skipped" in result or "prefiltered" in result or "vetoed" in result):
                    # Still skipped, prefiltered, or vetoed
                    retry_still_skipped.append(ticker)
                elif result is None:
                    # No trigger fired but data was fetched successfully
                    retry_success += 1
                else:
                    retry_still_skipped.append(ticker)
                
                if (i + 1) % 20 == 0:
                    print(f"  Retry progress: {i+1}/{len(all_skipped)} (success: {retry_success}, still skipped: {len(retry_still_skipped)})")
            
            print(f"\n  Retry complete: {retry_success} recovered, {len(retry_still_skipped)} still unavailable")
            if retry_still_skipped and len(retry_still_skipped) <= 50:
                print(f"  Still skipped: {retry_still_skipped}")
        
        # Final progress save (clear skipped since we retried)
        self.save_progress(len(tickers), self.stats, [])
        
        # Summary
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print(f"\n📊 SCAN STATISTICS (v3.1)")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  📋 Pre-filtered (cap/rev): {self.stats.get('prefiltered', 0)}")
        print(f"  ✓ Total Scanned (passed pre-filter): {self.stats['scanned']}")
        print(f"  ⚡ Triggered (added to attention): {self.stats['triggered']}")
        print(f"  🚩 Vetoed (red flags): {self.stats['vetoed']}")
        print(f"  ⏭️ Skipped (no yfinance data): {self.stats['skipped']}")
        print(f"  ❌ Errors: {self.stats['errors']}")
        
        # Pass rate calculation
        if self.stats['scanned'] > 0:
            pass_rate = (self.stats['triggered'] / self.stats['scanned']) * 100
            print(f"\n  📈 Trigger Pass Rate: {pass_rate:.1f}% (target: 10-20%)")
        
        print(f"\n📈 BY MARKET CAP:")
        for cap, count in self.stats["by_cap"].items():
            pct = (count / self.stats['scanned'] * 100) if self.stats['scanned'] > 0 else 0
            print(f"  {cap:12s}: {count:>5} ({pct:>5.1f}%)")
        
        print(f"\n⚡ BY TRIGGER TYPE:")
        for trigger, count in self.stats["by_trigger"].items():
            print(f"  {trigger:16s}: {count:>5}")
        
        if triggered_stocks:
            print(f"\n✅ Newly triggered stocks: {triggered_stocks[:30]}")
            if len(triggered_stocks) > 30:
                print(f"   ... and {len(triggered_stocks) - 30} more")
        
        # Get final attention count
        all_attention = pipeline_db.get_attention_stocks(status=None, limit=10000)
        print(f"\n📋 Total in attention collection: {len(all_attention)}")
        
        return {
            "scan_type": "full_universe_v2",
            "duration_minutes": elapsed / 60,
            "stats": self.stats,
            "triggered_count": len(triggered_stocks),
            "total_attention": len(all_attention)
        }


def main():
    parser = argparse.ArgumentParser(description="Full universe attention scan v3.1 (with pre-filters + tighter triggers)")
    parser.add_argument("--batch-size", type=int, default=50, 
                       help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                       help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--delay", type=int, default=DEFAULT_BATCH_DELAY_MS,
                       help=f"Delay between batches in ms (default: {DEFAULT_BATCH_DELAY_MS})")
    parser.add_argument("--treasury-rate", type=float, default=4.5,
                       help="Current 10-year Treasury rate for dynamic FCF yield (default: 4.5)")
    parser.add_argument("--min-market-cap", type=int, default=None,
                       help="Minimum market cap in dollars (default: 500M)")
    parser.add_argument("--no-prefilter", action="store_true",
                       help="Disable market cap pre-filter (scan all stocks like v3.0)")
    parser.add_argument("--fresh", action="store_true",
                       help="Clear existing attention stocks before scan (fresh start)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last position")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of stocks (for testing)")
    args = parser.parse_args()
    
    # Handle no-prefilter mode (backward compat with v3.0)
    min_market_cap = 0 if args.no_prefilter else args.min_market_cap
    
    try:
        # Fresh start: clear existing data
        if args.fresh:
            print("🗑️  Clearing existing attention stocks for fresh start...")
            deleted = pipeline_db.clear_attention_stocks()
            print(f"   Deleted {deleted} existing attention stocks\n")
        
        scanner = FullUniverseScanner(
            batch_size=args.batch_size, 
            max_workers=args.workers,
            batch_delay_ms=args.delay,
            treasury_rate=args.treasury_rate,
            min_market_cap=min_market_cap
        )
        results = scanner.run(resume=args.resume, limit=args.limit)
        print(f"\n✅ Full scan complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Scan interrupted. Use --resume to continue later.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
