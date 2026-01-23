"""
Daily Qualified Update v2.0 - THREADED Stage 2 Pipeline

Optimized version with multi-threading for faster processing.
Runs pillar scoring on attention list stocks and updates qualified_stocks.

Key Optimizations (v2.0):
- Multi-threaded processing (8 workers by default)
- Batch processing with progress tracking
- Rate limiting protection (same as full_universe_scan)
- Resume capability for interrupted runs
- Reduced delay between stocks (100ms vs 300ms)

Performance Comparison:
- v1.0: ~12 hours for 4000 stocks (single-threaded, 300ms delay)
- v2.0: ~30-45 minutes for 4000 stocks (8 workers, 100ms delay)

Features:
- 5-Pillar Scoring v3.0 (Value, Quality, Growth, Safety, Momentum)
- Classification (Buy/Hold/Watch) with sub-categories
- Composite score threshold (>=50 to qualify)
- Pillar Excellence path (2+ pillars >= 65)

Usage:
    python scripts/pipeline/daily_qualified_update_v2.py
    python scripts/pipeline/daily_qualified_update_v2.py --workers 12
    python scripts/pipeline/daily_qualified_update_v2.py --ticker AAPL
    python scripts/pipeline/daily_qualified_update_v2.py --force-all
    python scripts/pipeline/daily_qualified_update_v2.py --resume
    python scripts/pipeline/daily_qualified_update_v2.py --limit 500  # Test mode
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
from src.analysis.pillar_scorer import PillarScorer
from src.analysis.quality_growth_gate import QualityGrowthGate
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)

# Progress file for resume capability
PROGRESS_FILE = CACHE_DIR / "qualified_update_progress.json"

# Threading defaults - optimized for yfinance
DEFAULT_MAX_WORKERS = 4  # Reduced from 8 to avoid rate limiting
DEFAULT_BATCH_DELAY_MS = 300  # Increased from 100ms for reliability
MAX_REQUESTS_PER_HOUR = 1500  # Conservative limit
THROTTLE_PAUSE_SECONDS = 60  # Pause when throttled


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


class ThreadedQualifiedUpdater:
    """
    Threaded Stage 2 Updater: Scores attention stocks using 5-pillar system.
    
    Multi-threaded processing for ~10x speedup over v1.0.
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
    
    def __init__(
        self,
        max_workers: int = DEFAULT_MAX_WORKERS,
        batch_delay_ms: int = DEFAULT_BATCH_DELAY_MS
    ):
        self.screener = StockScreener()
        self.scorer = PillarScorer()
        self.gate = QualityGrowthGate()
        self.max_workers = max_workers
        self.batch_delay_ms = batch_delay_ms
        
        self.stats = {
            "processed": 0,
            "qualified": 0,
            "disqualified": 0,
            "skipped": 0,
            "errors": 0
        }
        self.stats_lock = threading.Lock()
        self.errors = []
        self.classifications = {"buy": 0, "hold": 0, "watch": 0}
        
        # Rate limiting
        self.request_count = 0
        self.hour_start = None
        self.consecutive_errors = 0
        self.throttle_pause_seconds = THROTTLE_PAUSE_SECONDS
        self.start_time = None
    
    def detect_and_handle_throttle(self) -> bool:
        """Detect throttling (many consecutive errors) and pause."""
        if self.consecutive_errors >= 20:
            print(f"\n⚠️  Rate limiting detected ({self.consecutive_errors} consecutive errors)")
            print(f"   Pausing for {self.throttle_pause_seconds} seconds...")
            time.sleep(self.throttle_pause_seconds)
            self.consecutive_errors = 0
            self.throttle_pause_seconds = min(self.throttle_pause_seconds * 1.5, 300)
            print("   Resuming...")
            return True
        return False
    
    def load_progress(self) -> Dict:
        """Load progress for resume capability."""
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE) as f:
                    return json.load(f)
        except:
            pass
        return {"last_index": 0, "processed_tickers": [], "stats": None}
    
    def save_progress(self, index: int, processed_tickers: List[str]):
        """Save progress for resume."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, "w") as f:
                json.dump({
                    "last_index": index,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processed_tickers": processed_tickers,
                    "stats": self.stats
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def clear_progress(self):
        """Clear progress file."""
        try:
            if PROGRESS_FILE.exists():
                PROGRESS_FILE.unlink()
        except:
            pass
    
    def check_rate_limit(self):
        """Check rate limit and pause if needed."""
        now = datetime.utcnow()
        
        if self.hour_start is None:
            self.hour_start = now
            self.request_count = 0
        
        if (now - self.hour_start).total_seconds() >= 3600:
            self.hour_start = now
            self.request_count = 0
        
        if self.request_count >= MAX_REQUESTS_PER_HOUR * 0.9:
            wait_seconds = 3600 - (now - self.hour_start).total_seconds()
            if wait_seconds > 0:
                print(f"\n⚠️  Approaching rate limit. Pausing for {wait_seconds/60:.0f} minutes...")
                time.sleep(wait_seconds + 60)
                self.hour_start = datetime.utcnow()
                self.request_count = 0
                print("   Resuming...")
    
    def calculate_classification(self, composite_score: float, margin_of_safety: float) -> str:
        """Determine Buy/Hold/Watch classification (v3.0)."""
        if (margin_of_safety >= self.BUY_MOS_THRESHOLD and 
            composite_score >= self.BUY_SCORE_THRESHOLD):
            return "buy"
        
        if (self.HOLD_MOS_MIN <= margin_of_safety <= self.HOLD_MOS_MAX and
            composite_score >= self.HOLD_SCORE_THRESHOLD):
            return "hold"
        
        return "watch"
    
    def check_pillar_excellence(self, pillar_scores: Dict) -> Dict:
        """Check Pillar Excellence qualification path."""
        excellent_pillars = []
        for pillar, data in pillar_scores.items():
            score = data.get("score", 0) if isinstance(data, dict) else data
            if score >= self.PILLAR_EXCELLENCE_THRESHOLD:
                excellent_pillars.append(pillar)
        
        return {
            "qualified": len(excellent_pillars) >= self.PILLAR_EXCELLENCE_COUNT,
            "excellent_pillars": excellent_pillars,
            "excellent_count": len(excellent_pillars)
        }
    
    def check_momentum(self, data: Dict) -> Dict:
        """Check momentum status."""
        price_vs_200ma = data.get("price_vs_200ma")
        price_vs_50ma = data.get("price_vs_50ma")
        
        below_200ma = price_vs_200ma < 0 if price_vs_200ma is not None else None
        above_50ma = price_vs_50ma > 0 if price_vs_50ma is not None else None
        
        ideal_entry = None
        if below_200ma is not None and above_50ma is not None:
            ideal_entry = below_200ma and above_50ma
        
        return {
            "price_vs_200ma": round(price_vs_200ma, 2) if price_vs_200ma else None,
            "price_vs_50ma": round(price_vs_50ma, 2) if price_vs_50ma else None,
            "below_200ma": below_200ma,
            "above_50ma": above_50ma,
            "accumulation_zone": below_200ma,
            "stabilizing": above_50ma,
            "ideal_entry": ideal_entry
        }
    
    def process_stock(self, ticker: str, attention_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Process a single stock (thread-safe).
        
        Returns qualified stock document or None.
        """
        try:
            # Fetch data
            data = self.screener._fetch_stock_data(ticker)
            
            with self.stats_lock:
                self.request_count += 1
            
            if not data:
                with self.stats_lock:
                    self.stats["skipped"] += 1
                return None
            
            with self.stats_lock:
                self.stats["processed"] += 1
                self.consecutive_errors = 0
            
            # Calculate pillar scores (v3.0 with 5 pillars)
            scoring_result = self.scorer.calculate_composite(data)
            composite_score = scoring_result["composite_score"]
            pillar_scores = scoring_result["pillars"]
            
            # Check qualification paths
            path1_qualified = composite_score >= self.MIN_QUALIFIED_SCORE
            pillar_excellence = self.check_pillar_excellence(pillar_scores)
            path2_qualified = pillar_excellence["qualified"]
            
            if not (path1_qualified or path2_qualified):
                with self.stats_lock:
                    self.stats["disqualified"] += 1
                return None
            
            with self.stats_lock:
                self.stats["qualified"] += 1
            
            # Build qualification path info
            qualification_path = []
            if path1_qualified:
                qualification_path.append(f"composite>={self.MIN_QUALIFIED_SCORE}")
            if path2_qualified:
                qualification_path.append(f"pillar_excellence:{pillar_excellence['excellent_pillars']}")
            
            # Get margin of safety
            margin_of_safety = data.get("margin_of_safety", 0) or 0
            
            # Classification
            classification = self.calculate_classification(composite_score, margin_of_safety)
            
            with self.stats_lock:
                self.classifications[classification] += 1
            
            # Momentum
            momentum = self.check_momentum(data)
            
            # v3.0: Watch sub-category
            watch_sub_category = None
            if classification == "watch":
                watch_sub_category = scoring_result.get("watch_sub_category", "needs_research")
            
            # Triggers from attention data
            triggers = []
            if attention_data:
                triggers = [
                    t.get("type") + (f":path{t.get('path')}" if t.get("path") else "")
                    for t in attention_data.get("triggers", [])
                ]
            
            # Build document
            qualified_doc = {
                "ticker": ticker,
                "company_name": data.get("company_name"),
                "sector": data.get("sector"),
                "industry": data.get("industry"),
                
                # Scores
                "pillar_scores": pillar_scores,
                "composite_score": composite_score,
                "classification": classification,
                "watch_sub_category": watch_sub_category,
                
                # Qualification
                "qualification_path": qualification_path,
                "pillar_excellence": pillar_excellence if path2_qualified else None,
                
                # Valuation
                "current_price": data.get("current_price"),
                "fair_value": data.get("fair_value"),
                "lstm_fair_value": data.get("lstm_fair_value"),
                "margin_of_safety": margin_of_safety,
                
                # Key Metrics
                "pe_ratio": data.get("pe_ratio"),
                "fcf_yield": data.get("fcf_yield"),
                "fcf_roic": data.get("fcf_roic"),
                "roe": data.get("roe"),
                "profit_margin": data.get("profit_margin"),
                "gross_margin": data.get("gross_margin"),
                "debt_equity": data.get("debt_equity"),
                "revenue_growth": data.get("revenue_growth"),
                "earnings_growth": data.get("earnings_growth"),
                "beta": data.get("beta"),
                
                # Market
                "market_cap": data.get("market_cap"),
                "pct_from_52w_high": data.get("pct_from_52w_high"),
                
                # Momentum
                "momentum": momentum,
                
                # Triggers
                "triggers": triggers,
                
                # Analysis
                "analysis": self.scorer.get_strength_weakness_analysis(scoring_result),
                
                # Timestamp
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return qualified_doc
            
        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
                self.consecutive_errors += 1
                if len(self.errors) < 50:
                    self.errors.append(f"{ticker}: {str(e)[:100]}")
            return None
    
    def run_update(
        self,
        ticker: Optional[str] = None,
        force_all: bool = False,
        save_to_db: bool = True,
        resume: bool = False,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Run threaded qualified update.
        
        Args:
            ticker: Single ticker to process
            force_all: Process all attention stocks
            save_to_db: Save to MongoDB
            resume: Resume from last position
            limit: Limit stocks (testing)
        """
        self.start_time = datetime.utcnow()
        self.hour_start = self.start_time
        
        print("=" * 70)
        print("DAILY QUALIFIED UPDATE v2.0 (THREADED)")
        print("=" * 70)
        print(f"\nSettings:")
        print(f"  • Workers: {self.max_workers}")
        print(f"  • Batch Delay: {self.batch_delay_ms}ms")
        print(f"  • Qualification: Composite ≥{self.MIN_QUALIFIED_SCORE} OR 2+ pillars ≥65")
        
        # Get attention list
        if ticker:
            attention_list = [{"ticker": ticker.upper()}]
            print(f"\n📌 Single ticker mode: {ticker.upper()}")
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
        
        total = len(attention_list)
        if limit:
            attention_list = attention_list[:limit]
            print(f"⚠️  Limited to first {limit} of {total} stocks (test mode)")
        
        # Resume handling
        start_idx = 0
        processed_tickers = []
        if resume and not ticker:
            progress = self.load_progress()
            start_idx = progress.get("last_index", 0)
            processed_tickers = progress.get("processed_tickers", [])
            
            if start_idx > 0 and start_idx < len(attention_list):
                print(f"📌 Resuming from position {start_idx}/{len(attention_list)}")
        else:
            self.clear_progress()
        
        # Estimate time
        requests_per_second = self.max_workers / (0.5 + self.batch_delay_ms/1000)
        est_time_mins = len(attention_list) / requests_per_second / 60
        print(f"\nEstimated time: ~{est_time_mins:.0f} minutes")
        print()
        
        # Process in batches with threading
        qualified_stocks = []
        remaining = attention_list[start_idx:]
        processed_count = start_idx
        batch_size = 50  # Smaller batches for better throttle detection
        
        for batch_start in range(0, len(remaining), batch_size):
            self.check_rate_limit()
            self.detect_and_handle_throttle()  # Check for throttling
            
            batch = remaining[batch_start:batch_start + batch_size]
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_doc = {
                    executor.submit(self.process_stock, doc["ticker"], doc): doc
                    for doc in batch
                }
                
                for future in as_completed(future_to_doc):
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                    except Exception as e:
                        with self.stats_lock:
                            self.stats["errors"] += 1
                            self.consecutive_errors += 1
            
            # Save to MongoDB
            for result in batch_results:
                result = convert_numpy_types(result)
                qualified_stocks.append(result)
                
                if save_to_db and is_pipeline_available():
                    pipeline_db.upsert_qualified_stock(result)
                    pipeline_db.update_attention_status(result["ticker"], "graduated")
            
            # Progress
            processed_count += len(batch)
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (len(attention_list) - processed_count) / rate / 60 if rate > 0 else 0
            
            print(f"  [{processed_count:>5}/{len(attention_list)}] "
                  f"Qual: {self.stats['qualified']:>4} | "
                  f"Skip: {self.stats['skipped']:>3} | "
                  f"Err: {self.stats['errors']:>3} | "
                  f"Rate: {rate:.1f}/s | "
                  f"ETA: {eta:.0f}m")
            
            # Save progress
            processed_tickers.extend([doc["ticker"] for doc in batch])
            self.save_progress(processed_count, processed_tickers)
            
            # Batch delay
            if self.batch_delay_ms > 0:
                time.sleep(self.batch_delay_ms / 1000)
        
        # Final cleanup
        self.clear_progress()
        
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Summary
        print("\n" + "=" * 70)
        print("UPDATE COMPLETE")
        print("=" * 70)
        print(f"\n📊 STATISTICS")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Processed: {self.stats['processed']}")
        print(f"  Qualified: {self.stats['qualified']}")
        print(f"  Disqualified: {self.stats['disqualified']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"  Errors: {self.stats['errors']}")
        
        print(f"\n📈 CLASSIFICATIONS:")
        print(f"  Buy: {self.classifications['buy']}")
        print(f"  Hold: {self.classifications['hold']}")
        print(f"  Watch: {self.classifications['watch']}")
        
        # Qualification rate
        if self.stats['processed'] > 0:
            qual_rate = self.stats['qualified'] / self.stats['processed'] * 100
            print(f"\n  Qualification Rate: {qual_rate:.1f}%")
        
        # Top stocks
        if qualified_stocks:
            print(f"\n🏆 TOP 15 QUALIFIED STOCKS (by composite score):")
            sorted_stocks = sorted(qualified_stocks, key=lambda x: x["composite_score"], reverse=True)[:15]
            
            for i, stock in enumerate(sorted_stocks, 1):
                pillars = stock["pillar_scores"]
                # Handle both dict and float formats
                def get_score(p):
                    return p.get("score", 0) if isinstance(p, dict) else p
                
                print(f"  {i:>2}. {stock['ticker']:6s} {stock['composite_score']:>5.1f} "
                      f"({stock['classification'].upper():4s}) "
                      f"V:{get_score(pillars.get('value', {})):>3.0f} "
                      f"Q:{get_score(pillars.get('quality', {})):>3.0f} "
                      f"G:{get_score(pillars.get('growth', {})):>3.0f} "
                      f"S:{get_score(pillars.get('safety', {})):>3.0f} "
                      f"M:{get_score(pillars.get('momentum', {})):>3.0f}")
        
        # Get final qualified count
        if is_pipeline_available():
            all_qualified = pipeline_db.get_qualified_stocks(limit=10000)
            print(f"\n📋 Total in qualified collection: {len(all_qualified)}")
        
        return {
            "scan_type": "daily_qualified_v2",
            "duration_minutes": elapsed / 60,
            "stats": self.stats,
            "classifications": self.classifications,
            "qualified_count": len(qualified_stocks),
            "qualified_stocks": qualified_stocks
        }


def main():
    parser = argparse.ArgumentParser(description="Daily Qualified Update v2.0 (Threaded)")
    parser.add_argument("--ticker", type=str, help="Process single ticker")
    parser.add_argument("--force-all", action="store_true", help="Process all attention stocks")
    parser.add_argument("--no-save", action="store_true", help="Dry run (don't save to DB)")
    parser.add_argument("--resume", action="store_true", help="Resume from last position")
    parser.add_argument("--limit", type=int, help="Limit stocks (testing)")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                       help=f"Number of workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--delay", type=int, default=DEFAULT_BATCH_DELAY_MS,
                       help=f"Batch delay in ms (default: {DEFAULT_BATCH_DELAY_MS})")
    
    args = parser.parse_args()
    
    try:
        updater = ThreadedQualifiedUpdater(
            max_workers=args.workers,
            batch_delay_ms=args.delay
        )
        
        results = updater.run_update(
            ticker=args.ticker,
            force_all=args.force_all,
            save_to_db=not args.no_save,
            resume=args.resume,
            limit=args.limit
        )
        
        # Detailed single ticker output
        if args.ticker and results.get("qualified_stocks"):
            stock = results["qualified_stocks"][0]
            print("\n" + "=" * 70)
            print(f"DETAILED: {stock['ticker']}")
            print("=" * 70)
            print(f"Company: {stock['company_name']}")
            print(f"Sector: {stock['sector']}")
            print(f"Classification: {stock['classification'].upper()}")
            print(f"Composite Score: {stock['composite_score']:.1f}")
            print(f"Margin of Safety: {stock['margin_of_safety']:.1f}%")
            print(f"\nPillar Scores:")
            for name, data in stock["pillar_scores"].items():
                score = data.get("score", 0) if isinstance(data, dict) else data
                print(f"  {name.upper()}: {score:.1f}")
        
        print(f"\n✅ Update complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted. Use --resume to continue.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
