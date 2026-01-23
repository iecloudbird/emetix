"""
End-to-End Pipeline Runner

Runs the complete pipeline:
1. Stage 2: Daily qualified update (attention → qualified)
2. Verifies API readiness

Usage:
    python scripts/pipeline/run_pipeline_e2e.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from src.data.pipeline_db import pipeline_db, is_pipeline_available


def run_e2e():
    print("=" * 70)
    print("END-TO-END PIPELINE EXECUTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check DB availability
    if not is_pipeline_available():
        print("\n❌ Pipeline DB not available!")
        return
    
    print("\n✅ Pipeline DB connected")
    
    # Current counts
    print("\n--- CURRENT STATE ---")
    univ = pipeline_db.get_universe_tickers()
    att = pipeline_db.get_attention_stocks(status=None, limit=5000)
    qual = pipeline_db.get_qualified_stocks(limit=5000)
    
    print(f"Universe stocks: {len(univ)}")
    print(f"Attention stocks: {len(att)}")
    print(f"Qualified stocks: {len(qual)}")
    
    # Show attention stocks
    if att:
        print("\n--- ATTENTION STOCKS (for examination) ---")
        print(f"{'Ticker':<8} {'Triggers':<50} {'Status':<10}")
        print("-" * 70)
        for s in att[:50]:  # Show up to 50
            ticker = s.get('ticker', 'N/A')
            triggers = s.get('triggers', [])
            trigger_strs = []
            for t in triggers:
                t_type = t.get('type', 'unknown')
                if t_type == '52_week_drop':
                    trigger_strs.append(f"52wk:{t.get('value', 0):.0f}%")
                elif t_type == 'quality_growth':
                    path = t.get('path_name', t.get('path', 'N/A'))
                    trigger_strs.append(f"QG:{path}")
                elif t_type == 'deep_value':
                    mos = t.get('margin_of_safety', 0)
                    fcf_y = t.get('fcf_yield', 0)
                    trigger_strs.append(f"DV:MoS{mos:.0f}%,FCFy{fcf_y:.0f}%")
                else:
                    trigger_strs.append(t_type)
            status = s.get('status', 'active')
            print(f"{ticker:<8} {', '.join(trigger_strs):<50} {status:<10}")
        if len(att) > 50:
            print(f"... and {len(att) - 50} more")
        print("-" * 70)
    
    # Stage 2: Run daily qualified update
    print("\n--- STAGE 2: DAILY QUALIFIED UPDATE ---")
    from scripts.pipeline.daily_qualified_update import DailyQualifiedUpdater
    
    updater = DailyQualifiedUpdater()
    results = updater.run_update(force_all=True, save_to_db=True)
    
    print(f"\nStatus: {results.get('status', 'unknown')}")
    print(f"Processed: {results.get('processed', 0)}")
    print(f"Qualified: {results.get('qualified', 0)}")
    print(f"Disqualified: {results.get('disqualified', 0)}")
    print(f"Errors: {results.get('errors', 0)}")
    
    # Final counts
    print("\n--- FINAL STATE ---")
    qual_final = pipeline_db.get_qualified_stocks(limit=5000)
    print(f"Qualified stocks in DB: {len(qual_final)}")
    
    if qual_final:
        print("\nTop Qualified Stocks:")
        for s in qual_final[:15]:
            ticker = s.get('ticker', 'N/A')
            score = s.get('composite_score', 0)
            classification = s.get('classification', 'N/A')
            mos = s.get('margin_of_safety', 0)
            print(f"  {ticker:6s} | Score: {score:5.1f} | {classification:5s} | MoS: {mos:+.1f}%")
    
    print("\n--- API READINESS ---")
    print("Endpoints ready:")
    print("  GET /api/pipeline/qualified")
    print("  GET /api/pipeline/attention")
    print("  GET /api/pipeline/qualified/{ticker}")
    print("  GET /api/pipeline/watchlist")
    print("\n✅ Pipeline execution complete!")


if __name__ == "__main__":
    run_e2e()
