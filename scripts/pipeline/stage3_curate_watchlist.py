"""
Stage 3: Watchlist Curation & Prioritization (v3.1)

Curates the qualified_stocks collection into actionable watchlists for web display.
Reduces 1000+ qualified stocks to top 50-150 picks with analyst-like justifications.

Key Features:
- Ranks stocks by composite score with tie-breakers (MoS, pillar excellence)
- Curates sub-lists: Top Buy (Strong/Moderate), Top Hold, Top Watch (by category)
- Generates AI-assisted analyst justifications for each pick
- Outputs JSON for frontend API consumption
- Caches results for fast web loading

Watchlist Targets (ideal for web display):
- Top 30 Buy (Strong + Moderate conviction)
- Top 30 Hold (Quality monitoring)
- Top 40 Watch (Categorized opportunities)
- Total: ~100 stocks (vs 1000+ raw qualified)

Usage:
    python scripts/pipeline/stage3_curate_watchlist.py
    python scripts/pipeline/stage3_curate_watchlist.py --output-json data/processed/watchlist.json
    python scripts/pipeline/stage3_curate_watchlist.py --buy-limit 20 --hold-limit 20
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.settings import PROCESSED_DATA_DIR
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)

# Curation thresholds
CURATION_CONFIG = {
    # List size limits (for web usability)
    "buy_strong_limit": 15,      # Strong Buy (MoS >= 35%, Score >= 75)
    "buy_moderate_limit": 15,    # Moderate Buy (MoS >= 25%, Score >= 70)
    "hold_limit": 30,            # Hold (quality monitoring)
    "watch_per_category": 10,    # Per watch sub-category
    
    # Strong Buy criteria (highest conviction)
    "strong_buy_mos": 35,        # MoS >= 35%
    "strong_buy_score": 75,      # Composite >= 75
    
    # Moderate Buy criteria
    "moderate_buy_mos": 25,      # MoS >= 25%
    "moderate_buy_score": 70,    # Composite >= 70
    
    # Tie-breaker weights for ranking
    "tiebreaker_weights": {
        "mos": 0.35,              # Higher MoS preferred
        "value_quality_sum": 0.35, # Sum of Value + Quality pillars
        "growth": 0.20,           # Growth pillar
        "safety": 0.10,           # Safety as secondary factor
    },
    
    # Pillar excellence threshold
    "pillar_excellence": 70,      # Score >= 70 = excellent
    
    # Watch sub-categories to curate
    "watch_categories": [
        "growth_watch",           # High growth potential, needs time
        "value_watch",            # Deep value, needs catalyst
        "momentum_watch",         # Technical breakout potential
        "speculative_watch",      # Higher risk/reward
    ]
}


def calculate_tiebreaker_score(stock: Dict) -> float:
    """
    Calculate tie-breaker score for ranking stocks with similar composites.
    
    Prioritizes:
    1. Higher Margin of Safety (value focus)
    2. Value + Quality pillar sum (compounder focus)
    3. Growth potential
    4. Safety as secondary
    
    Returns: 0-100 tie-breaker score
    """
    weights = CURATION_CONFIG["tiebreaker_weights"]
    
    pillars = stock.get("pillars") or stock.get("pillar_scores") or {}
    # Check both field names for margin of safety
    mos = stock.get("margin_of_safety_pct") or stock.get("margin_of_safety") or 0
    
    # Normalize MoS (0-100 scale, 50% MoS = 100)
    mos_score = min(100, max(0, mos * 2))
    
    # Extract pillar scores - handle nested structure
    def get_pillar_value(p):
        if isinstance(p, dict):
            return p.get("score", 0)
        return p if isinstance(p, (int, float)) else 0
    
    value_score = get_pillar_value(pillars.get("value", 0))
    quality_score = get_pillar_value(pillars.get("quality", 0))
    growth_score = get_pillar_value(pillars.get("growth", 0))
    safety_score = get_pillar_value(pillars.get("safety", 0))
    
    # Value + Quality sum (normalize to 0-100)
    vq_sum_score = min(100, (value_score + quality_score) / 2)
    
    # Calculate weighted tie-breaker
    tiebreaker = (
        mos_score * weights["mos"] +
        vq_sum_score * weights["value_quality_sum"] +
        growth_score * weights["growth"] +
        safety_score * weights["safety"]
    )
    
    return round(tiebreaker, 1)


def count_excellent_pillars(stock: Dict) -> int:
    """Count pillars with score >= excellence threshold."""
    pillars = stock.get("pillars") or stock.get("pillar_scores") or {}
    threshold = CURATION_CONFIG["pillar_excellence"]
    count = 0
    for pillar_name, pillar_score in pillars.items():
        # Handle nested pillar structure
        if isinstance(pillar_score, dict):
            score = pillar_score.get("score", 0)
        else:
            score = pillar_score if isinstance(pillar_score, (int, float)) else 0
        if score >= threshold:
            count += 1
    return count


def generate_justification(stock: Dict) -> Dict[str, str]:
    """
    Generate analyst-like justification for a stock pick.
    
    Returns:
        {
            "short": "One-liner for list display",
            "long": "Full paragraph for detail view"
        }
    """
    ticker = stock.get("ticker", "???")
    classification = stock.get("classification", "Watch")
    composite = stock.get("composite_score", 0)
    # Check both field names for margin of safety
    mos = stock.get("margin_of_safety_pct") or stock.get("margin_of_safety") or 0
    pillars = stock.get("pillars") or stock.get("pillar_scores") or {}
    triggers = stock.get("triggers", [])
    sector = stock.get("sector", "Unknown")
    # Get sub_category for meaningful labels
    sub_category = stock.get("sub_category") or stock.get("watch_sub_category") or ""
    
    # Extract pillar scores if nested structure
    pillar_values = {}
    for k, v in pillars.items():
        if isinstance(v, dict):
            pillar_values[k] = v.get("score", 0)
        else:
            pillar_values[k] = v if isinstance(v, (int, float)) else 0
    
    # Find strongest pillars
    sorted_pillars = sorted(
        [(k, v) for k, v in pillar_values.items() if v > 0],
        key=lambda x: x[1],
        reverse=True
    )
    top_pillars = sorted_pillars[:2] if sorted_pillars else []
    
    # Find trigger signals
    trigger_types = []
    for t in triggers:
        if isinstance(t, dict):
            trigger_types.append(t.get("type", ""))
        elif isinstance(t, str):
            trigger_types.append(t)
    
    # Build short justification based on sub_category label
    if classification == "Buy":
        if sub_category == "Strong":
            short = f"Top Quality Pick (Score {composite:.0f})"
        elif sub_category == "Steady":
            short = f"Steady Compounder (Score {composite:.0f})"
        elif mos >= 35:
            short = f"Deep Value (MoS {mos:.0f}%)"
        elif mos >= 20:
            short = f"Quality + Value (MoS {mos:.0f}%)"
        else:
            short = f"Quality Growth (Score {composite:.0f})"
        if top_pillars:
            short += f" | {top_pillars[0][0].title()}"
    elif classification == "Hold":
        if sub_category == "Steady":
            short = f"Quality Hold - Steady (Score {composite:.0f})"
        elif sub_category == "Review":
            short = f"Under Review (Score {composite:.0f})"
        else:
            short = f"Fair Value Monitor (Score {composite:.0f})"
        if top_pillars:
            short += f" | {top_pillars[0][0].title()}"
    else:
        # Watch category - use sub_category as label
        if sub_category == "Speculative":
            short = "Speculative Opportunity"
        elif sub_category == "Review":
            short = "Needs Research"
        elif "deep_value" in trigger_types:
            short = "Deep Value - Catalyst Needed"
        elif "consistent_growth" in trigger_types:
            short = "High Growth Reinvestor"
        elif "moat_strength" in trigger_types:
            short = "Durable Moat Detected"
        else:
            sub_label = sub_category.replace("_", " ").title() if sub_category else "Monitor"
            short = f"{sub_label} Candidate"
    
    # Build long justification
    long_parts = [f"**{ticker}** ({sector})"]
    
    if classification == "Buy":
        long_parts.append(
            f"Presents a compelling buying opportunity with {mos:.0f}% margin of safety "
            f"and composite score of {composite:.0f}. "
        )
    elif classification == "Hold":
        long_parts.append(
            f"Quality hold with composite score {composite:.0f} and near fair value "
            f"(MoS {mos:.0f}%). "
        )
    else:
        long_parts.append(
            f"Monitoring candidate with composite score {composite:.0f}. "
        )
    
    # Add pillar strengths
    if top_pillars:
        pillar_str = ", ".join([f"{p[0].title()} ({p[1]:.0f})" for p in top_pillars])
        long_parts.append(f"Key strengths: {pillar_str}. ")
    
    # Add trigger context
    if "quality_growth" in trigger_types:
        long_parts.append("Meets quality growth criteria (revenue + FCF ROIC). ")
    if "deep_value" in trigger_types:
        long_parts.append("Deep value with rate-adjusted FCF yield. ")
    if "significant_drop" in trigger_types:
        long_parts.append("Recovery candidate from significant pullback. ")
    if "moat_strength" in trigger_types:
        long_parts.append("Shows durable competitive moat. ")
    
    # Add watch-specific notes
    if classification == "Watch":
        watch_sub = stock.get("watch_sub_category", "")
        if watch_sub == "high_quality_expensive":
            long_parts.append("High quality but currently expensive - wait for better entry.")
        elif watch_sub == "cheap_junk":
            long_parts.append("Cheap valuation but quality concerns - avoid value trap.")
        elif watch_sub == "growth_watch":
            long_parts.append("Strong growth trajectory - monitor for confirmation.")
    
    return {
        "short": short,
        "long": " ".join(long_parts)
    }


def curate_watchlist(
    buy_strong_limit: int = None,
    buy_moderate_limit: int = None,
    hold_limit: int = None,
    watch_per_category: int = None
) -> Dict:
    """
    Curate qualified stocks into actionable watchlists.
    
    Args:
        buy_strong_limit: Max strong buy stocks (default: 15)
        buy_moderate_limit: Max moderate buy stocks (default: 15)
        hold_limit: Max hold stocks (default: 30)
        watch_per_category: Max watch per sub-category (default: 10)
        
    Returns:
        Curated watchlist with justifications
    """
    # Use defaults from config
    buy_strong_limit = buy_strong_limit or CURATION_CONFIG["buy_strong_limit"]
    buy_moderate_limit = buy_moderate_limit or CURATION_CONFIG["buy_moderate_limit"]
    hold_limit = hold_limit or CURATION_CONFIG["hold_limit"]
    watch_per_category = watch_per_category or CURATION_CONFIG["watch_per_category"]
    
    print("=" * 70)
    print("STAGE 3: WATCHLIST CURATION & PRIORITIZATION (v3.1)")
    print("=" * 70)
    
    # Fetch all qualified stocks from MongoDB
    print("\nFetching qualified stocks from database...")
    qualified = pipeline_db.get_qualified_stocks(limit=10000)
    
    if not qualified:
        print("ERROR: No qualified stocks found. Run Stage 2 first.")
        return {"error": "No qualified stocks"}
    
    print(f"  Found {len(qualified)} qualified stocks")
    
    # Enrich with tie-breaker scores and pillar excellence
    print("\nCalculating tie-breaker scores...")
    for stock in qualified:
        stock["tiebreaker_score"] = calculate_tiebreaker_score(stock)
        stock["excellent_pillars"] = count_excellent_pillars(stock)
    
    # Separate by classification
    buys = [s for s in qualified if s.get("classification") == "Buy"]
    holds = [s for s in qualified if s.get("classification") == "Hold"]
    watches = [s for s in qualified if s.get("classification") == "Watch"]
    
    print(f"  Raw counts: {len(buys)} Buy, {len(holds)} Hold, {len(watches)} Watch")
    
    # ===== CURATE BUYS =====
    # Sort by composite DESC, then tie-breaker DESC
    buys_sorted = sorted(
        buys,
        key=lambda x: (x.get("composite_score", 0), x.get("tiebreaker_score", 0)),
        reverse=True
    )
    
    # Separate Strong vs Moderate buys
    strong_buys = []
    moderate_buys = []
    
    for stock in buys_sorted:
        # Check both field names for margin of safety
        mos = stock.get("margin_of_safety_pct") or stock.get("margin_of_safety") or 0
        score = stock.get("composite_score", 0)
        
        if mos >= CURATION_CONFIG["strong_buy_mos"] and score >= CURATION_CONFIG["strong_buy_score"]:
            if len(strong_buys) < buy_strong_limit:
                stock["conviction"] = "Strong"
                stock["justification"] = generate_justification(stock)
                strong_buys.append(stock)
        elif len(moderate_buys) < buy_moderate_limit:
            stock["conviction"] = "Moderate"
            stock["justification"] = generate_justification(stock)
            moderate_buys.append(stock)
    
    # ===== CURATE HOLDS =====
    holds_sorted = sorted(
        holds,
        key=lambda x: (x.get("composite_score", 0), x.get("tiebreaker_score", 0)),
        reverse=True
    )[:hold_limit]
    
    for stock in holds_sorted:
        stock["conviction"] = "Monitor"
        stock["justification"] = generate_justification(stock)
    
    # ===== CURATE WATCHES (by sub-category) =====
    watch_curated = {}
    for category in CURATION_CONFIG["watch_categories"]:
        category_stocks = [
            s for s in watches
            if s.get("watch_sub_category") == category
        ]
        # Sort by composite + tie-breaker
        category_sorted = sorted(
            category_stocks,
            key=lambda x: (x.get("composite_score", 0), x.get("tiebreaker_score", 0)),
            reverse=True
        )[:watch_per_category]
        
        for stock in category_sorted:
            stock["conviction"] = category.replace("_", " ").title()
            stock["justification"] = generate_justification(stock)
        
        watch_curated[category] = category_sorted
    
    # ===== BUILD OUTPUT =====
    total_curated = (
        len(strong_buys) + len(moderate_buys) + 
        len(holds_sorted) + 
        sum(len(v) for v in watch_curated.values())
    )
    
    # Prepare output structure
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "curation_version": "3.1",
        "source_qualified_count": len(qualified),
        "curated_total": total_curated,
        "summary": {
            "strong_buy": len(strong_buys),
            "moderate_buy": len(moderate_buys),
            "hold": len(holds_sorted),
            "watch_by_category": {k: len(v) for k, v in watch_curated.items()}
        },
        "watchlists": {
            "strong_buy": _serialize_stocks(strong_buys),
            "moderate_buy": _serialize_stocks(moderate_buys),
            "hold": _serialize_stocks(holds_sorted),
            "watch": {k: _serialize_stocks(v) for k, v in watch_curated.items()}
        }
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("CURATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 CURATION SUMMARY")
    print(f"  Source: {len(qualified)} qualified stocks")
    print(f"  Curated: {total_curated} stocks ({(total_curated/len(qualified)*100):.1f}% of qualified)")
    print(f"\n  💰 STRONG BUY ({len(strong_buys)}):")
    for s in strong_buys[:5]:
        mos_val = s.get('margin_of_safety_pct') or s.get('margin_of_safety') or 0
        print(f"      {s['ticker']:6s} Score:{s['composite_score']:.0f} MoS:{mos_val:.0f}%")
    if len(strong_buys) > 5:
        print(f"      ... and {len(strong_buys)-5} more")
    
    print(f"\n  📈 MODERATE BUY ({len(moderate_buys)}):")
    for s in moderate_buys[:5]:
        mos_val = s.get('margin_of_safety_pct') or s.get('margin_of_safety') or 0
        print(f"      {s['ticker']:6s} Score:{s['composite_score']:.0f} MoS:{mos_val:.0f}%")
    if len(moderate_buys) > 5:
        print(f"      ... and {len(moderate_buys)-5} more")
    
    print(f"\n  ⏸️ HOLD ({len(holds_sorted)}):")
    for s in holds_sorted[:5]:
        print(f"      {s['ticker']:6s} Score:{s['composite_score']:.0f}")
    if len(holds_sorted) > 5:
        print(f"      ... and {len(holds_sorted)-5} more")
    
    print(f"\n  👁️ WATCH (by category):")
    for cat, stocks in watch_curated.items():
        print(f"      {cat}: {len(stocks)} stocks")
    
    return output


def _serialize_stocks(stocks: List[Dict]) -> List[Dict]:
    """Serialize stock list for JSON output (remove internal fields)."""
    serialized = []
    for s in stocks:
        # Get margin of safety (check both field names)
        mos = s.get("margin_of_safety_pct") or s.get("margin_of_safety") or 0
        
        # Get current price
        price = s.get("price") or s.get("current_price") or 0
        
        # Calculate fair value from price and MoS
        # MoS = (FV - Price) / Price * 100 => FV = Price * (1 + MoS/100)
        if s.get("fair_value"):
            fair_value = s.get("fair_value")
        elif s.get("lstm_fair_value"):
            fair_value = s.get("lstm_fair_value")
        elif price > 0:
            fair_value = price * (1 + mos / 100)
        else:
            fair_value = None
        
        # Get sub_category for label display
        sub_category = s.get("sub_category") or s.get("watch_sub_category")
        
        serialized.append({
            "ticker": s.get("ticker"),
            "composite_score": s.get("composite_score"),
            "margin_of_safety": mos,
            "fair_value": round(fair_value, 2) if fair_value else None,
            "current_price": round(price, 2) if price else None,
            "classification": s.get("classification"),
            "conviction": s.get("conviction"),
            "pillars": s.get("pillars") or s.get("pillar_scores"),
            "sector": s.get("sector"),
            "industry": s.get("industry"),
            "market_cap": s.get("market_cap"),
            "tiebreaker_score": s.get("tiebreaker_score"),
            "excellent_pillars": s.get("excellent_pillars"),
            "justification": s.get("justification"),
            "triggers": [t.get("type") if isinstance(t, dict) else t for t in s.get("triggers", [])],
            "watch_sub_category": s.get("watch_sub_category"),
            "sub_category": sub_category,  # For label display (Strong, Steady, Review, etc.)
        })
    return serialized


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Curate qualified stocks into web-ready watchlists"
    )
    parser.add_argument("--buy-strong-limit", type=int, default=15,
                       help="Max strong buy stocks (default: 15)")
    parser.add_argument("--buy-moderate-limit", type=int, default=15,
                       help="Max moderate buy stocks (default: 15)")
    parser.add_argument("--hold-limit", type=int, default=30,
                       help="Max hold stocks (default: 30)")
    parser.add_argument("--watch-per-category", type=int, default=10,
                       help="Max watch stocks per category (default: 10)")
    parser.add_argument("--output-json", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()
    
    if not is_pipeline_available():
        print("ERROR: Pipeline DB not available")
        return
    
    try:
        result = curate_watchlist(
            buy_strong_limit=args.buy_strong_limit,
            buy_moderate_limit=args.buy_moderate_limit,
            hold_limit=args.hold_limit,
            watch_per_category=args.watch_per_category
        )
        
        if "error" in result:
            print(f"\n❌ Curation failed: {result['error']}")
            return
        
        # Save to MongoDB for frontend API
        print("\n📤 Saving to MongoDB...")
        if pipeline_db.save_curated_watchlist(result):
            print("   ✅ Curated watchlist saved to MongoDB")
        else:
            print("   ⚠️ Failed to save to MongoDB (API will use fallback)")
        
        # Save to JSON as backup/offline reference
        output_path = args.output_json
        if not output_path:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PROCESSED_DATA_DIR / "curated_watchlist.json"
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n✅ Curated watchlist saved to: {output_path}")
        print(f"   Total: {result['curated_total']} stocks ready for web display")
        print(f"   API: GET /api/pipeline/curated")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
