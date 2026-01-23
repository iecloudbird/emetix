"""
Stock Diagnosis Script (v2.2)

Diagnose why a specific stock may or may not be in the attention/qualified lists.
Runs the complete evaluation pipeline on a single ticker and shows detailed output.

Pipeline Overview:
  Stage 1 (Attention): Triggered by any of 4 paths:
    - A: Significant Drop (≥-40% from 52w high + FCF positive)
    - B: Quality Growth Gate (4 paths: rev growth + FCF ROIC)
    - C: Deep Value (MoS ≥30% AND FCF Yield ≥5%)
    - D: Consistent Growth (3yr CAGR ≥20% + Gross Margin ≥30%)
    
  Stage 2 (Qualification): Pass either path:
    - Path 1: Composite Score ≥50 (4-pillar weighted average)
    - Path 2: Pillar Excellence (2+ pillars score ≥65%)

Usage:
    python scripts/pipeline/diagnose_stock.py AAPL
    python scripts/pipeline/diagnose_stock.py OSCR BABA PLTR
"""
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from src.analysis.stock_screener import StockScreener
from src.analysis.quality_growth_gate import QualityGrowthGate
from src.analysis.pillar_scorer import PillarScorer
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)


def format_number(value: Any, prefix: str = "", suffix: str = "") -> str:
    """Format a number for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000_000:
            return f"{prefix}{value / 1_000_000_000:.2f}B{suffix}"
        elif abs(value) >= 1_000_000:
            return f"{prefix}{value / 1_000_000:.2f}M{suffix}"
        elif abs(value) >= 1_000:
            return f"{prefix}{value / 1_000:.2f}K{suffix}"
        else:
            return f"{prefix}{value:.2f}{suffix}"
    return str(value)


def diagnose_stock(
    ticker: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full diagnostic on a single stock.
    
    Returns detailed evaluation results including:
    - Data fetch status
    - Market cap category (Large/Mid/Small/Micro)
    - Trigger evaluations (Significant Drop, Quality Growth, Deep Value)
    - Pillar scores (Value, Quality, Growth, Safety)
    - Qualification paths (Composite ≥50 OR 2+ Pillars ≥65)
    - Current status in pipeline
    """
    ticker = ticker.upper().strip()
    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "data_fetched": False,
        "filters": {},
        "triggers": [],
        "pillar_scores": None,
        "in_attention": False,
        "in_qualified": False,
        "diagnosis": []
    }
    
    screener = StockScreener()
    gate = QualityGrowthGate()
    scorer = PillarScorer()
    
    print("\n" + "=" * 70)
    print(f"STOCK DIAGNOSIS: {ticker}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Check current pipeline status
    # -------------------------------------------------------------------------
    print("\n📊 PIPELINE STATUS")
    print("-" * 40)
    
    if is_pipeline_available():
        # Check if in attention list
        attention_stocks = pipeline_db.get_attention_stocks(status=None, limit=5000)
        attention_tickers = [s.get("ticker") for s in attention_stocks]
        result["in_attention"] = ticker in attention_tickers
        
        # Check if in qualified list
        qualified_stocks = pipeline_db.get_qualified_stocks(limit=5000)
        qualified_tickers = [s.get("ticker") for s in qualified_stocks]
        result["in_qualified"] = ticker in qualified_tickers
        
        # Check if in universe
        universe_tickers = pipeline_db.get_universe_tickers()
        in_universe = ticker in universe_tickers
        
        print(f"  In Universe:    {'✅ Yes' if in_universe else '❌ No'}")
        print(f"  In Attention:   {'✅ Yes' if result['in_attention'] else '❌ No'}")
        print(f"  In Qualified:   {'✅ Yes' if result['in_qualified'] else '❌ No'}")
        
        if not in_universe:
            result["diagnosis"].append("Stock is NOT in universe collection - won't be scanned")
    else:
        print("  ⚠️ Pipeline DB not available")
    
    # -------------------------------------------------------------------------
    # Step 2: Fetch stock data
    # -------------------------------------------------------------------------
    print("\n📥 DATA FETCH")
    print("-" * 40)
    
    try:
        data = screener._fetch_stock_data(ticker)
        
        if not data:
            print(f"  ❌ Failed to fetch data for {ticker}")
            print("     Possible reasons:")
            print("     - Invalid ticker symbol")
            print("     - Delisted stock")
            print("     - yfinance API rate limit")
            print("     - Network issue")
            result["diagnosis"].append("Data fetch failed - stock won't be evaluated")
            return result
        
        result["data_fetched"] = True
        print(f"  ✅ Data fetched successfully")
        
    except Exception as e:
        print(f"  ❌ Error fetching data: {e}")
        result["diagnosis"].append(f"Data fetch error: {str(e)}")
        return result
    
    # -------------------------------------------------------------------------
    # Step 3: Display basic info
    # -------------------------------------------------------------------------
    print("\n📋 BASIC INFO")
    print("-" * 40)
    
    company_name = data.get("company_name") or data.get("name") or "Unknown"
    sector = data.get("sector") or "Unknown"
    industry = data.get("industry") or "Unknown"
    price = data.get("current_price") or data.get("price")
    market_cap = data.get("market_cap") or 0
    avg_volume = data.get("avg_volume") or data.get("volume") or 0
    
    print(f"  Company:      {company_name}")
    print(f"  Sector:       {sector}")
    print(f"  Industry:     {industry}")
    print(f"  Price:        ${price:.2f}" if price else "  Price:        N/A")
    print(f"  Market Cap:   {format_number(market_cap, '$')}")
    print(f"  Avg Volume:   {format_number(avg_volume)}")
    
    result["basic_info"] = {
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "price": price,
        "market_cap": market_cap,
        "avg_volume": avg_volume
    }
    
    # -------------------------------------------------------------------------
    # Step 4: Market Cap Category (no longer a filter)
    # -------------------------------------------------------------------------
    print("\n📊 MARKET CAP CATEGORY")
    print("-" * 40)
    
    # Categorize by market cap (informational only - no filtering)
    if market_cap >= 10_000_000_000:
        cap_category = "Large Cap ($10B+)"
    elif market_cap >= 2_000_000_000:
        cap_category = "Mid Cap ($2B-$10B)"
    elif market_cap >= 300_000_000:
        cap_category = "Small Cap ($300M-$2B)"
    else:
        cap_category = "Micro Cap (<$300M)"
    
    print(f"  Market Cap:    {format_number(market_cap, '$')}")
    print(f"  Category:      {cap_category}")
    print(f"  Note: Market cap is for categorization only, NOT a filter")
    
    result["filters"]["market_cap"] = {
        "value": market_cap,
        "category": cap_category,
        "note": "No minimum - all caps accepted"
    }
    
    # -------------------------------------------------------------------------
    # Step 5: Check attention triggers
    # -------------------------------------------------------------------------
    print("\n⚡ ATTENTION TRIGGERS")
    print("-" * 40)
    
    triggers_fired = []
    
    # Trigger A: Significant Drop (≥-40% + FCF positive)
    pct_from_high = data.get("pct_from_52w_high")
    week_52_high = data.get("52_week_high") or data.get("fiftyTwoWeekHigh")
    fcf = data.get("free_cash_flow") or data.get("freeCashflow") or 0
    trigger_a_pass = pct_from_high is not None and pct_from_high <= -40 and fcf > 0

    print(f"\n  [A] Significant Drop (≤-40% + FCF positive)")
    print(f"      52-Week High: ${week_52_high:.2f}" if week_52_high else "      52-Week High: N/A")
    print(f"      Current:      ${price:.2f}" if price else "      Current: N/A")
    print(f"      Drop:         {pct_from_high:.1f}%" if pct_from_high is not None else "      Drop: N/A")
    print(f"      FCF Positive: {'Yes' if fcf > 0 else 'No'} ({format_number(fcf, '$')})")
    print(f"      Status:       {'🔔 TRIGGERED' if trigger_a_pass else '⬜ Not triggered'}")

    if trigger_a_pass:
        triggers_fired.append({
            "type": "significant_drop",
            "value": pct_from_high,
            "threshold": -40,
            "fcf_positive": True
        })
    
    # Trigger B: Quality Growth Gate
    print(f"\n  [B] Quality Growth Gate")
    
    # Extract required values for gate evaluation
    revenue_growth = data.get("revenue_growth") or data.get("revenueGrowth") or 0
    # fcf already extracted for Trigger A
    total_assets = data.get("total_assets") or data.get("totalAssets") or 1
    current_liabilities = data.get("current_liabilities") or data.get("currentLiabilities") or 0
    
    # Calculate FCF ROIC
    fcf_roic = QualityGrowthGate.calculate_fcf_roic(fcf, total_assets, current_liabilities)
    
    print(f"      Revenue Growth: {revenue_growth:.1f}%")
    print(f"      FCF:            {format_number(fcf, '$')}")
    print(f"      FCF ROIC:       {fcf_roic:.1f}%")
    
    gate_result = gate.evaluate(
        revenue_growth=revenue_growth,
        fcf_roic=fcf_roic,
        free_cash_flow=fcf
    )
    trigger_b_pass = gate_result.get("passed", False)
    
    print(f"      Paths Matched:  {gate_result.get('paths_matched', [])}")
    print(f"      Best Path:      {gate_result.get('best_path_name', 'None')}")
    print(f"      Status:         {'🔔 TRIGGERED' if trigger_b_pass else '⬜ Not triggered'}")
    
    if gate_result.get("evaluation_details"):
        print("      Path Details:")
        for path_key, path_info in gate_result.get("evaluation_details", {}).items():
            path_status = "✓" if path_info.get("passed") else "✗"
            print(f"        [{path_status}] {path_info.get('name', path_key)}")
    
    if trigger_b_pass:
        triggers_fired.append({
            "type": "quality_growth",
            "path": gate_result.get("best_path"),
            "path_name": gate_result.get("best_path_name"),
            "revenue_growth": revenue_growth,
            "fcf_roic": fcf_roic
        })
    
    # Trigger C: Deep Value
    mos = data.get("margin_of_safety", 0) or 0
    fcf_yield = data.get("fcf_yield", 0) or 0
    fair_value = data.get("fair_value") or data.get("intrinsic_value")
    trigger_c_pass = mos >= 30 and fcf_yield >= 5  # AND instead of OR
    
    print(f"\n  [C] Deep Value (MoS≥30% AND FCF Yield≥5%)")
    print(f"      Fair Value:    ${fair_value:.2f}" if fair_value else "      Fair Value: N/A")
    print(f"      Margin of Safety: {mos:.1f}% {'(≥ 30%)' if mos >= 30 else '(< 30%)'}")
    print(f"      FCF Yield:     {fcf_yield:.1f}% {'(≥ 5%)' if fcf_yield >= 5 else '(< 5%)'}")
    print(f"      Status:        {'🔔 TRIGGERED' if trigger_c_pass else '⬜ Not triggered'}")
    
    if trigger_c_pass:
        triggers_fired.append({
            "type": "deep_value",
            "margin_of_safety": mos,
            "fcf_yield": fcf_yield
        })
    
    # Trigger D: Consistent Growth (CAGR-based)
    # For growth stocks reinvesting heavily (may have negative FCF)
    gross_margin = data.get("gross_margin", 0) or 0
    total_revenue = data.get("total_revenue") or data.get("revenue") or 0
    
    print(f"\n  [D] Consistent Growth (3yr CAGR≥20% + Gross Margin≥30%)")
    print(f"      Gross Margin:  {gross_margin:.1f}% {'(≥ 30%)' if gross_margin >= 30 else '(< 30%)'}")
    print(f"      Revenue:       {format_number(total_revenue, '$')}")
    
    trigger_d_pass = False
    cagr_info = None
    
    # Only check CAGR if basic requirements met
    if gross_margin >= 30 and total_revenue >= 50_000_000:
        # Import and calculate CAGR
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            income_stmt = stock.income_stmt
            
            if income_stmt is not None and not income_stmt.empty:
                revenue_row = None
                for row_name in ['Total Revenue', 'Revenue', 'Operating Revenue']:
                    if row_name in income_stmt.index:
                        revenue_row = income_stmt.loc[row_name]
                        break
                
                if revenue_row is not None:
                    revenues = revenue_row.dropna().sort_index()
                    if len(revenues) >= 2:
                        latest_rev = revenues.iloc[-1]
                        oldest_idx = max(0, len(revenues) - 4)  # ~3 years
                        oldest_rev = revenues.iloc[oldest_idx]
                        years = len(revenues) - 1 - oldest_idx
                        
                        if years >= 2 and oldest_rev > 0 and latest_rev > 0:
                            cagr = ((latest_rev / oldest_rev) ** (1 / years) - 1) * 100
                            cagr_info = {"cagr": cagr, "years": years}
                            trigger_d_pass = cagr >= 20
        except Exception as e:
            print(f"      CAGR calc error: {e}")
    
    if cagr_info:
        print(f"      3-Year CAGR:   {cagr_info['cagr']:.1f}% over {cagr_info['years']} years {'(≥ 20%)' if cagr_info['cagr'] >= 20 else '(< 20%)'}")
    else:
        print(f"      3-Year CAGR:   N/A (requires Gross Margin≥30% + Rev≥$50M)")
    
    print(f"      Status:        {'🔔 TRIGGERED' if trigger_d_pass else '⬜ Not triggered'}")
    
    if trigger_d_pass:
        triggers_fired.append({
            "type": "consistent_growth",
            "cagr": cagr_info["cagr"],
            "years": cagr_info["years"],
            "gross_margin": gross_margin
        })
    
    result["triggers"] = triggers_fired
    any_trigger = len(triggers_fired) > 0
    
    print(f"\n  Triggers Fired:     {len(triggers_fired)}/4")
    print(f"  Would Add to Attention: {'✅ YES' if any_trigger else '❌ NO'}")
    
    if not any_trigger:
        result["diagnosis"].append("No attention triggers fired - stock won't be added to attention list")
    
    # -------------------------------------------------------------------------
    # Step 6: Pillar Scoring (Stage 2)
    # -------------------------------------------------------------------------
    print("\n📊 PILLAR SCORING (Stage 2)")
    print("-" * 40)
    
    try:
        scoring_result = scorer.calculate_composite(data)
        
        if scoring_result:
            result["pillar_scores"] = scoring_result
            composite = scoring_result.get("composite_score", 0)
            
            print(f"\n  Composite Score: {composite:.1f}/100")
            print()
            
            pillars = scoring_result.get("pillars", {})
            for pillar_name, pillar_data in pillars.items():
                score = pillar_data.get("score", 0)
                weight = pillar_data.get("weight", 0)
                print(f"  {pillar_name.upper():12s}: {score:5.1f} (weight: {weight*100:.0f}%)")
                
                # Show component details
                components = pillar_data.get("components", {})
                for comp_name, comp_val in components.items():
                    if isinstance(comp_val, (int, float)):
                        print(f"    └─ {comp_name}: {comp_val:.2f}")
            
            # Classification (using new thresholds)
            print(f"\n  Classification: ", end="")
            if composite >= 65 and mos >= 20:
                print("🟢 BUY")
            elif composite >= 65 and mos >= -10:
                print("🟡 HOLD")
            else:
                print("⚪ WATCH")
            
            # Pillar Excellence check
            excellent_pillars = []
            for pillar_name, pillar_data in pillars.items():
                score = pillar_data.get("score", 0)
                if score >= 65:
                    excellent_pillars.append(pillar_name)
            
            pillar_excellence = len(excellent_pillars) >= 2
            
            # Qualification via two paths
            path1_qualified = composite >= 50  # Lowered from 60
            path2_qualified = pillar_excellence
            qualified = path1_qualified or path2_qualified
            
            print(f"\n  Qualification Paths:")
            print(f"    Path 1 (Composite ≥50): {composite:.1f} {'✅' if path1_qualified else '❌'}")
            print(f"    Path 2 (2+ Pillars ≥65): {excellent_pillars if excellent_pillars else 'None'} {'✅' if path2_qualified else '❌'}")
            print(f"  Would Qualify: {'✅ YES' if qualified else '❌ NO'}")
            
            if not qualified:
                result["diagnosis"].append(f"Composite score {composite:.1f} < 50 AND less than 2 excellent pillars - would not qualify")
        else:
            print("  ⚠️ Could not calculate pillar scores")
            
    except Exception as e:
        print(f"  ❌ Scoring error: {e}")
        result["diagnosis"].append(f"Scoring error: {str(e)}")
    
    # -------------------------------------------------------------------------
    # Step 6: Summary & Diagnosis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    if not result["diagnosis"]:
        if any_trigger:
            result["diagnosis"].append("Stock SHOULD be in attention list (check if scan was run)")
        else:
            result["diagnosis"].append("No triggers fired - stock would not be added to attention")
    
    for i, msg in enumerate(result["diagnosis"], 1):
        print(f"  {i}. {msg}")
    
    print("\n" + "=" * 70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why a stock is/isn't in the attention or qualified list"
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbol(s) to diagnose")
    args = parser.parse_args()
    
    for ticker in args.tickers:
        try:
            diagnose_stock(ticker=ticker)
        except Exception as e:
            print(f"\n❌ Error diagnosing {ticker}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
