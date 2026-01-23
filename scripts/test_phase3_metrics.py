"""
Test Phase 3 Core Metrics - QualityGrowthGate, PillarScorer, AttentionTriggers
"""
import sys
sys.path.insert(0, '.')

from src.analysis.stock_screener import StockScreener
from src.analysis.quality_growth_gate import QualityGrowthGate, AttentionTriggers
from src.analysis.pillar_scorer import PillarScorer


def test_phase3_metrics(ticker: str = 'AAPL'):
    """Test new Phase 3 metrics with a real stock"""
    print(f"Testing Phase 3 metrics for {ticker}...")
    
    # Test stock data fetch with new metrics
    print('\n=== Stock Screener Data ===')
    screener = StockScreener()
    data = screener._fetch_stock_data(ticker)
    
    if not data:
        print(f'Failed to fetch data for {ticker}')
        return
    
    print(f"\n{ticker} Data:")
    print(f"  Current Price: ${data.get('current_price', 'N/A')}")
    print(f"  FCF ROIC: {data.get('fcf_roic', 'N/A')}%")
    print(f"  Total Assets: ${data.get('total_assets', 0):,.0f}")
    print(f"  Invested Capital: ${data.get('invested_capital', 0):,.0f}")
    print(f"  SMA 50: {data.get('sma_50', 'N/A')}")
    print(f"  SMA 200: {data.get('sma_200', 'N/A')}")
    print(f"  Price vs 50MA: {data.get('price_vs_50ma', 'N/A')}%")
    print(f"  Price vs 200MA: {data.get('price_vs_200ma', 'N/A')}%")
    print(f"  Revenue Growth: {data.get('revenue_growth', 'N/A')}%")
    print(f"  Next Year Rev Growth: {data.get('next_year_revenue_growth', 'N/A')}%")
    print(f"  Margin of Safety: {data.get('margin_of_safety', 'N/A')}%")
    
    # Test Quality Growth Gate
    print('\n=== Quality Growth Gate ===')
    gate = QualityGrowthGate()
    result = gate.evaluate(
        revenue_growth=data.get('revenue_growth', 0) or 0,
        fcf_roic=data.get('fcf_roic', 0) or 0,
        free_cash_flow=data.get('free_cash_flow', 0) or 0
    )
    print(f"  Passed: {result['passed']}")
    print(f"  Best Path: {result['best_path_name']}")
    print(f"  Paths Matched: {result['paths_matched']}")
    for path_key, details in result['evaluation_details'].items():
        status = "✓" if details['passed'] else "✗"
        print(f"    {status} {details['name']}: Growth {details['growth_actual']:.1f}% >= {details['growth_required']}%, {details['quality_metric']}")
    
    # Test Pillar Scorer
    print('\n=== Pillar Scoring ===')
    scorer = PillarScorer()
    scores = scorer.calculate_composite(data)
    print(f"  Composite Score: {scores['composite_score']}")
    print(f"  Classification: {scores['classification']}")
    print(f"  Qualified (>=60): {scores['qualified']}")
    print("\n  Pillar Breakdown:")
    for pillar_name, pillar_data in scores['pillars'].items():
        print(f"    {pillar_name.upper()}: {pillar_data['score']:.1f} (weight: {pillar_data['weight']*100:.0f}%)")
    
    # Strength/Weakness analysis
    analysis = scorer.get_strength_weakness_analysis(scores)
    print(f"\n  Strengths: {', '.join(analysis['strengths']) or 'None'}")
    print(f"  Weaknesses: {', '.join(analysis['weaknesses']) or 'None'}")
    print(f"  Balanced: {analysis['balanced']} (range: {analysis['score_range']})")
    
    # Test Attention Triggers
    print('\n=== Attention Triggers ===')
    triggers = AttentionTriggers.evaluate_all_triggers(data)
    print(f"  Any Triggered: {triggers['any_triggered']}")
    print(f"  Trigger Count: {triggers['trigger_count']}")
    if triggers['triggers']:
        for t in triggers['triggers']:
            signal = t.get('signal') or t.get('path_name', 'N/A')
            print(f"    - {t['type']}: {signal}")
    else:
        print("    No triggers fired (stock doesn't meet attention criteria)")
    
    print('\n✅ Phase 3 Core Metrics Test Complete!')
    return data, result, scores, triggers


if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    test_phase3_metrics(ticker)
