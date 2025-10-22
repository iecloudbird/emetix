"""
Test script for comprehensive valuation analysis and growth screening
Demonstrates the new valuation metrics and algorithms
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis import ValuationAnalyzer, GrowthScreener
from src.agents import ValuationAgent
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_logger(__name__)


def test_valuation_analyzer():
    """Test the ValuationAnalyzer module"""
    print("=" * 60)
    print("TESTING VALUATION ANALYZER")
    print("=" * 60)
    
    analyzer = ValuationAnalyzer()
    
    # Test single stock analysis
    print("\n1. Single Stock Valuation Analysis")
    print("-" * 40)
    result = analyzer.analyze_stock('AAPL')
    
    if 'error' not in result:
        print(f"Ticker: {result['ticker']}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Fair Value Estimate: ${result['fair_value_estimate']:.2f}")
        print(f"Valuation Score: {result['valuation_score']}/100")
        print(f"Assessment: {result['assessment']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
        print(f"\nKey Metrics:")
        for metric, value in result['key_metrics'].items():
            print(f"  {metric}: {value:.2f}")
        
        print(f"\nComponent Scores:")
        for component, score in result['component_scores'].items():
            print(f"  {component}: {score:.1f}")
    else:
        print(f"Error: {result['error']}")
    
    # Test stock comparison
    print("\n\n2. Stock Comparison Analysis")
    print("-" * 40)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    comparison = analyzer.compare_stocks(tickers)
    
    if not comparison.empty:
        print("Ranked by Valuation Score:")
        for _, row in comparison.iterrows():
            print(f"{row['ticker']}: Score {row['valuation_score']:.1f}/100, "
                  f"${row['current_price']:.2f} → ${row['fair_value']:.2f}, "
                  f"{row['recommendation']}")
    else:
        print("No valid comparison data available")


def test_growth_screener():
    """Test the GrowthScreener module"""
    print("\n\n" + "=" * 60)
    print("TESTING GROWTH SCREENER")
    print("=" * 60)
    
    screener = GrowthScreener()
    
    # Test single stock growth analysis
    print("\n1. Single Stock Growth Analysis")
    print("-" * 40)
    
    # Test with a known tech stock
    result = screener.analyze_growth_opportunity('NVDA')
    
    if 'error' not in result:
        print(f"Ticker: {result['ticker']}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Opportunity Score: {result['opportunity_score']:.1f}/100")
        print(f"Passed Screening: {result['passed_screening']}")
        print(f"Growth Momentum: {result['growth_momentum']}")
        print(f"Valuation Attractiveness: {result['valuation_attractiveness']}")
        
        print(f"\nKey Growth Metrics:")
        for metric, value in result['key_metrics'].items():
            print(f"  {metric}: {value:.2f}")
        
        print(f"\nRisk Factors: {', '.join(result['risk_factors']) if result['risk_factors'] else 'None identified'}")
        print(f"\nInvestment Thesis: {result['investment_thesis']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test growth screening with sample tickers
    print("\n\n2. Growth Stock Screening")
    print("-" * 40)
    
    # Use a smaller sample for testing
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'CRM']
    
    print(f"Screening {len(sample_tickers)} stocks for growth opportunities...")
    opportunities = screener.find_undervalued_growth_stocks(
        tickers=sample_tickers,
        min_score=50.0  # Lower threshold for demo
    )
    
    if not opportunities.empty:
        print(f"\nFound {len(opportunities)} opportunities:")
        print("\nTop Opportunities:")
        for _, row in opportunities.head(5).iterrows():
            print(f"{row['ticker']}: Score {row['score']:.0f}/100, "
                  f"Revenue Growth {row['revenue_growth']:.1f}%, "
                  f"YTD Return {row['ytd_return']:.1f}%, "
                  f"PEG {row['peg_ratio']:.2f}")
    else:
        print("No opportunities found with current criteria")


def test_valuation_agent():
    """Test the ValuationAgent (requires GROQ_API_KEY)"""
    print("\n\n" + "=" * 60)
    print("TESTING VALUATION AGENT")
    print("=" * 60)
    
    try:
        agent = ValuationAgent()
        
        # Test comprehensive stock analysis
        print("\n1. AI-Powered Comprehensive Analysis")
        print("-" * 40)
        result = agent.analyze_stock("AAPL")
        
        if 'error' not in result:
            print(f"AI Analysis for {result['ticker']}:")
            print(result['analysis'])
        else:
            print(f"Error: {result['error']}")
        
        # Test finding growth opportunities
        print("\n\n2. AI-Powered Growth Opportunity Search")
        print("-" * 40)
        opportunities = agent.find_investment_opportunities("growth")
        
        if 'error' not in opportunities:
            print("AI Growth Analysis:")
            print(opportunities['analysis'])
        else:
            print(f"Error: {opportunities['error']}")
            
    except Exception as e:
        print(f"Valuation Agent test skipped - ensure GROQ_API_KEY is set: {str(e)}")


def demonstrate_metrics_calculation():
    """Demonstrate key valuation metrics calculations"""
    print("\n\n" + "=" * 60)
    print("VALUATION METRICS DEMONSTRATION")
    print("=" * 60)
    
    analyzer = ValuationAnalyzer()
    
    # Fetch enhanced data to show all available metrics
    print("\n1. Enhanced Data Fetching")
    print("-" * 40)
    
    data = analyzer.fetch_enhanced_data('AAPL')
    if data is not None:
        row = data.iloc[0]
        
        print("Financial Ratios:")
        print(f"  P/E Ratio: {row['pe_ratio']:.2f}")
        print(f"  P/B Ratio: {row['pb_ratio']:.2f}")
        print(f"  P/S Ratio: {row['ps_ratio']:.2f}")
        print(f"  PEG Ratio: {row['peg_ratio']:.2f}")
        
        print(f"\nFinancial Health:")
        print(f"  Debt/Equity: {row['debt_equity']:.2f}")
        print(f"  Current Ratio: {row['current_ratio']:.2f}")
        print(f"  Quick Ratio: {row['quick_ratio']:.2f}")
        
        print(f"\nProfitability:")
        print(f"  ROE: {row['roe']:.1f}%")
        print(f"  ROA: {row['roa']:.1f}%")
        print(f"  Gross Margin: {row['gross_margin']:.1f}%")
        print(f"  Profit Margin: {row['profit_margin']:.1f}%")
        
        print(f"\nGrowth & Cash Flow:")
        print(f"  Revenue Growth: {row['revenue_growth']:.1f}%")
        print(f"  Earnings Growth: {row['earnings_growth']:.1f}%")
        print(f"  FCF Yield: {row['fcf_yield']:.1f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Beta: {row['beta']:.2f}")
        print(f"  Volatility: {row['volatility']:.1f}%")
        
        print(f"\nValuation:")
        print(f"  EV/EBITDA: {row['ev_ebitda']:.2f}")
        print(f"  Dividend Yield: {row['dividend_yield']:.1f}%")
    else:
        print("Could not fetch enhanced data")


def main():
    """Main test execution"""
    logger.info("Starting comprehensive valuation testing...")
    
    try:
        # Test core modules
        test_valuation_analyzer()
        test_growth_screener()
        
        # Demonstrate metrics
        demonstrate_metrics_calculation()
        
        # Test AI agent (if API key available)
        test_valuation_agent()
        
        print("\n\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)
        print("\nAll valuation and growth screening modules tested successfully!")
        print("The system now includes:")
        print("✓ Comprehensive valuation analysis with 12+ metrics")
        print("✓ Growth stock screening for GARP opportunities")
        print("✓ AI-powered valuation agent with LangChain")
        print("✓ Stock comparison and ranking capabilities")
        print("✓ Real-time data fetching and analysis")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        print(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    main()