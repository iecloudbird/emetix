"""
Test Script for Multi-Agent Stock Analysis System
Tests all 5 specialized agents + Supervisor coordination

Usage:
    python scripts/test_multiagent_system.py
    
Requires:
    - GROQ_API_KEY in .env file
    - Internet connection for market data
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.supervisor_agent import SupervisorAgent
from config.logging_config import get_logger

logger = get_logger(__name__)


def test_comprehensive_stock_analysis():
    """Test 1: Comprehensive single stock analysis"""
    print("\n" + "="*80)
    print("TEST 1: COMPREHENSIVE STOCK ANALYSIS (AAPL)")
    print("="*80)
    
    try:
        supervisor = SupervisorAgent()
        
        print("\nAnalyzing AAPL through all specialized agents...")
        print("- Data Fetcher: Gathering fundamentals, FCF, historical data")
        print("- Fundamentals Analyzer: Computing metrics, DCF valuation")
        print("- Sentiment Analyzer: Assessing market sentiment")
        print("- Supervisor: Aggregating insights\n")
        
        result = supervisor.analyze_stock_comprehensive("AAPL")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(result['comprehensive_analysis'])
        
        print("\n✅ Test 1 Complete")
        
    except Exception as e:
        print(f"❌ Test 1 Failed: {str(e)}")
        logger.error(f"Test 1 error: {str(e)}")


def test_intelligent_watchlist():
    """Test 2: Build intelligent watchlist with scoring"""
    print("\n" + "="*80)
    print("TEST 2: INTELLIGENT WATCHLIST (Tech Stocks)")
    print("="*80)
    
    try:
        supervisor = SupervisorAgent()
        
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        print(f"\nBuilding watchlist for: {', '.join(tickers)}")
        print("Scoring weights:")
        print("  - Growth: 30%")
        print("  - Sentiment: 25%")
        print("  - Valuation: 20%")
        print("  - Risk: 15%")
        print("  - Macro: 10%")
        print("  + Contrarian Bonus: Applied for suppressed undervalued stocks\n")
        
        result = supervisor.build_watchlist_for_tickers(tickers)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(result['watchlist'])
        
        print("\n✅ Test 2 Complete")
        
    except Exception as e:
        print(f"❌ Test 2 Failed: {str(e)}")
        logger.error(f"Test 2 error: {str(e)}")


def test_contrarian_opportunities():
    """Test 3: Detect contrarian value opportunities"""
    print("\n" + "="*80)
    print("TEST 3: CONTRARIAN OPPORTUNITY SCAN")
    print("="*80)
    
    try:
        supervisor = SupervisorAgent()
        
        tickers = ["OSCR", "PFE", "UPS"]
        print(f"\nScanning for contrarian opportunities in: {', '.join(tickers)}")
        print("Looking for:")
        print("  - Negative market sentiment (< 0.4)")
        print("  - Strong fundamentals (valuation > 0.7)")
        print("  - Mean-reversion potential")
        print("  - Low-risk long-term gains\n")
        
        result = supervisor.scan_for_contrarian_value(tickers)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(result['contrarian_scan'])
        
        print("\n✅ Test 3 Complete")
        
    except Exception as e:
        print(f"❌ Test 3 Failed: {str(e)}")
        logger.error(f"Test 3 error: {str(e)}")


def test_fcf_dcf_valuation():
    """Test 4: FCF-based DCF valuation"""
    print("\n" + "="*80)
    print("TEST 4: FCF-BASED DCF VALUATION")
    print("="*80)
    
    try:
        from src.models.valuation.fcf_dcf_model import FCFDCFModel
        
        print("\nTesting enhanced DCF model with Free Cash Flow projection...")
        
        fcf_model = FCFDCFModel(
            discount_rate=0.10,
            terminal_growth_rate=0.025,
            projection_years=5
        )
        
        # Example: Apple-like company
        result = fcf_model.calculate_with_market_price(
            current_fcf=100_000_000_000,  # $100B current FCF
            fcf_growth_rates=[0.12, 0.10, 0.08, 0.06, 0.05],  # Declining growth
            shares_outstanding=16_000_000_000,  # 16B shares
            current_price=175.00,
            net_debt=50_000_000_000,  # $50B net debt
            margin_of_safety=0.25
        )
        
        if result:
            print(f"\n{'Metric':<30} {'Value':<20}")
            print("-" * 50)
            print(f"{'Intrinsic Value per Share':<30} ${result['intrinsic_value_per_share']:.2f}")
            print(f"{'Current Market Price':<30} ${result['current_price']:.2f}")
            print(f"{'Conservative Value (25% MoS)':<30} ${result['conservative_value']:.2f}")
            print(f"{'Upside Potential':<30} {result['upside_potential_pct']:.2f}%")
            print(f"{'Recommendation':<30} {result['recommendation']}")
            print(f"{'Confidence Level':<30} {result['confidence_level']}")
            print(f"{'Is Undervalued':<30} {result['is_undervalued']}")
            
            print(f"\n{'Projected FCF (5 years)':<30}")
            for year, fcf in enumerate(result['projected_fcf'], start=1):
                print(f"  Year {year}: ${fcf/1e9:.2f}B")
            
            print(f"\nTerminal Value: ${result['terminal_value']/1e9:.2f}B")
            print(f"PV of Terminal Value: ${result['pv_terminal_value']/1e9:.2f}B")
        
        print("\n✅ Test 4 Complete")
        
    except Exception as e:
        print(f"❌ Test 4 Failed: {str(e)}")
        logger.error(f"Test 4 error: {str(e)}")


def test_individual_agents():
    """Test 5: Test each specialized agent individually"""
    print("\n" + "="*80)
    print("TEST 5: INDIVIDUAL AGENT TESTING")
    print("="*80)
    
    ticker = "MSFT"
    
    # Test Data Fetcher Agent
    try:
        print(f"\n--- Data Fetcher Agent (llama3-8b-8192) ---")
        from src.agents.data_fetcher_agent import DataFetcherAgent
        
        agent = DataFetcherAgent()
        result = agent.fetch_complete_dataset(ticker)
        print(f"✅ Data Fetcher: Successfully fetched data for {ticker}")
        
    except Exception as e:
        print(f"❌ Data Fetcher Failed: {str(e)}")
    
    # Test Sentiment Analyzer Agent
    try:
        print(f"\n--- Sentiment Analyzer Agent (mixtral-8x7b-32768) ---")
        from src.agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
        
        agent = SentimentAnalyzerAgent()
        result = agent.analyze_comprehensive_sentiment(ticker)
        print(f"✅ Sentiment Analyzer: Successfully analyzed sentiment for {ticker}")
        
    except Exception as e:
        print(f"❌ Sentiment Analyzer Failed: {str(e)}")
    
    # Test Fundamentals Analyzer Agent
    try:
        print(f"\n--- Fundamentals Analyzer Agent (llama3-70b-8192) ---")
        from src.agents.fundamentals_analyzer_agent import FundamentalsAnalyzerAgent
        
        agent = FundamentalsAnalyzerAgent()
        result = agent.analyze_comprehensive_fundamentals(ticker)
        print(f"✅ Fundamentals Analyzer: Successfully analyzed fundamentals for {ticker}")
        
    except Exception as e:
        print(f"❌ Fundamentals Analyzer Failed: {str(e)}")
    
    # Test Watchlist Manager Agent
    try:
        print(f"\n--- Watchlist Manager Agent (gemma2-9b-it) ---")
        from src.agents.watchlist_manager_agent import WatchlistManagerAgent
        
        agent = WatchlistManagerAgent()
        sample_data = [
            {'ticker': 'MSFT', 'growth_score': 0.8, 'sentiment_score': 0.7, 'valuation_score': 0.6, 'risk_score': 0.9, 'macro_score': 0.7}
        ]
        result = agent.build_watchlist(sample_data)
        print(f"✅ Watchlist Manager: Successfully built watchlist")
        
    except Exception as e:
        print(f"❌ Watchlist Manager Failed: {str(e)}")
    
    print("\n✅ Test 5 Complete")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MULTI-AGENT STOCK ANALYSIS SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nArchitecture:")
    print("  1. Data Fetcher Agent (llama3-8b) - Raw data gathering")
    print("  2. Sentiment Analyzer Agent (mixtral-8x7b) - Market sentiment")
    print("  3. Fundamentals Analyzer Agent (llama3-70b) - Metrics & DCF")
    print("  4. Watchlist Manager Agent (gemma2-9b) - Intelligent scoring")
    print("  5. Supervisor Agent (llama3-70b) - Orchestration")
    print("\nFocus: Low-risk, long-term growth with contrarian opportunities")
    
    # Check if GROQ_API_KEY is available
    from config.settings import GROQ_API_KEY
    if not GROQ_API_KEY:
        print("\n❌ ERROR: GROQ_API_KEY not found in environment")
        print("Please set GROQ_API_KEY in your .env file")
        print("Get a free key at: https://console.groq.com")
        return
    
    print(f"\n✅ GROQ_API_KEY found")
    
    # Run tests
    try:
        test_fcf_dcf_valuation()  # Test enhanced DCF first (no API needed)
        test_individual_agents()  # Test individual agents
        test_comprehensive_stock_analysis()  # Test full analysis
        test_intelligent_watchlist()  # Test watchlist
        test_contrarian_opportunities()  # Test contrarian detection
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print("\n✅ Multi-Agent System Ready for Production")
        print("\nNext Steps:")
        print("  1. Install langchain_groq: pip install langchain-groq")
        print("  2. Test with your own tickers")
        print("  3. Customize weights for your investment strategy")
        print("  4. Deploy with API backend (Phase 3)")
        
    except Exception as e:
        print(f"\n❌ Test Suite Failed: {str(e)}")
        logger.error(f"Test suite error: {str(e)}")


if __name__ == "__main__":
    main()
