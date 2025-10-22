"""
Interactive Stock Analysis Tool
Tests specific stock tickers using the complete JobHedge Investor system

Usage:
    python scripts/analyze_stock.py AAPL
    python scripts/analyze_stock.py MSFT GOOGL TSLA  # Multiple stocks
    python scripts/analyze_stock.py --help
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis import ValuationAnalyzer, GrowthScreener
from src.agents import ValuationAgent, RiskAgent
from src.data.fetchers.news_sentiment_fetcher import NewsSentimentFetcher
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_logger(__name__)


class StockAnalysisTool:
    """Comprehensive stock analysis using all JobHedge systems"""
    
    def __init__(self):
        """Initialize all analyzers and agents"""
        self.valuation_analyzer = ValuationAnalyzer()
        self.growth_screener = GrowthScreener()
        self.news_fetcher = NewsSentimentFetcher()
        
        # AI agents (require GROQ_API_KEY)
        try:
            self.valuation_agent = ValuationAgent()
            self.risk_agent = RiskAgent()
            self.ai_available = True
        except Exception as e:
            logger.warning(f"AI agents unavailable: {str(e)}")
            self.ai_available = False
    
    def analyze_single_stock(self, ticker: str, detailed: bool = True):
        """
        Comprehensive analysis of a single stock
        
        Args:
            ticker: Stock ticker symbol
            detailed: Include AI agent analysis (requires API key)
        """
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE ANALYSIS: {ticker.upper()}")
        print(f"{'='*80}\n")
        
        # 1. Valuation Analysis
        print(f"{'─'*80}")
        print("1️⃣  VALUATION METRICS")
        print(f"{'─'*80}")
        
        valuation = self.valuation_analyzer.analyze_stock(ticker)
        
        if 'error' not in valuation:
            print(f"\n💰 Current Price: ${valuation['current_price']:.2f}")
            print(f"🎯 Fair Value Est: ${valuation['fair_value_estimate']:.2f}")
            print(f"📈 Valuation Score: {valuation['valuation_score']:.0f}/100")
            print(f"🏷️  Assessment: {valuation['assessment']}")
            print(f"⚠️  Risk Level: {valuation['risk_level']}")
            print(f"💡 Recommendation: {valuation['recommendation']}")
            
            print(f"\n📊 Key Metrics:")
            for metric, value in valuation['key_metrics'].items():
                print(f"   • {metric}: {value:.2f}")
            
            print(f"\n🎨 Component Scores:")
            for component, score in valuation['component_scores'].items():
                print(f"   • {component}: {score:.1f}/100")
        else:
            print(f"❌ Error: {valuation['error']}")
            return
        
        # 2. Growth Opportunity Analysis
        print(f"\n{'─'*80}")
        print("2️⃣  GROWTH OPPORTUNITY ANALYSIS")
        print(f"{'─'*80}")
        
        growth = self.growth_screener.analyze_growth_opportunity(ticker)
        
        if 'error' not in growth:
            print(f"\n🚀 Opportunity Score: {growth['opportunity_score']:.0f}/100")
            print(f"✅ Passed GARP Screening: {'YES' if growth['passed_screening'] else 'NO'}")
            print(f"📈 Growth Momentum: {growth['growth_momentum']}")
            print(f"💎 Valuation Attractiveness: {growth['valuation_attractiveness']}")
            
            print(f"\n📊 Growth Metrics:")
            for metric, value in growth['key_metrics'].items():
                print(f"   • {metric}: {value:.2f}")
            
            if growth['risk_factors']:
                print(f"\n⚠️  Risk Factors:")
                for risk in growth['risk_factors']:
                    print(f"   • {risk}")
            
            print(f"\n💡 Investment Thesis:")
            print(f"   {growth['investment_thesis']}")
        else:
            print(f"⚠️  Growth analysis unavailable")
        
        # 3. News Sentiment Analysis
        print(f"\n{'─'*80}")
        print("3️⃣  NEWS SENTIMENT ANALYSIS")
        print(f"{'─'*80}")
        
        news = self.news_fetcher.fetch_all_news(ticker)
        
        print(f"\n📰 Total Articles: {news['total_articles']}")
        print(f"📋 Unique Articles: {news['unique_articles']} ({news['duplicates_removed']} duplicates removed)")
        print(f"📊 Sentiment Score: {news['sentiment_score']:.2f}/1.0")
        print(f"🎯 Sentiment Label: {news['sentiment_label']}")
        print(f"🎖️  Confidence: {news['confidence']}")
        print(f"🌐 Sources: {', '.join(news['sources_used'])}")
        
        if news['fallback_triggered']:
            print(f"⚠️  Fallback triggered (NewsAPI exhausted, Finnhub used)")
        
        print(f"\n📈 Sentiment Breakdown:")
        if news['unique_articles'] > 0:
            print(f"   • Positive: {news['positive_count']} ({news['positive_count']/news['unique_articles']*100:.1f}%)")
            print(f"   • Negative: {news['negative_count']} ({news['negative_count']/news['unique_articles']*100:.1f}%)")
            print(f"   • Neutral: {news['neutral_count']} ({news['neutral_count']/news['unique_articles']*100:.1f}%)")
        
        if news['articles']:
            print(f"\n📰 Top 3 Recent Headlines:")
            for i, article in enumerate(news['articles'][:3], 1):
                sentiment_emoji = {'POSITIVE': '🟢', 'NEGATIVE': '🔴', 'NEUTRAL': '⚪'}
                print(f"\n   {i}. {sentiment_emoji.get(article['sentiment'], '')} [{article['sentiment']}]")
                print(f"      {article['title'][:70]}")
                print(f"      📍 {article['source']} | {article['publisher']}")
        
        # 4. AI Agent Analysis (if available)
        if detailed and self.ai_available:
            print(f"\n{'─'*80}")
            print("4️⃣  AI-POWERED ANALYSIS")
            print(f"{'─'*80}")
            
            # Valuation Agent
            print(f"\n🤖 AI Valuation Analysis:")
            val_result = self.valuation_agent.analyze_stock(ticker)
            if 'error' not in val_result:
                print(f"\n{val_result['analysis']}")
            else:
                print(f"⚠️  {val_result['error']}")
            
            # Risk Agent
            print(f"\n🤖 AI Risk Assessment:")
            risk_result = self.risk_agent.assess_risk(ticker)
            if 'error' not in risk_result:
                print(f"\n{risk_result['analysis']}")
            else:
                print(f"⚠️  {risk_result['error']}")
        
        # 5. Summary & Recommendation
        print(f"\n{'─'*80}")
        print("5️⃣  FINAL SUMMARY")
        print(f"{'─'*80}")
        
        print(f"\n📊 Overall Assessment:")
        print(f"   • Valuation: {valuation['assessment']}")
        print(f"   • Score: {valuation['valuation_score']:.0f}/100")
        print(f"   • Risk: {valuation['risk_level']}")
        print(f"   • Sentiment: {news['sentiment_label']} ({news['sentiment_score']:.2f})")
        print(f"   • Growth: {growth['growth_momentum'] if 'error' not in growth else 'N/A'}")
        
        print(f"\n💡 Investment Decision:")
        print(f"   {valuation['recommendation']}")
        
        if growth['passed_screening'] if 'error' not in growth else False:
            print(f"\n✨ Special Note: This stock passed GARP screening!")
            print(f"   Consider for growth-at-reasonable-price strategy")
        
        print(f"\n{'='*80}\n")
    
    def compare_stocks(self, tickers: list):
        """
        Compare multiple stocks side-by-side
        
        Args:
            tickers: List of stock ticker symbols
        """
        print(f"\n{'='*80}")
        print(f"📊 STOCK COMPARISON: {', '.join([t.upper() for t in tickers])}")
        print(f"{'='*80}\n")
        
        # Valuation comparison
        print("1️⃣  VALUATION COMPARISON")
        print(f"{'─'*80}\n")
        
        comparison = self.valuation_analyzer.compare_stocks(tickers)
        
        if not comparison.empty:
            print(f"{'Rank':<6}{'Ticker':<8}{'Score':<8}{'Price':<10}{'Fair Value':<12}{'Upside':<10}{'Recommendation':<15}")
            print(f"{'─'*80}")
            
            for idx, row in comparison.iterrows():
                upside = ((row['fair_value'] - row['current_price']) / row['current_price'] * 100)
                print(f"{idx+1:<6}{row['ticker']:<8}{row['valuation_score']:<8.0f}"
                      f"${row['current_price']:<9.2f}${row['fair_value']:<11.2f}"
                      f"{upside:>8.1f}%  {row['recommendation']:<15}")
        else:
            print("❌ No comparison data available")
        
        # Growth screening comparison
        print(f"\n2️⃣  GROWTH SCREENING COMPARISON")
        print(f"{'─'*80}\n")
        
        opportunities = self.growth_screener.find_undervalued_growth_stocks(
            tickers=tickers,
            min_score=0  # Show all
        )
        
        if not opportunities.empty:
            print(f"{'Rank':<6}{'Ticker':<8}{'Score':<8}{'Rev Growth':<12}{'YTD Return':<12}{'PEG':<8}{'Passed':<8}")
            print(f"{'─'*80}")
            
            for idx, row in opportunities.iterrows():
                passed = '✅' if row['passed_screening'] else '❌'
                print(f"{idx+1:<6}{row['ticker']:<8}{row['score']:<8.0f}"
                      f"{row['revenue_growth']:>10.1f}%  {row['ytd_return']:>10.1f}%  "
                      f"{row['peg_ratio']:<8.2f}{passed:<8}")
        else:
            print("❌ No growth screening data available")
        
        # News sentiment comparison
        print(f"\n3️⃣  NEWS SENTIMENT COMPARISON")
        print(f"{'─'*80}\n")
        
        print(f"{'Ticker':<8}{'Articles':<10}{'Sentiment':<12}{'Score':<8}{'Confidence':<12}")
        print(f"{'─'*80}")
        
        for ticker in tickers:
            try:
                news = self.news_fetcher.fetch_all_news(ticker)
                sentiment_emoji = {'BULLISH': '🟢', 'NEUTRAL': '⚪', 'BEARISH': '🔴'}
                emoji = sentiment_emoji.get(news['sentiment_label'], '⚪')
                print(f"{ticker.upper():<8}{news['unique_articles']:<10}"
                      f"{emoji} {news['sentiment_label']:<10}{news['sentiment_score']:<8.2f}"
                      f"{news['confidence']:<12}")
            except Exception as e:
                print(f"{ticker.upper():<8}{'Error':<10}{'N/A':<12}{'N/A':<8}{'N/A':<12}")
        
        print(f"\n{'='*80}\n")
    
    def find_best_opportunities(self, tickers: list):
        """
        Find the best investment opportunities from a list
        
        Args:
            tickers: List of stock ticker symbols to screen
        """
        print(f"\n{'='*80}")
        print(f"🔍 FINDING BEST OPPORTUNITIES FROM {len(tickers)} STOCKS")
        print(f"{'='*80}\n")
        
        # Growth screening
        print("🚀 Growth Opportunities (GARP Strategy):")
        print(f"{'─'*80}\n")
        
        opportunities = self.growth_screener.find_undervalued_growth_stocks(
            tickers=tickers,
            min_score=50.0  # Only show strong opportunities
        )
        
        if not opportunities.empty:
            print(f"Found {len(opportunities)} opportunities passing GARP criteria:\n")
            
            for idx, row in opportunities.head(5).iterrows():
                print(f"{idx+1}. {row['ticker']} - Score: {row['score']:.0f}/100")
                print(f"   • Revenue Growth: {row['revenue_growth']:.1f}%")
                print(f"   • YTD Return: {row['ytd_return']:.1f}%")
                print(f"   • PEG Ratio: {row['peg_ratio']:.2f}")
                print(f"   • Assessment: {row['assessment']}\n")
        else:
            print("❌ No opportunities found passing GARP criteria")
        
        print(f"\n{'='*80}\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Analyze stocks using JobHedge Investor system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single stock
  python scripts/analyze_stock.py AAPL
  
  # Analyze with basic info only (no AI agents)
  python scripts/analyze_stock.py TSLA --basic
  
  # Compare multiple stocks
  python scripts/analyze_stock.py AAPL MSFT GOOGL --compare
  
  # Find best opportunities
  python scripts/analyze_stock.py AAPL MSFT GOOGL TSLA NVDA META --opportunities
        """
    )
    
    parser.add_argument('tickers', nargs='+', help='Stock ticker symbols (e.g., AAPL MSFT)')
    parser.add_argument('--basic', action='store_true', help='Skip AI agent analysis')
    parser.add_argument('--compare', action='store_true', help='Compare multiple stocks')
    parser.add_argument('--opportunities', action='store_true', help='Find best opportunities')
    
    args = parser.parse_args()
    
    # Initialize tool
    tool = StockAnalysisTool()
    
    try:
        if args.opportunities:
            # Find best opportunities
            tool.find_best_opportunities(args.tickers)
        
        elif args.compare:
            # Compare stocks
            tool.compare_stocks(args.tickers)
        
        else:
            # Analyze individual stocks
            for ticker in args.tickers:
                tool.analyze_single_stock(ticker, detailed=not args.basic)
        
        print("✅ Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
