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
import pandas as pd
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

# Import enhanced consensus scorer
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_enhanced_consensus import EnhancedConsensusScorer
    ENHANCED_CONSENSUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced consensus scorer not available: {e}")
    ENHANCED_CONSENSUS_AVAILABLE = False


class StockAnalysisTool:
    """Comprehensive stock analysis using all JobHedge systems"""
    
    def __init__(self):
        """Initialize all analyzers and agents"""
        self.valuation_analyzer = ValuationAnalyzer()
        self.growth_screener = GrowthScreener()
        self.news_fetcher = NewsSentimentFetcher()
        
        # Enhanced consensus scorer (LSTM + RF + P/E)
        if ENHANCED_CONSENSUS_AVAILABLE:
            try:
                self.consensus_scorer = EnhancedConsensusScorer()
                self.consensus_available = True
                logger.info("✅ Enhanced Consensus Scorer initialized")
            except Exception as e:
                logger.warning(f"Consensus scorer initialization failed: {e}")
                self.consensus_available = False
        else:
            self.consensus_available = False
        
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
        print(f"   COMPREHENSIVE ANALYSIS: {ticker.upper()}")
        print(f"{'='*80}\n")
        
        # 1. Valuation Analysis
        print(f"{'='*80}")
        print("[1]  VALUATION METRICS")
        print(f"{'='*80}")
        
        valuation = self.valuation_analyzer.analyze_stock(ticker)
        
        if 'error' not in valuation:
            print(f"\n$  Current Price: ${valuation['current_price']:.2f}")
            print(f"   Fair Value Est: ${valuation['fair_value_estimate']:.2f}")
            print(f"   Valuation Score: {valuation['valuation_score']:.0f}/100")
            print(f"    Assessment: {valuation['assessment']}")
            print(f"[!]  Risk Level: {valuation['risk_level']}")
            print(f"   Recommendation: {valuation['recommendation']}")
            
            print(f"\n   Key Metrics:")
            for metric, value in valuation['key_metrics'].items():
                print(f"   • {metric}: {value:.2f}")
            
            print(f"\n   Component Scores:")
            for component, score in valuation['component_scores'].items():
                print(f"   • {component}: {score:.1f}/100")
        else:
            print(f"[X] Error: {valuation['error']}")
            return
        
        # 2. Growth Opportunity Analysis
        print(f"\n{'='*80}")
        print("[2]  GROWTH OPPORTUNITY ANALYSIS")
        print(f"{'='*80}")
        
        growth = self.growth_screener.analyze_growth_opportunity(ticker)
        
        if 'error' not in growth:
            print(f"\n   Opportunity Score: {growth['opportunity_score']:.0f}/100")
            print(f"[+] Passed GARP Screening: {'YES' if growth['passed_screening'] else 'NO'}")
            print(f"   Growth Momentum: {growth['growth_momentum']}")
            print(f"   Valuation Attractiveness: {growth['valuation_attractiveness']}")
            
            print(f"\n   Growth Metrics:")
            for metric, value in growth['key_metrics'].items():
                print(f"   • {metric}: {value:.2f}")
            
            if growth['risk_factors']:
                print(f"\n[!]  Risk Factors:")
                for risk in growth['risk_factors']:
                    print(f"   • {risk}")
            
            print(f"\n   Investment Thesis:")
            print(f"   {growth['investment_thesis']}")
        else:
            print(f"[!]  Growth analysis unavailable")
        
        # 3. News Sentiment Analysis
        print(f"\n{'='*80}")
        print("[3]  NEWS SENTIMENT ANALYSIS")
        print(f"{'='*80}")
        
        news = self.news_fetcher.fetch_all_news(ticker)
        
        print(f"\n   Total Articles: {news['total_articles']}")
        print(f"   Unique Articles: {news['unique_articles']} ({news['duplicates_removed']} duplicates removed)")
        print(f"   Sentiment Score: {news['sentiment_score']:.2f}/1.0")
        print(f"   Sentiment Label: {news['sentiment_label']}")
        print(f"    Confidence: {news['confidence']}")
        print(f"   Sources: {', '.join(news['sources_used'])}")
        
        if news['fallback_triggered']:
            print(f"[!]  Fallback triggered (NewsAPI exhausted, Finnhub used)")
        
        print(f"\n   Sentiment Breakdown:")
        if news['unique_articles'] > 0:
            print(f"   • Positive: {news['positive_count']} ({news['positive_count']/news['unique_articles']*100:.1f}%)")
            print(f"   • Negative: {news['negative_count']} ({news['negative_count']/news['unique_articles']*100:.1f}%)")
            print(f"   • Neutral: {news['neutral_count']} ({news['neutral_count']/news['unique_articles']*100:.1f}%)")
        
        if news['articles']:
            print(f"\n   Top 3 Recent Headlines:")
            for i, article in enumerate(news['articles'][:3], 1):
                sentiment_emoji = {'POSITIVE': '+', 'NEGATIVE': '-', 'NEUTRAL': '='}
                print(f"\n   {i}. {sentiment_emoji.get(article['sentiment'], '')} [{article['sentiment']}]")
                print(f"      {article['title'][:70]}")
                print(f"         {article['source']} | {article['publisher']}")
        
        # 4. Enhanced ML Consensus Score (70% LSTM + 20% RF + 10% P/E)
        print(f"\n{'='*80}")
        print("[4]  ENHANCED ML CONSENSUS SCORE")  
        print(f"{'='*80}")
        
        if self.consensus_available:
            try:
                # Get comprehensive consensus analysis
                consensus_analysis = self.consensus_scorer.analyze_stock_comprehensive(ticker)
                
                # Display consensus results
                print(f"\n🎯 Consensus Score: {consensus_analysis['consensus_score']:.3f} / 1.000")
                print(f"   Recommendation: {consensus_analysis['recommendation']}")
                print(f"   Confidence: {consensus_analysis['confidence']:.3f}")
                
                # Get individual scores
                individual_scores = consensus_analysis['individual_scores']
                lstm_score = individual_scores.get('lstm_dcf', 0)
                rf_score = individual_scores.get('rf_risk_sentiment', 0)
                pe_score = individual_scores.get('pe_sanity_score', 0)
                
                # Show component breakdown
                print(f"\n   Component Breakdown (70-20-10 Weighting):")
                if lstm_score:
                    print(f"   • LSTM-DCF (70%):     {lstm_score:.3f} → {lstm_score * 0.70:.3f}")
                else:
                    print(f"   • LSTM-DCF (70%):     N/A → 0.000")
                if rf_score:
                    print(f"   • RF Risk+Sent (20%): {rf_score:.3f} → {rf_score * 0.20:.3f}")
                else:
                    print(f"   • RF Risk+Sent (20%): N/A → 0.000")
                if pe_score:
                    print(f"   • P/E Sanity (10%):   {pe_score:.3f} → {pe_score * 0.10:.3f}")
                else:
                    print(f"   • P/E Sanity (10%):   N/A → 0.000")
                print(f"   ─────────────────────────────────────")
                print(f"   = Final Score:        {consensus_analysis['consensus_score']:.3f}")
                
                # Component details
                print(f"\n   🧠 LSTM-DCF Component (Truth Engine - 70%):")
                if lstm_score and lstm_score > 0:
                    print(f"      Score: {lstm_score:.3f} (Contribution: {lstm_score * 0.70:.3f})")
                    print(f"      Method: Fundamental-based DCF analysis")
                else:
                    print(f"      Score: N/A (LSTM model not available)")
                
                print(f"\n   🌲 RF Risk+Sentiment Component (Risk Brake - 20%):")
                if rf_score and rf_score > 0:
                    print(f"      Score: {rf_score:.3f} (Contribution: {rf_score * 0.20:.3f})")
                    print(f"      Features: 14-factor risk & sentiment analysis")
                    print(f"      Method: Enhanced Random Forest (210 samples)")
                else:
                    print(f"      Score: N/A (RF model not available)")
                
                print(f"\n   📊 P/E Sanity Component (Market Anchor - 10%):")
                if pe_score and pe_score > 0:
                    print(f"      Score: {pe_score:.3f} (Contribution: {pe_score * 0.10:.3f})")
                    print(f"      Method: P/E ratio market reality check")
                else:
                    print(f"      Score: N/A (P/E data not available)")
                
                # Investment thesis
                print(f"\n   💡 Consensus Investment Thesis:")
                if lstm_score and rf_score and pe_score:
                    if consensus_analysis['consensus_score'] > 0.6:
                        print(f"      Strong positive consensus across all three models.")
                        print(f"      Fundamental value, risk assessment, and market conditions align positively.")
                    elif consensus_analysis['consensus_score'] > 0.4:
                        print(f"      Mixed signals with balanced risk-reward profile.")
                        print(f"      Hold position while monitoring key metrics.")
                    else:
                        print(f"      Negative consensus indicates significant risks.")
                        print(f"      Consider reducing exposure or avoiding position.")
                else:
                    print(f"      Limited model availability - use caution in decision making.")
                
                # Final recommendation with reasoning
                print(f"\n   🎯 Final Consensus Recommendation:")
                score = consensus_analysis['consensus_score']
                if score >= 0.7:
                    print(f"      [++] STRONG BUY - High consensus confidence ({score:.3f}/1.000)")
                    print(f"          All available models agree on significant upside potential")
                elif score >= 0.6:
                    print(f"      [+]  BUY - Positive consensus ({score:.3f}/1.000)")  
                    print(f"          Majority of signals point to upside")
                elif score >= 0.4:
                    print(f"      [=]  HOLD - Neutral consensus ({score:.3f}/1.000)")
                    print(f"          Mixed signals, fair value range")
                elif score >= 0.3:
                    print(f"      [-]  SELL - Negative consensus ({score:.3f}/1.000)")
                    print(f"          Risk factors outweigh opportunities")
                else:
                    print(f"      [--] STRONG SELL - Poor consensus ({score:.3f}/1.000)")
                    print(f"          Multiple models indicate significant downside")
                    
                print(f"\n   📋 Methodology: 70% fundamental truth + 20% risk sentiment + 10% market reality")
                print(f"      Confidence Level: {consensus_analysis['confidence']:.3f}")
                
                # Model availability status
                model_count = sum([1 for score in [lstm_score, rf_score, pe_score] if score and score > 0])
                print(f"      Models Available: {model_count}/3 (LSTM: {'✓' if lstm_score else '✗'}, RF: {'✓' if rf_score else '✗'}, P/E: {'✓' if pe_score else '✗'})")
                
            except Exception as e:
                print(f"\n[!] Enhanced consensus analysis unavailable: {str(e)}")
                logger.debug(f"Consensus analysis error: {e}", exc_info=True)
        else:
            print(f"\n[!] Enhanced consensus scorer not available")
            print(f"    Missing required models or components")
        
        # 5. AI Agent Analysis (if available)
        if detailed and self.ai_available:
            print(f"\n{'='*80}")
            print("[5]  AI-POWERED ANALYSIS")
            print(f"{'='*80}")
            
            # Valuation Agent
            print(f"\n🤖 AI Valuation Analysis:")
            val_result = self.valuation_agent.analyze_stock(ticker)
            if 'error' not in val_result:
                print(f"\n{val_result['analysis']}")
            else:
                print(f"[!]  {val_result['error']}")
            
            # Risk Agent
            print(f"\n🤖 AI Risk Assessment:")
            risk_result = self.risk_agent.assess_risk(ticker)
            if 'error' not in risk_result:
                print(f"\n{risk_result['analysis']}")
            else:
                print(f"[!]  {risk_result['error']}")
        
        # 6. Summary & Recommendation
        print(f"\n{'='*80}")
        print("[6]  FINAL SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n   Overall Assessment:")
        print(f"   • Valuation: {valuation['assessment']}")
        print(f"   • Score: {valuation['valuation_score']:.0f}/100")
        print(f"   • Risk: {valuation['risk_level']}")
        print(f"   • Sentiment: {news['sentiment_label']} ({news['sentiment_score']:.2f})")
        print(f"   • Growth: {growth['growth_momentum'] if 'error' not in growth else 'N/A'}")
        
        print(f"\n   Investment Decision:")
        print(f"   {valuation['recommendation']}")
        
        if growth['passed_screening'] if 'error' not in growth else False:
            print(f"\n   Special Note: This stock passed GARP screening!")
            print(f"   Consider for growth-at-reasonable-price strategy")
        
        print(f"\n{'='*80}\n")
    
    def compare_stocks(self, tickers: list):
        """
        Compare multiple stocks side-by-side
        
        Args:
            tickers: List of stock ticker symbols
        """
        print(f"\n{'='*80}")
        print(f"   STOCK COMPARISON: {', '.join([t.upper() for t in tickers])}")
        print(f"{'='*80}\n")
        
        # Valuation comparison
        print("[1]  VALUATION COMPARISON")
        print(f"{'='*80}\n")
        
        comparison = self.valuation_analyzer.compare_stocks(tickers)
        
        if not comparison.empty:
            print(f"{'Rank':<6}{'Ticker':<8}{'Score':<8}{'Price':<10}{'Fair Value':<12}{'Upside':<10}{'Recommendation':<15}")
            print(f"{'='*80}")
            
            for idx, row in comparison.iterrows():
                upside = ((row['fair_value'] - row['current_price']) / row['current_price'] * 100)
                print(f"{idx+1:<6}{row['ticker']:<8}{row['valuation_score']:<8.0f}"
                      f"${row['current_price']:<9.2f}${row['fair_value']:<11.2f}"
                      f"{upside:>8.1f}%  {row['recommendation']:<15}")
        else:
            print("[X] No comparison data available")
        
        # Growth screening comparison
        print(f"\n[2]  GROWTH SCREENING COMPARISON")
        print(f"{'='*80}\n")
        
        opportunities = self.growth_screener.find_undervalued_growth_stocks(
            tickers=tickers,
            min_score=0  # Show all
        )
        
        if not opportunities.empty:
            print(f"{'Rank':<6}{'Ticker':<8}{'Score':<8}{'Rev Growth':<12}{'YTD Return':<12}{'PEG':<8}{'Passed':<8}")
            print(f"{'='*80}")
            
            for idx, row in opportunities.iterrows():
                passed = '[+]' if row['passed_screening'] else '[X]'
                print(f"{idx+1:<6}{row['ticker']:<8}{row['score']:<8.0f}"
                      f"{row['revenue_growth']:>10.1f}%  {row['ytd_return']:>10.1f}%  "
                      f"{row['peg_ratio']:<8.2f}{passed:<8}")
        else:
            print("[X] No growth screening data available")
        
        # News sentiment comparison
        print(f"\n[3]  NEWS SENTIMENT COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Ticker':<8}{'Articles':<10}{'Sentiment':<12}{'Score':<8}{'Confidence':<12}")
        print(f"{'='*80}")
        
        for ticker in tickers:
            try:
                news = self.news_fetcher.fetch_all_news(ticker)
                sentiment_emoji = {'BULLISH': '+', 'NEUTRAL': '=', 'BEARISH': '-'}
                emoji = sentiment_emoji.get(news['sentiment_label'], '=')
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
        print(f"   FINDING BEST OPPORTUNITIES FROM {len(tickers)} STOCKS")
        print(f"{'='*80}\n")
        
        # Growth screening
        print("   Growth Opportunities (GARP Strategy):")
        print(f"{'='*80}\n")
        
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
            print("[X] No opportunities found passing GARP criteria")
        
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
        
        print("[+] Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n\n[!]  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\n[X] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
