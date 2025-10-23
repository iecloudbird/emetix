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
        
        # 4. ML-Powered Valuation (LSTM-DCF + RF Ensemble)
        print(f"\n{'─'*80}")
        print("4️⃣  ML-POWERED VALUATION")
        print(f"{'─'*80}")
        
        try:
            from src.models.deep_learning.lstm_dcf import LSTMDCFModel
            from src.models.ensemble.rf_ensemble import RFEnsembleModel
            from src.data.processors.time_series_processor import TimeSeriesProcessor
            from src.data.fetchers import YFinanceFetcher
            from config.settings import MODELS_DIR
            import torch
            import yfinance as yf
            
            fetcher = YFinanceFetcher()
            ts_processor = TimeSeriesProcessor()
            
            # Load models
            lstm_model = None
            rf_model = None
            
            lstm_path = MODELS_DIR / "lstm_dcf_final.pth"
            if lstm_path.exists():
                try:
                    lstm_model = LSTMDCFModel(input_size=12, hidden_size=128, num_layers=3)
                    lstm_model.load_model(str(lstm_path))
                    lstm_model.eval()
                except Exception as e:
                    logger.debug(f"Could not load LSTM model: {e}")
            
            rf_path = MODELS_DIR / "rf_ensemble.pkl"
            if rf_path.exists():
                try:
                    rf_model = RFEnsembleModel()
                    rf_model.load(str(rf_path))
                except Exception as e:
                    logger.debug(f"Could not load RF model: {e}")
            
            # LSTM-DCF Valuation
            if lstm_model:
                print(f"\n🧠 LSTM Price Forecast Model (Deep Learning):")
                print(f"{'─'*80}")
                
                try:
                    # Fetch time-series data
                    ts_data = ts_processor.fetch_sequential_data(ticker, period='5y')
                    
                    if ts_data is not None and not ts_data.empty:
                        X, _ = ts_processor.create_sequences(ts_data, target_col='close')
                        
                        if len(X) > 0:
                            last_seq = torch.tensor(X[-1:], dtype=torch.float32)
                            
                            # Get current stock info
                            stock_data = fetcher.fetch_stock_data(ticker)
                            
                            if stock_data is not None:
                                # Extract current price and fundamentals
                                if isinstance(stock_data, dict):
                                    current_price = stock_data.get('current_price', 0)
                                    fcf = stock_data.get('free_cash_flow', 0)
                                    shares = stock_data.get('shares_outstanding', 1e9)
                                    revenue = stock_data.get('revenue', 0)
                                else:
                                    import yfinance as yf
                                    stock = yf.Ticker(ticker)
                                    info = stock.info
                                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                                    fcf = info.get('freeCashflow', 0)
                                    shares = info.get('sharesOutstanding', 1e9)
                                    revenue = info.get('totalRevenue', 0)
                                
                                # Forecast future prices using LSTM
                                lstm_model.eval()
                                with torch.no_grad():
                                    # Get prediction for next period
                                    price_pred = lstm_model.forward(last_seq).item()
                                    
                                    # Scale prediction back to price range
                                    # The model outputs normalized values, so we scale relative to current price
                                    # Using historical volatility as a guide
                                    hist_std = ts_data['close'].std()
                                    hist_mean = ts_data['close'].mean()
                                    
                                    # Denormalize: pred is likely in normalized space
                                    predicted_price = price_pred * hist_std + hist_mean
                                
                                # Simple DCF using actual FCF if available
                                if fcf > 0 and shares > 0:
                                    # Calculate intrinsic value using FCF
                                    wacc = 0.08  # 8% discount rate
                                    terminal_growth = 0.03  # 3% terminal growth
                                    
                                    # 10-year DCF
                                    fcf_per_share = fcf / shares
                                    
                                    # Sum of discounted FCF (assuming modest growth)
                                    pv_fcf = 0
                                    fcf_growth = 0.05  # Assume 5% annual FCF growth
                                    for year in range(1, 11):
                                        future_fcf = fcf_per_share * ((1 + fcf_growth) ** year)
                                        pv = future_fcf / ((1 + wacc) ** year)
                                        pv_fcf += pv
                                    
                                    # Terminal value
                                    terminal_fcf = fcf_per_share * ((1 + fcf_growth) ** 10) * (1 + terminal_growth)
                                    terminal_value = terminal_fcf / (wacc - terminal_growth)
                                    pv_terminal = terminal_value / ((1 + wacc) ** 10)
                                    
                                    # Fair value = PV of FCF + PV of Terminal Value
                                    fair_value_dcf = pv_fcf + pv_terminal
                                    
                                    # SMART BLENDING: Adjust weights based on FCF reliability
                                    # High FCF per share (>$5): Trust DCF more (70/30)
                                    # Low FCF per share (<$5): Trust LSTM more (30/70)
                                    # Very low FCF (<$1): Use LSTM only (0/100)
                                    
                                    if fcf_per_share >= 5.0:
                                        # Mature company with strong FCF - trust DCF
                                        dcf_weight = 0.70
                                        lstm_weight = 0.30
                                        valuation_type = "DCF-Dominant (Mature)"
                                    elif fcf_per_share >= 1.0:
                                        # Moderate FCF - balanced approach
                                        dcf_weight = 0.50
                                        lstm_weight = 0.50
                                        valuation_type = "Balanced (Hybrid)"
                                    else:
                                        # Low FCF - growth stock, trust price momentum
                                        dcf_weight = 0.20
                                        lstm_weight = 0.80
                                        valuation_type = "LSTM-Dominant (Growth)"
                                    
                                    fair_value = dcf_weight * fair_value_dcf + lstm_weight * predicted_price
                                else:
                                    # Fallback: Use LSTM price prediction only
                                    fair_value = predicted_price
                                    dcf_weight = 0.0
                                    lstm_weight = 1.0
                                    valuation_type = "LSTM-Only (No FCF)"
                                
                                # Calculate metrics
                                gap = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
                                safety_margin = abs(gap) if gap > 0 else 0
                                
                                # Display results
                                print(f"\n💰 Valuation Results:")
                                print(f"   Current Market Price: ${current_price:,.2f}")
                                print(f"   ML Fair Value:        ${fair_value:,.2f}")
                                
                                if fcf > 0:
                                    print(f"     • DCF Component ({dcf_weight*100:.0f}%): ${fair_value_dcf:,.2f}")
                                    print(f"     • LSTM Forecast ({lstm_weight*100:.0f}%): ${predicted_price:,.2f}")
                                    print(f"     • Valuation Type: {valuation_type}")
                                    if fcf_per_share < 1.0:
                                        print(f"     ⚠️  Low FCF/share (${fcf_per_share:.2f}) - LSTM weighted higher")
                                else:
                                    print(f"     • LSTM Price Forecast: ${predicted_price:,.2f}")
                                    print(f"     • (No FCF data - using price-based valuation)")
                                
                                print(f"   Valuation Gap:        {gap:+.2f}%")
                                
                                if gap > 0:
                                    print(f"   🛡️  Safety Margin:      {safety_margin:.2f}% (UNDERVALUED)")
                                    print(f"   💎 Potential Upside:   ${(fair_value - current_price):,.2f} per share")
                                elif gap < -10:
                                    print(f"   ⚠️  Overvaluation:      {abs(gap):.2f}% (OVERVALUED)")
                                    print(f"   📉 Potential Downside: ${(current_price - fair_value):,.2f} per share")
                                else:
                                    print(f"   ✅ Fair Value Range:   Within ±10% (FAIRLY VALUED)")
                                
                                if fcf > 0:
                                    print(f"\n📊 DCF Components:")
                                    print(f"   Free Cash Flow:       ${fcf/1e9:,.2f}B")
                                    print(f"   FCF per Share:        ${fcf/shares:,.2f}")
                                    print(f"   PV of 10Y FCF:        ${pv_fcf:,.2f}")
                                    print(f"   Terminal Value (PV):  ${pv_terminal:,.2f}")
                                    print(f"   Assumed FCF Growth:   5.0% annually")
                                
                                print(f"\n📈 LSTM Price Forecast:")
                                print(f"   Historical Mean:      ${hist_mean:,.2f}")
                                print(f"   Historical Std Dev:   ${hist_std:,.2f}")
                                print(f"   Predicted Next Price: ${predicted_price:,.2f}")
                                
                                # Investment recommendation
                                print(f"\n💡 ML Hybrid Recommendation:")
                                if gap > 15:
                                    print(f"   ✅ STRONG BUY - Significant undervaluation with {safety_margin:.1f}% safety margin")
                                elif gap > 5:
                                    print(f"   ✅ BUY - Moderate undervaluation detected")
                                elif gap > -5:
                                    print(f"   ⚖️  HOLD - Fairly valued")
                                elif gap > -15:
                                    print(f"   ⚠️  WEAK SELL - Slightly overvalued")
                                else:
                                    print(f"   🔴 SELL - Significantly overvalued")
                                
                                confidence = "High" if (fcf > 0 and abs(gap) > 20) else "Medium" if abs(gap) > 10 else "Low"
                                print(f"   Confidence Level: {confidence}")
                                print(f"   Method: {'Hybrid (DCF + LSTM)' if fcf > 0 else 'LSTM Price Forecast'}")
                            else:
                                print(f"\n⚠️  Could not fetch stock fundamentals")
                        else:
                            print(f"\n⚠️  Insufficient historical data (need 60+ periods)")
                    else:
                        print(f"\n⚠️  Could not fetch time-series data")
                        
                except Exception as e:
                    print(f"\n⚠️  LSTM valuation error: {str(e)}")
                    logger.debug(f"LSTM error: {e}", exc_info=True)
            else:
                print(f"\n⚠️  LSTM model not loaded")
            
            # RF Ensemble Analysis
            if rf_model:
                print(f"\n\n🌲 Random Forest Ensemble (Multi-Metric Analysis):")
                print(f"{'─'*80}")
                
                try:
                    stock_data = fetcher.fetch_stock_data(ticker)
                    
                    if stock_data is not None and not (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
                        # Prepare features
                        X = rf_model.prepare_features(stock_data)
                        
                        # Get predictions
                        result = rf_model.predict_score(X)
                        
                        print(f"\n📊 Ensemble Predictions:")
                        print(f"   Composite Score:      {result['ensemble_score']:.4f} (0-1 scale)")
                        print(f"   Regression Score:     {result['regression_score']:.4f}")
                        print(f"   Classification Prob:  {result['classification_prob']:.2%}")
                        print(f"   Undervalued:          {'YES ✅' if result['is_undervalued'] else 'NO ❌'}")
                        
                        # Feature importance
                        importance = rf_model.get_feature_importance()
                        if not importance.empty:
                            print(f"\n🔍 Top 5 Value Drivers (Feature Importance):")
                            for idx, row in importance.head(5).iterrows():
                                bar_length = int(row['importance'] * 30)
                                bar = '█' * bar_length + '░' * (30 - bar_length)
                                print(f"   {row['feature']:<20} {bar} {row['importance']:.1%}")
                        
                        # ML-based recommendation
                        print(f"\n💡 RF Ensemble Recommendation:")
                        if result['is_undervalued'] and result['classification_prob'] > 0.7:
                            print(f"   ✅ BUY - High probability ({result['classification_prob']:.1%}) of undervaluation")
                        elif result['ensemble_score'] > 0.6:
                            print(f"   ⚖️  HOLD - Positive indicators but not clearly undervalued")
                        elif result['ensemble_score'] < 0.4:
                            print(f"   🔴 AVOID - Weak fundamentals detected")
                        else:
                            print(f"   ⚠️  NEUTRAL - Mixed signals")
                    else:
                        print(f"\n⚠️  Could not fetch stock data")
                        
                except Exception as e:
                    print(f"\n⚠️  RF Ensemble error: {str(e)}")
                    logger.debug(f"RF Ensemble error: {e}", exc_info=True)
            else:
                print(f"\n⚠️  RF Ensemble model not loaded")
            
            # Consensus view
            if lstm_model and rf_model:
                print(f"\n\n🎯 ML Consensus View:")
                print(f"{'─'*80}")
                print(f"   Both models loaded - comparing signals...")
                print(f"   Use LSTM-DCF for intrinsic value & safety margin")
                print(f"   Use RF Ensemble for relative value & feature analysis")
                print(f"   Best when both models agree on direction")
            
        except Exception as e:
            print(f"\n⚠️  ML analysis unavailable: {str(e)}")
            logger.debug(f"ML analysis error: {e}", exc_info=True)
        
        # 5. AI Agent Analysis (if available)
        if detailed and self.ai_available:
            print(f"\n{'─'*80}")
            print("5️⃣  AI-POWERED ANALYSIS")
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
        
        # 6. Summary & Recommendation
        print(f"\n{'─'*80}")
        print("6️⃣  FINAL SUMMARY")
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
