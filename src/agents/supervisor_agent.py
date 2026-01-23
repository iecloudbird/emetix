"""
Supervisor Agent - Orchestrates multi-agent stock analysis workflow
Part of Multi-Agent Stock Analysis System
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import json
from typing import Dict, List, Optional
from config.logging_config import get_logger
from src.utils.llm_provider import get_llm

# Import specialized agents
from src.agents.data_fetcher_agent import DataFetcherAgent
from src.agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
from src.agents.fundamentals_analyzer_agent import FundamentalsAnalyzerAgent
from src.agents.watchlist_manager_agent import WatchlistManagerAgent
from src.agents.enhanced_valuation_agent import EnhancedValuationAgent

logger = get_logger(__name__)


class SupervisorAgent:
    """
    Orchestrates the multi-agent workflow for Emetix
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Supervisor Agent
        """
        self.logger = logger
        # Use large model tier for complex orchestration with slight temperature
        self.llm = get_llm(model_tier="large", temperature=0.2)
        
        # Initialize specialized agents
        try:
            self.data_fetcher = DataFetcherAgent(api_key)
            self.sentiment_analyzer = SentimentAnalyzerAgent(api_key)
            self.fundamentals_analyzer = FundamentalsAnalyzerAgent(api_key)
            self.watchlist_manager = WatchlistManagerAgent(api_key)
            self.enhanced_valuation = EnhancedValuationAgent(api_key)
            self.logger.info("All specialized agents initialized successfully (including EnhancedValuationAgent)")
        except Exception as e:
            self.logger.warning(f"Some agents failed to initialize: {str(e)}")
            # Continue anyway - tools can still work
        
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup supervisor coordination tools"""
        
        def orchestrate_stock_analysis_tool(ticker: str) -> str:
            """Orchestrate comprehensive stock analysis (all agents)"""
            try:
                results = {}
                
                # Phase 1: Data fetching (parallel ready)
                try:
                    data_result = self.data_fetcher.fetch_complete_dataset(ticker)
                    results['data'] = data_result.get('data', 'No data fetched')
                except Exception as e:
                    results['data'] = f"Data fetch error: {str(e)}"
                
                # Phase 2: Parallel analysis (fundamentals + sentiment)
                try:
                    fundamental_result = self.fundamentals_analyzer.analyze_comprehensive_fundamentals(ticker)
                    results['fundamentals'] = fundamental_result.get('fundamental_analysis', 'No fundamental analysis')
                except Exception as e:
                    results['fundamentals'] = f"Fundamental analysis error: {str(e)}"
                
                try:
                    sentiment_result = self.sentiment_analyzer.analyze_comprehensive_sentiment(ticker)
                    results['sentiment'] = sentiment_result.get('sentiment_analysis', 'No sentiment analysis')
                except Exception as e:
                    results['sentiment'] = f"Sentiment analysis error: {str(e)}"
                
                # Phase 2.5: ML-Powered Valuation (LSTM-DCF + Consensus)
                try:
                    ml_query = f"Provide comprehensive ML-powered valuation for {ticker} using LSTM-DCF and consensus scoring."
                    ml_valuation = self.enhanced_valuation.analyze(ml_query)
                    results['ml_valuation'] = ml_valuation
                except Exception as e:
                    results['ml_valuation'] = f"ML valuation error: {str(e)}"
                
                # Phase 3: Aggregation (contrarian detection if applicable)
                summary = f"""
COMPREHENSIVE ANALYSIS FOR {ticker}

=== FUNDAMENTAL ANALYSIS ===
{results['fundamentals']}

=== MARKET SENTIMENT ===
{results['sentiment']}

=== ML-POWERED VALUATION ===
{results['ml_valuation']}

=== DATA SUMMARY ===
{results['data']}

=== INVESTMENT RECOMMENDATION ===
Based on multi-agent analysis, considering fundamentals, sentiment, and ML predictions.
"""
                return summary
                
            except Exception as e:
                return f"Error orchestrating analysis for {ticker}: {str(e)}"
        
        def build_intelligent_watchlist_tool(tickers_json: str) -> str:
            """
            Build intelligent watchlist with scoring
            
            Args:
                tickers_json: JSON string like ["AAPL", "MSFT", "GOOGL"] or comma-separated like "AAPL, MSFT, GOOGL"
            """
            try:
                # Handle multiple input formats
                if not tickers_json or tickers_json.strip() == '':
                    return "Error: Empty ticker list provided"
                
                # Try to parse as JSON first
                try:
                    tickers = json.loads(tickers_json)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try comma-separated
                    tickers = [t.strip() for t in tickers_json.replace("'", "").replace('"', '').split(',')]
                
                if not tickers:
                    return "Error: No valid tickers found"
                
                self.logger.info(f"Building watchlist for tickers: {tickers}")
                
                # Fetch data for all tickers (could be parallelized)
                stocks_data = []
                for ticker in tickers:
                    try:
                        # Simplified scoring for demo - in production, call all agents
                        import yfinance as yf
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        # Quick heuristic scores (0-1 scale)
                        revenue_growth = info.get('revenueGrowth', 0)
                        growth_score = min(max(revenue_growth * 5, 0), 1)  # Scale to 0-1
                        
                        pe_ratio = info.get('trailingPE', 20)
                        valuation_score = min(max((25 - pe_ratio) / 25, 0), 1)
                        
                        # Sentiment proxy (month momentum)
                        hist = stock.history(period="1mo")
                        if not hist.empty:
                            month_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                            sentiment_score = min(max((month_return + 0.1) * 2.5, 0), 1)
                        else:
                            sentiment_score = 0.5
                        
                        beta = info.get('beta', 1.0)
                        risk_score = min(max((2 - beta) / 2, 0), 1)  # Lower beta = higher score
                        
                        stocks_data.append({
                            'ticker': ticker,
                            'growth_score': round(growth_score, 2),
                            'sentiment_score': round(sentiment_score, 2),
                            'valuation_score': round(valuation_score, 2),
                            'risk_score': round(risk_score, 2),
                            'macro_score': 0.6  # Default
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {ticker}: {str(e)}")
                
                # Build watchlist
                watchlist_result = self.watchlist_manager.build_watchlist(stocks_data)
                
                return str(watchlist_result)
                
            except Exception as e:
                return f"Error building watchlist: {str(e)}"
        
        def find_contrarian_opportunities_tool(tickers_json: str) -> str:
            """
            Find contrarian opportunities (suppressed + undervalued)
            
            Args:
                tickers_json: JSON string like ["OSCR", "PFE", "UPS"] or comma-separated like "OSCR, PFE, UPS"
            """
            try:
                # Handle multiple input formats
                if not tickers_json or tickers_json.strip() == '':
                    return "Error: Empty ticker list provided"
                
                # Try to parse as JSON first
                try:
                    tickers = json.loads(tickers_json)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try comma-separated
                    tickers = [t.strip() for t in tickers_json.replace("'", "").replace('"', '').split(',')]
                
                if not tickers:
                    return "Error: No valid tickers found"
                
                self.logger.info(f"Scanning for contrarian opportunities in: {tickers}")
                
                opportunities = []
                for ticker in tickers:
                    try:
                        # Get sentiment
                        sentiment_result = self.sentiment_analyzer.analyze_comprehensive_sentiment(ticker)
                        
                        # Get fundamentals
                        fundamental_result = self.fundamentals_analyzer.analyze_comprehensive_fundamentals(ticker)
                        
                        # Detect contrarian signal
                        contrarian_result = self.sentiment_analyzer.detect_contrarian_signals(
                            ticker,
                            {'valuation_score': 70}  # Placeholder
                        )
                        
                        opportunities.append({
                            'ticker': ticker,
                            'sentiment': sentiment_result.get('sentiment_analysis', 'N/A'),
                            'fundamentals': fundamental_result.get('fundamental_analysis', 'N/A'),
                            'contrarian_signal': contrarian_result.get('contrarian_analysis', 'N/A')
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing {ticker}: {str(e)}")
                
                summary = f"""
CONTRARIAN OPPORTUNITY SCAN

Found {len(opportunities)} stocks analyzed for contrarian potential.

Contrarian Logic:
- Negative sentiment (<0.4) suppressing fundamentally strong stocks (>0.7 valuation)
- Mean-reversion potential for long-term gains
- Lower risk due to margin of safety in valuation

Opportunities:
"""
                for opp in opportunities:
                    summary += f"\n\n{opp['ticker']}:\n{opp['contrarian_signal']}"
                
                return summary
                
            except Exception as e:
                return f"Error finding contrarian opportunities: {str(e)}"
        
        def compare_peer_stocks_tool(tickers_json: str) -> str:
            """
            Compare peer stocks across fundamentals and sentiment
            
            Args:
                tickers_json: JSON string like '["AAPL", "MSFT", "GOOGL"]'
            """
            try:
                tickers = json.loads(tickers_json)
                
                comparison_result = self.fundamentals_analyzer.compare_peer_fundamentals(tickers)
                
                return str(comparison_result)
                
            except Exception as e:
                return f"Error comparing peers: {str(e)}"
        
        def ml_powered_valuation_tool(ticker: str) -> str:
            """
            Get ML-powered valuation using LSTM-DCF and consensus scoring
            
            Args:
                ticker: Stock ticker symbol
                
            Returns:
                Comprehensive ML valuation with fair value estimates
            """
            try:
                query = f"Provide comprehensive ML-powered valuation for {ticker}. Include LSTM-DCF fair value and consensus valuation. Explain if stock is undervalued or overvalued."
                
                ml_result = self.enhanced_valuation.analyze(query)
                
                return str(ml_result)
                
            except Exception as e:
                return f"Error in ML valuation for {ticker}: {str(e)}"
        
        tools = [
            Tool(
                name="OrchestrateStockAnalysis",
                func=orchestrate_stock_analysis_tool,
                description="Orchestrate comprehensive stock analysis by coordinating Data Fetcher, Fundamentals Analyzer, and Sentiment Analyzer. Returns complete analysis with investment recommendation. Use for single stock deep-dive."
            ),
            Tool(
                name="BuildIntelligentWatchlist",
                func=build_intelligent_watchlist_tool,
                description="Build intelligent watchlist with dynamic scoring for multiple stocks. Input format: comma-separated tickers like 'AAPL, MSFT, GOOGL, TSLA, NVDA'. Calculates weighted composite scores (growth 30%, sentiment 25%, valuation 20%, risk 15%, macro 10%) with contrarian bonuses. Returns ranked watchlist with buy/hold/sell signals."
            ),
            Tool(
                name="FindContrarianOpportunities",
                func=find_contrarian_opportunities_tool,
                description="Find contrarian investment opportunities (negative sentiment + strong fundamentals). Input format: comma-separated tickers like 'OSCR, PFE, UPS'. Detects suppressed stocks with mean-reversion potential for low-risk long-term gains. Use for value investing strategy."
            ),
            Tool(
                name="ComparePeerStocks",
                func=compare_peer_stocks_tool,
                description="Compare multiple stocks in same sector/industry across fundamental metrics. Ranks by valuation, growth, profitability, and financial health. Use for relative value assessment."
            ),
            Tool(
                name="MLPoweredValuation",
                func=ml_powered_valuation_tool,
                description="Get ML-powered stock valuation using LSTM-DCF (deep learning fair value) and consensus scoring. Provides fair value estimates with confidence intervals. Use when user asks for ML valuation, AI analysis, or advanced pricing models."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent using langgraph"""
        try:
            # System prompt for the agent
            system_prompt = """You are a Supervisor Agent that orchestrates multiple specialized agents for comprehensive stock analysis.

You have access to tools that can:
1. Analyze fundamentals (growth, valuation, health, profitability, DCF)
2. Analyze market sentiment (news, social, analyst ratings)
3. Run ML-powered valuation models (LSTM-DCF)
4. Assess investment risk (beta, volatility, debt levels)
5. Screen for macro-economic factors

When analyzing a stock comprehensively:
1. Use FundamentalsAnalysis for deep financial metrics
2. Use SentimentAnalysis for market psychology
3. Use MLValuation for fair value estimates
4. Use RiskAssessment for safety evaluation
5. Use MacroScreening for economic context

Coordinate all agents and synthesize insights into actionable investment recommendations.
Focus on identifying low-risk, long-term opportunities vs speculative plays."""
            
            # Create agent using langgraph prebuilt
            agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=system_prompt
            )
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating Supervisor Agent: {str(e)}")
            raise
    
    def analyze_stock_comprehensive(self, ticker: str) -> Dict:
        """
        Comprehensive stock analysis (all agents)
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete analysis from all agents
        """
        try:
            query = f"Provide comprehensive investment analysis for {ticker}. Coordinate all specialized agents to assess fundamentals, sentiment, valuation (DCF), and risk. Focus on identifying whether this is a low-risk long-term opportunity or a speculative play."
            
            # New langgraph API uses messages format
            result = self.agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            output = result["messages"][-1].content if result.get("messages") else "No response"
            
            return {
                'ticker': ticker,
                'comprehensive_analysis': output,
                'agent': 'SupervisorAgent',
                'model': 'gemini-2.5-flash'
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {'error': str(e)}
    
    def build_watchlist_for_tickers(
        self,
        tickers: List[str],
        custom_weights: Optional[Dict] = None
    ) -> Dict:
        """
        Build intelligent watchlist for list of tickers
        
        Args:
            tickers: List of stock ticker symbols
            custom_weights: Optional custom weights (growth, sentiment, valuation, risk, macro)
            
        Returns:
            Ranked watchlist with scores and signals
        """
        try:
            tickers_json = json.dumps(tickers)
            
            weights_str = ""
            if custom_weights:
                weights_str = f" Use custom weights: {json.dumps(custom_weights)}"
            
            query = f"Build intelligent watchlist for these stocks: {tickers_json}.{weights_str} Calculate composite scores, apply contrarian bonuses, and rank from best to worst. Identify top picks for low-risk long-term growth."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'tickers': tickers,
                'watchlist': result['output'],
                'agent': 'SupervisorAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error building watchlist: {str(e)}")
            return {'error': str(e)}
    
    def scan_for_contrarian_value(self, tickers: List[str]) -> Dict:
        """
        Scan for contrarian value opportunities
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Contrarian opportunities with analysis
        """
        try:
            tickers_json = json.dumps(tickers)
            
            query = f"Scan these stocks for contrarian value opportunities: {tickers_json}. Look for stocks suppressed by negative sentiment but with strong fundamentals (high valuation scores). These are low-risk long-term plays with mean-reversion potential."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'tickers': tickers,
                'contrarian_scan': result['output'],
                'agent': 'SupervisorAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error scanning for contrarian value: {str(e)}")
            return {'error': str(e)}
    
    # Public API methods (for FastAPI routes)
    
    def orchestrate_stock_analysis(self, ticker: str) -> Dict:
        """
        API-compatible method for comprehensive stock analysis.
        Wraps analyze_stock_comprehensive for FastAPI routes.
        """
        return self.analyze_stock_comprehensive(ticker)
    
    def build_intelligent_watchlist(self, tickers: List[str]) -> Dict:
        """
        API-compatible method for building intelligent watchlist.
        Wraps build_watchlist_for_tickers for FastAPI routes.
        """
        return self.build_watchlist_for_tickers(tickers)
    
    def find_contrarian_opportunities(self, tickers: List[str]) -> Dict:
        """
        API-compatible method for finding contrarian opportunities.
        Wraps scan_for_contrarian_value for FastAPI routes.
        """
        return self.scan_for_contrarian_value(tickers)


# Example usage
if __name__ == "__main__":
    print("=== Supervisor Agent - Multi-Agent Orchestration ===\n")
    
    # Initialize supervisor
    supervisor = SupervisorAgent()
    
    # Test 1: Comprehensive single stock analysis
    print("1. Comprehensive Analysis for AAPL...")
    result1 = supervisor.analyze_stock_comprehensive("AAPL")
    print(f"\n{result1['comprehensive_analysis']}\n")
    
    # Test 2: Build watchlist
    print("\n2. Building Watchlist for Tech Stocks...")
    result2 = supervisor.build_watchlist_for_tickers(["AAPL", "MSFT", "GOOGL", "TSLA"])
    print(f"\n{result2['watchlist']}\n")
    
    # Test 3: Contrarian scan
    print("\n3. Scanning for Contrarian Opportunities...")
    result3 = supervisor.scan_for_contrarian_value(["PFE", "OSCR", "UPS"])
    print(f"\n{result3['contrarian_scan']}\n")
