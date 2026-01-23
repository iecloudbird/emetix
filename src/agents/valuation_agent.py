"""
Valuation Agent using LangChain
Integrates comprehensive valuation analysis and growth screening capabilities
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import pandas as pd
from config.logging_config import get_logger
from src.analysis import ValuationAnalyzer, GrowthScreener
from src.utils.llm_provider import get_llm

logger = get_logger(__name__)


class ValuationAgent:
    """
    AI Agent for comprehensive stock valuation analysis using LangChain
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Valuation Agent
        """
        self.logger = logger
        # Use default model tier for valuation analysis
        self.llm = get_llm(model_tier="default", temperature=0)
        self.valuation_analyzer = ValuationAnalyzer()
        self.growth_screener = GrowthScreener()
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _setup_tools(self) -> list:
        """Setup tools for the valuation agent"""
        
        def comprehensive_valuation_tool(ticker: str) -> str:
            """Perform comprehensive valuation analysis for a stock"""
            try:
                result = self.valuation_analyzer.analyze_stock(ticker)
                if 'error' in result:
                    return f"Error analyzing {ticker}: {result['error']}"
                
                analysis = f"""
Comprehensive Valuation Analysis for {ticker}:

Current Price: ${result['current_price']:.2f}
Fair Value Estimate: ${result['fair_value_estimate']:.2f}
Valuation Score: {result['valuation_score']}/100
Assessment: {result['assessment']}
Risk Level: {result['risk_level']}
Recommendation: {result['recommendation']}

Key Metrics:
- P/E Ratio: {result['key_metrics']['pe_ratio']:.2f}
- P/B Ratio: {result['key_metrics']['pb_ratio']:.2f}
- PEG Ratio: {result['key_metrics']['peg_ratio']:.2f}
- Debt/Equity: {result['key_metrics']['debt_equity']:.2f}
- ROE: {result['key_metrics']['roe']:.1f}%
- FCF Yield: {result['key_metrics']['fcf_yield']:.1f}%
- Dividend Yield: {result['key_metrics']['dividend_yield']:.1f}%

Component Scores: {result['component_scores']}
                """
                return analysis.strip()
                
            except Exception as e:
                return f"Error performing valuation analysis: {str(e)}"
        
        def growth_opportunity_analysis_tool(ticker: str) -> str:
            """Analyze growth opportunity potential for a stock"""
            try:
                result = self.growth_screener.analyze_growth_opportunity(ticker)
                if 'error' in result:
                    return f"Error analyzing growth opportunity for {ticker}: {result['error']}"
                
                analysis = f"""
Growth Opportunity Analysis for {ticker}:

Current Price: ${result['current_price']:.2f}
Opportunity Score: {result['opportunity_score']}/100
Passed Screening: {result['passed_screening']}
Growth Momentum: {result['growth_momentum']}
Valuation Attractiveness: {result['valuation_attractiveness']}

Key Growth Metrics:
- Revenue Growth: {result['key_metrics']['revenue_growth']:.1f}%
- YTD Return: {result['key_metrics']['ytd_return']:.1f}%
- PEG Ratio: {result['key_metrics']['peg_ratio']:.2f}
- ROE: {result['key_metrics']['roe']:.1f}%
- Gross Margin: {result['key_metrics']['gross_margin']:.1f}%

Risk Factors: {', '.join(result['risk_factors']) if result['risk_factors'] else 'None identified'}

Investment Thesis: {result['investment_thesis']}
                """
                return analysis.strip()
                
            except Exception as e:
                return f"Error analyzing growth opportunity: {str(e)}"
        
        def stock_comparison_tool(tickers_str: str) -> str:
            """Compare valuation metrics across multiple stocks"""
            try:
                tickers = [t.strip().upper() for t in tickers_str.split(',')]
                comparison = self.valuation_analyzer.compare_stocks(tickers)
                
                if comparison.empty:
                    return "No valid data found for comparison"
                
                # Format comparison results
                result = "Stock Valuation Comparison:\n\n"
                for _, row in comparison.iterrows():
                    result += f"{row['ticker']}: Score {row['valuation_score']}/100, "
                    result += f"Price ${row['current_price']:.2f}, "
                    result += f"Fair Value ${row['fair_value']:.2f}, "
                    result += f"{row['assessment']}, "
                    result += f"Recommendation: {row['recommendation']}\n"
                
                # Rank by score
                best = comparison.iloc[0]
                result += f"\nBest Opportunity: {best['ticker']} with score {best['valuation_score']}/100"
                
                return result.strip()
                
            except Exception as e:
                return f"Error comparing stocks: {str(e)}"
        
        def growth_screener_tool(criteria_str: str) -> str:
            """Screen for undervalued growth stocks with custom criteria"""
            try:
                # Parse criteria if provided (simple format: "min_growth=15,max_ytd=5")
                criteria = {}
                if criteria_str and criteria_str.lower() != "default":
                    for param in criteria_str.split(','):
                        if '=' in param:
                            key, value = param.split('=')
                            key = key.strip()
                            try:
                                value = float(value.strip())
                                if key == "min_growth":
                                    criteria['min_revenue_growth'] = value
                                elif key == "max_ytd":
                                    criteria['max_ytd_return'] = value
                                elif key == "max_peg":
                                    criteria['max_peg_ratio'] = value
                                elif key == "min_roe":
                                    criteria['min_roe'] = value
                            except ValueError:
                                continue
                
                # Run screening
                opportunities = self.growth_screener.find_undervalued_growth_stocks(
                    criteria=criteria if criteria else None,
                    min_score=60.0
                )
                
                if opportunities.empty:
                    return "No undervalued growth opportunities found with current criteria"
                
                # Format results
                result = f"Found {len(opportunities)} Undervalued Growth Opportunities:\n\n"
                
                # Show top 5 opportunities
                top_opportunities = opportunities.head(5)
                for _, row in top_opportunities.iterrows():
                    result += f"{row['ticker']} (Score: {row['score']:.0f}/100):\n"
                    result += f"  Revenue Growth: {row['revenue_growth']:.1f}%\n"
                    result += f"  YTD Return: {row['ytd_return']:.1f}%\n"
                    result += f"  PEG Ratio: {row['peg_ratio']:.2f}\n"
                    result += f"  ROE: {row['roe']:.1f}%\n"
                    result += f"  Sector: {row['sector']}\n\n"
                
                return result.strip()
                
            except Exception as e:
                return f"Error screening for growth stocks: {str(e)}"
        
        tools = [
            Tool(
                name="ComprehensiveValuation",
                func=comprehensive_valuation_tool,
                description="Perform comprehensive valuation analysis including P/E, P/B, PEG ratios, financial health metrics, and generate a valuation score with recommendation. Use this for detailed stock valuation."
            ),
            Tool(
                name="GrowthOpportunityAnalysis",
                func=growth_opportunity_analysis_tool,
                description="Analyze growth opportunity potential for stocks with revenue growth but lagged price performance. Provides growth momentum assessment and investment thesis."
            ),
            Tool(
                name="StockComparison",
                func=stock_comparison_tool,
                description="Compare valuation metrics across multiple stocks. Input should be comma-separated ticker symbols (e.g., 'AAPL,MSFT,GOOGL'). Returns ranked comparison with recommendations."
            ),
            Tool(
                name="GrowthScreener",
                func=growth_screener_tool,
                description="Screen for undervalued growth stocks using customizable criteria. Input format: 'min_growth=15,max_ytd=5,max_peg=1.5' or 'default' for standard criteria. Finds GARP opportunities."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent using langgraph"""
        try:
            # System prompt for the agent
            system_prompt = """You are a Stock Valuation Agent specialized in analyzing stock fair values and growth opportunities.
            
You have access to tools that can:
1. Fetch comprehensive stock fundamentals and metrics
2. Calculate detailed fair value estimates using DCF methodology
3. Screen for growth opportunities using GARP criteria

When analyzing a stock:
1. First fetch the stock data using StockFundamentals tool
2. Calculate the fair value using FairValueCalculator tool
3. Assess growth potential using GrowthScreener tool
4. Provide a comprehensive investment recommendation

Always base your analysis on quantitative data and provide clear reasoning."""
            
            # Create agent using langgraph prebuilt
            agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=system_prompt
            )
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating valuation agent: {str(e)}")
            raise
    
    def analyze_stock(self, ticker: str) -> dict:
        """
        Comprehensive stock analysis including valuation and growth potential
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            query = f"Provide a comprehensive investment analysis for {ticker} including detailed valuation metrics, growth opportunity assessment, and investment recommendation with reasoning."
            
            # New langgraph API uses messages format
            result = self.agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            output = result["messages"][-1].content if result.get("messages") else "No response"
            
            return {
                'ticker': ticker,
                'analysis': output,
                'agent': 'ValuationAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error in stock analysis: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'agent': 'ValuationAgent'
            }
    
    def find_investment_opportunities(self, strategy: str = "growth") -> dict:
        """
        Find investment opportunities based on strategy
        
        Args:
            strategy: Investment strategy ("growth", "value", or "balanced")
        
        Returns:
            Dictionary with opportunities and analysis
        """
        try:
            if strategy.lower() == "growth":
                query = "Find undervalued growth stocks with strong revenue growth but lagged price performance. Use default screening criteria and provide detailed analysis of top opportunities."
            elif strategy.lower() == "value":
                query = "Find deeply undervalued stocks with strong fundamentals. Focus on stocks with low P/E, P/B ratios and high ROE."
            else:  # balanced
                query = "Find balanced investment opportunities combining growth potential with reasonable valuations (GARP strategy). Screen for PEG ratios under 1.5."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'strategy': strategy,
                'analysis': result['output'],
                'agent': 'ValuationAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")
            return {
                'strategy': strategy,
                'error': str(e),
                'agent': 'ValuationAgent'
            }
    
    def compare_investment_options(self, tickers: list) -> dict:
        """
        Compare multiple investment options with detailed analysis
        
        Args:
            tickers: List of stock ticker symbols
        
        Returns:
            Comparative analysis with recommendations
        """
        try:
            tickers_str = ", ".join(tickers)
            query = f"Compare the investment potential of these stocks: {tickers_str}. Provide comprehensive valuation analysis for each, rank them by investment attractiveness, and recommend the best choice with detailed reasoning."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'tickers': tickers,
                'comparison': result['output'],
                'agent': 'ValuationAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing investments: {str(e)}")
            return {
                'tickers': tickers,
                'error': str(e),
                'agent': 'ValuationAgent'
            }


# Example usage
if __name__ == "__main__":
    # Initialize agent (requires GROQ_API_KEY in environment)
    agent = ValuationAgent()
    
    # Single stock analysis
    print("=== Comprehensive Stock Analysis ===")
    result = agent.analyze_stock("AAPL")
    print(result['analysis'])
    
    # Find growth opportunities
    print("\n=== Growth Investment Opportunities ===")
    opportunities = agent.find_investment_opportunities("growth")
    print(opportunities['analysis'])
    
    # Compare stocks
    print("\n=== Investment Comparison ===")
    comparison = agent.compare_investment_options(["AAPL", "MSFT", "GOOGL"])
    print(comparison['comparison'])