"""
Risk Agent using LangChain
Based on Phase 2 draft code
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import pandas as pd
from config.logging_config import get_logger
from src.utils.llm_provider import get_llm

logger = get_logger(__name__)


class RiskAgent:
    """
    AI Agent for stock risk assessment using LangChain
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Risk Agent
        """
        self.logger = logger
        # Use LLM provider (defaults to Gemini for better free tier)
        self.llm = get_llm(model_tier="default", temperature=0)
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _setup_tools(self) -> list:
        """Setup tools for the agent"""
        from src.data.fetchers import YFinanceFetcher
        
        fetcher = YFinanceFetcher()
        
        def valuation_tool(ticker: str) -> str:
            """Fetch stock fundamentals and metrics"""
            try:
                data = fetcher.fetch_stock_data(ticker)
                if data is not None and not data.empty:
                    return data.to_dict('records')[0]
                return f"No data available for {ticker}"
            except Exception as e:
                return f"Error fetching data: {str(e)}"
        
        def risk_scoring_tool(ticker: str) -> str:
            """Calculate risk score for a stock"""
            try:
                data = fetcher.fetch_stock_data(ticker)
                if data is None or data.empty:
                    return "No data available"
                
                beta = data['beta'].iloc[0]
                volatility = data['volatility'].iloc[0]
                debt_equity = data['debt_equity'].iloc[0]
                
                # Simple risk scoring
                risk_score = 0
                if beta < 1.0:
                    risk_score += 1
                if volatility < 30:
                    risk_score += 1
                if debt_equity < 1.0:
                    risk_score += 1
                
                risk_level = "Low" if risk_score >= 2 else "Medium" if risk_score == 1 else "High"
                
                return f"Risk Level: {risk_level} (Score: {risk_score}/3, Beta: {beta:.2f}, Volatility: {volatility:.2f}%, D/E: {debt_equity:.2f})"
            except Exception as e:
                return f"Error calculating risk: {str(e)}"
        
        tools = [
            Tool(
                name="StockValuation",
                func=valuation_tool,
                description="Fetch comprehensive stock fundamentals including P/E ratio, debt/equity, beta, volatility, and market cap. Use this when you need detailed financial metrics for a stock."
            ),
            Tool(
                name="RiskScoring",
                func=risk_scoring_tool,
                description="Calculate a risk score for a stock based on beta, volatility, and debt levels. Use this to assess investment risk."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent using langgraph"""
        try:
            # System prompt for the agent
            system_prompt = """You are a Risk Assessment Agent specialized in analyzing stock investment risks.
            
You have access to tools that can:
1. Fetch stock fundamentals and metrics (P/E ratio, debt/equity, beta, volatility)
2. Calculate risk scores based on key metrics

When assessing a stock:
1. First fetch the stock data using StockValuation tool
2. Then calculate the risk score using RiskScoring tool
3. Provide a comprehensive analysis with your recommendation

Always be thorough and base your analysis on the data retrieved."""
            
            # Create agent using langgraph prebuilt
            agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=system_prompt
            )
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating agent: {str(e)}")
            raise
    
    def assess_risk(self, ticker: str) -> dict:
        """
        Assess risk for a given stock
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with risk assessment
        """
        try:
            query = f"Assess the investment risk for stock ticker {ticker}. Provide a comprehensive analysis including risk level, key metrics, and recommendation."
            
            # New langgraph API uses messages format
            result = self.agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            output = result["messages"][-1].content if result.get("messages") else "No response"
            
            return {
                'ticker': ticker,
                'analysis': output,
                'agent': 'RiskAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'agent': 'RiskAgent'
            }
    
    def assess_risk_with_context(self, ticker: str, context_query: str) -> dict:
        """
        Assess risk with comprehensive analysis context
        
        Args:
            ticker: Stock ticker symbol
            context_query: Detailed query with all analysis data
        
        Returns:
            Dictionary with contextual risk assessment
        """
        try:
            result = self.agent_executor.invoke({"input": context_query})
            
            return {
                'ticker': ticker,
                'analysis': result['output'],
                'agent': 'RiskAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error in contextual risk assessment: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'agent': 'RiskAgent'
            }
    
    def compare_stocks(self, tickers: list) -> dict:
        """
        Compare risk across multiple stocks
        
        Args:
            tickers: List of stock tickers
        
        Returns:
            Comparison analysis
        """
        try:
            tickers_str = ", ".join(tickers)
            query = f"Compare the investment risk across these stocks: {tickers_str}. Rank them from lowest to highest risk and explain your reasoning."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'tickers': tickers,
                'comparison': result['output'],
                'agent': 'RiskAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing stocks: {str(e)}")
            return {
                'tickers': tickers,
                'error': str(e),
                'agent': 'RiskAgent'
            }


# Example usage
if __name__ == "__main__":
    # Initialize agent (requires GROQ_API_KEY in environment)
    agent = RiskAgent()
    
    # Assess single stock
    print("=== Risk Assessment for AAPL ===")
    result = agent.assess_risk("AAPL")
    print(result['analysis'])
    
    # Compare stocks
    print("\n=== Comparing Multiple Stocks ===")
    comparison = agent.compare_stocks(["AAPL", "MSFT", "TSLA"])
    print(comparison['comparison'])
