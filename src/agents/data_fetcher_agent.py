"""
Data Fetcher Agent - Specialized for gathering raw financial data
Part of Multi-Agent Stock Analysis System
"""
from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain import hub
import os
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from config.settings import GROQ_API_KEY
from config.logging_config import get_logger

logger = get_logger(__name__)


class DataFetcherAgent:
    """
    Specialized agent for fetching raw fundamentals, historical data, and FCF metrics
    Uses Groq Llama3-8B for fast data parsing and validation
    """
    
    def __init__(self, api_key: str = GROQ_API_KEY):
        """
        Initialize Data Fetcher Agent
        
        Args:
            api_key: Groq API key
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        self.logger = logger
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup specialized data fetching tools"""
        
        def fetch_fundamentals_tool(ticker: str) -> str:
            """Fetch comprehensive fundamental data for a stock"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                data = {
                    'ticker': ticker,
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                    'current_ratio': info.get('currentRatio', 0),
                    'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                    'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                    'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
                    'current_price': info.get('currentPrice', 0),
                    'eps': info.get('trailingEps', 0),
                    'beta': info.get('beta', 1.0),
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error fetching fundamentals for {ticker}: {str(e)}"
        
        def fetch_fcf_data_tool(ticker: str) -> str:
            """Fetch Free Cash Flow data for DCF valuation"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                cashflow = stock.cashflow
                
                # Get FCF from cash flow statement
                if not cashflow.empty:
                    try:
                        # Try to get Operating Cash Flow and CapEx
                        operating_cf = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
                        capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
                        fcf = operating_cf + capex  # CapEx is negative
                    except:
                        fcf = info.get('freeCashflow', 0)
                else:
                    fcf = info.get('freeCashflow', 0)
                
                # Get historical FCF if available
                historical_fcf = []
                if not cashflow.empty:
                    try:
                        for col in cashflow.columns[:5]:  # Last 5 years
                            ocf = cashflow.loc['Operating Cash Flow'][col] if 'Operating Cash Flow' in cashflow.index else 0
                            ce = cashflow.loc['Capital Expenditure'][col] if 'Capital Expenditure' in cashflow.index else 0
                            historical_fcf.append(ocf + ce)
                    except:
                        pass
                
                data = {
                    'ticker': ticker,
                    'current_fcf': fcf,
                    'operating_cashflow': info.get('operatingCashflow', 0),
                    'fcf_yield': (fcf / info.get('marketCap', 1)) * 100 if info.get('marketCap', 0) > 0 else 0,
                    'shares_outstanding': info.get('sharesOutstanding', 0),
                    'total_debt': info.get('totalDebt', 0),
                    'total_cash': info.get('totalCash', 0),
                    'net_debt': info.get('totalDebt', 0) - info.get('totalCash', 0),
                    'historical_fcf': historical_fcf if historical_fcf else 'Not available'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error fetching FCF data for {ticker}: {str(e)}"
        
        def fetch_historical_prices_tool(ticker: str) -> str:
            """Fetch historical price data for volatility and trend analysis"""
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if hist.empty:
                    return f"No historical data available for {ticker}"
                
                # Calculate metrics
                current_price = hist['Close'].iloc[-1]
                year_high = hist['Close'].max()
                year_low = hist['Close'].min()
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
                
                # Price momentum
                sma_50 = hist['Close'].tail(50).mean()
                sma_200 = hist['Close'].tail(200).mean() if len(hist) >= 200 else hist['Close'].mean()
                
                data = {
                    'ticker': ticker,
                    'current_price': current_price,
                    '52_week_high': year_high,
                    '52_week_low': year_low,
                    'ytd_return_pct': ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                    'volatility_annual_pct': volatility,
                    'sma_50': sma_50,
                    'sma_200': sma_200,
                    'price_vs_sma50_pct': ((current_price - sma_50) / sma_50) * 100,
                    'trend': 'UPTREND' if sma_50 > sma_200 else 'DOWNTREND'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error fetching historical data for {ticker}: {str(e)}"
        
        def fetch_peer_comparison_tool(ticker: str) -> str:
            """Fetch industry peers for comparison"""
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                data = {
                    'ticker': ticker,
                    'sector': sector,
                    'industry': industry,
                    'sector_pe': info.get('sectorPE', 'N/A'),
                    'industry_pe': info.get('industryPE', 'N/A'),
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error fetching peer data for {ticker}: {str(e)}"
        
        tools = [
            Tool(
                name="FetchFundamentals",
                func=fetch_fundamentals_tool,
                description="Fetch comprehensive fundamental data including ratios, growth rates, and financial health metrics for a stock ticker. Returns structured JSON data."
            ),
            Tool(
                name="FetchFCFData",
                func=fetch_fcf_data_tool,
                description="Fetch Free Cash Flow data for DCF valuation including current FCF, historical FCF, shares outstanding, and net debt. Critical for intrinsic value calculation."
            ),
            Tool(
                name="FetchHistoricalPrices",
                func=fetch_historical_prices_tool,
                description="Fetch historical price data for volatility analysis, trend detection, and momentum indicators (SMA 50/200). Returns 1-year historical metrics."
            ),
            Tool(
                name="FetchPeerComparison",
                func=fetch_peer_comparison_tool,
                description="Fetch industry and sector information for peer comparison analysis. Returns sector/industry classification and average metrics."
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent executor"""
        try:
            # Use standard ReAct prompt
            prompt = hub.pull("hwchase17/react")
            
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10  # Increased from 3 to allow complex multi-source data fetching
            )
            
            return agent_executor
            
        except Exception as e:
            self.logger.error(f"Error creating Data Fetcher Agent: {str(e)}")
            raise
    
    def fetch_complete_dataset(self, ticker: str) -> Dict:
        """
        Fetch complete dataset for a stock (all tools combined)
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete dataset dictionary
        """
        try:
            query = f"Fetch complete financial dataset for {ticker} including fundamentals, FCF data, historical prices, and peer information. Provide structured output for each category."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'ticker': ticker,
                'data': result['output'],
                'agent': 'DataFetcherAgent',
                'model': 'llama3-8b-8192'
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching complete dataset: {str(e)}")
            return {'error': str(e)}
    
    def fetch_watchlist_data(self, tickers: List[str]) -> Dict:
        """
        Fetch data for multiple stocks in watchlist
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Aggregated watchlist data
        """
        try:
            tickers_str = ", ".join(tickers)
            query = f"Fetch fundamental data for these watchlist stocks: {tickers_str}. Focus on key metrics: price, PE ratio, growth rates, FCF, and risk metrics (beta, volatility)."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'tickers': tickers,
                'data': result['output'],
                'agent': 'DataFetcherAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching watchlist data: {str(e)}")
            return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    print("=== Data Fetcher Agent Test ===\n")
    
    # Initialize agent
    agent = DataFetcherAgent()
    
    # Test single stock
    print("Fetching complete dataset for AAPL...")
    result = agent.fetch_complete_dataset("AAPL")
    print(f"\nResult:\n{result['data']}")
    
    # Test watchlist
    print("\n\nFetching watchlist data...")
    watchlist = agent.fetch_watchlist_data(["AAPL", "MSFT", "GOOGL"])
    print(f"\nWatchlist Data:\n{watchlist['data']}")
