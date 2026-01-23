"""
Fundamentals Analyzer Agent - Specialized for financial metrics computation
Part of Multi-Agent Stock Analysis System
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config.logging_config import get_logger
from src.models.valuation.fcf_dcf_model import FCFDCFModel
from src.utils.llm_provider import get_llm

logger = get_logger(__name__)


class FundamentalsAnalyzerAgent:
    """
    Specialized agent for computing and interpreting key metrics
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Fundamentals Analyzer Agent
        """
        self.logger = logger
        # Use large model tier for complex financial reasoning
        self.llm = get_llm(model_tier="large", temperature=0)
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
        self.dcf_model = FCFDCFModel()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup fundamental analysis tools"""
        
        def calculate_growth_metrics_tool(ticker: str) -> str:
            """Calculate growth metrics from historical data"""
            try:
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Get growth metrics
                revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
                earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
                
                # Calculate historical growth from financials
                financials = stock.financials
                if not financials.empty and 'Total Revenue' in financials.index:
                    revenues = financials.loc['Total Revenue'].values
                    if len(revenues) >= 2:
                        historical_cagr = ((revenues[0] / revenues[-1]) ** (1/len(revenues)) - 1) * 100
                    else:
                        historical_cagr = 0
                else:
                    historical_cagr = revenue_growth
                
                data = {
                    'ticker': ticker,
                    'revenue_growth_ttm_pct': round(revenue_growth, 2),
                    'earnings_growth_ttm_pct': round(earnings_growth, 2),
                    'revenue_cagr_historical_pct': round(historical_cagr, 2),
                    'growth_quality': 'HIGH' if revenue_growth > 15 and earnings_growth > 10 else 'MEDIUM' if revenue_growth > 5 else 'LOW',
                    'growth_sustainability': 'SUSTAINABLE' if abs(revenue_growth - earnings_growth) < 10 else 'VOLATILE'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating growth metrics for {ticker}: {str(e)}"
        
        def calculate_valuation_ratios_tool(ticker: str) -> str:
            """Calculate comprehensive valuation ratios"""
            try:
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                pe_ratio = info.get('trailingPE', 0)
                forward_pe = info.get('forwardPE', 0)
                pb_ratio = info.get('priceToBook', 0)
                ps_ratio = info.get('priceToSalesTrailing12Months', 0)
                peg_ratio = info.get('pegRatio', 0)
                ev_ebitda = info.get('enterpriseToEbitda', 0)
                ev_revenue = info.get('enterpriseToRevenue', 0)
                
                # Assess valuation
                pe_assessment = 'UNDERVALUED' if 0 < pe_ratio < 15 else 'FAIR' if pe_ratio < 25 else 'OVERVALUED'
                peg_assessment = 'UNDERVALUED' if 0 < peg_ratio < 1 else 'FAIR' if peg_ratio < 1.5 else 'OVERVALUED'
                
                data = {
                    'ticker': ticker,
                    'pe_ratio': round(pe_ratio, 2),
                    'forward_pe': round(forward_pe, 2),
                    'pb_ratio': round(pb_ratio, 2),
                    'ps_ratio': round(ps_ratio, 2),
                    'peg_ratio': round(peg_ratio, 2),
                    'ev_ebitda': round(ev_ebitda, 2),
                    'ev_revenue': round(ev_revenue, 2),
                    'pe_assessment': pe_assessment,
                    'peg_assessment': peg_assessment,
                    'overall_valuation': 'ATTRACTIVE' if peg_assessment == 'UNDERVALUED' else 'EXPENSIVE' if pe_assessment == 'OVERVALUED' else 'FAIR'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating valuation ratios for {ticker}: {str(e)}"
        
        def calculate_financial_health_tool(ticker: str) -> str:
            """Calculate financial health metrics"""
            try:
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                debt_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
                current_ratio = info.get('currentRatio', 0)
                quick_ratio = info.get('quickRatio', 0)
                
                # Calculate interest coverage if available
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                
                interest_coverage = 'N/A'
                if not income_stmt.empty:
                    try:
                        ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else 0
                        interest_expense = income_stmt.loc['Interest Expense'].iloc[0] if 'Interest Expense' in income_stmt.index else 0
                        if interest_expense != 0:
                            interest_coverage = abs(ebit / interest_expense)
                    except:
                        pass
                
                # Health assessment
                debt_health = 'STRONG' if debt_equity < 0.5 else 'GOOD' if debt_equity < 1.0 else 'WEAK'
                liquidity_health = 'STRONG' if current_ratio > 2.0 else 'GOOD' if current_ratio > 1.5 else 'WEAK'
                
                data = {
                    'ticker': ticker,
                    'debt_equity_ratio': round(debt_equity, 2),
                    'current_ratio': round(current_ratio, 2),
                    'quick_ratio': round(quick_ratio, 2),
                    'interest_coverage': round(interest_coverage, 2) if isinstance(interest_coverage, (int, float)) else interest_coverage,
                    'debt_health': debt_health,
                    'liquidity_health': liquidity_health,
                    'overall_health': 'EXCELLENT' if debt_health == 'STRONG' and liquidity_health == 'STRONG' else 'GOOD' if 'WEAK' not in [debt_health, liquidity_health] else 'CONCERNING'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating financial health for {ticker}: {str(e)}"
        
        def calculate_profitability_tool(ticker: str) -> str:
            """Calculate profitability metrics"""
            try:
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                roa = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
                profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
                gross_margin = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
                operating_margin = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
                
                # Profitability assessment
                roe_quality = 'EXCELLENT' if roe > 20 else 'GOOD' if roe > 15 else 'AVERAGE' if roe > 10 else 'POOR'
                margin_quality = 'EXCELLENT' if profit_margin > 20 else 'GOOD' if profit_margin > 10 else 'AVERAGE' if profit_margin > 5 else 'POOR'
                
                data = {
                    'ticker': ticker,
                    'roe_pct': round(roe, 2),
                    'roa_pct': round(roa, 2),
                    'profit_margin_pct': round(profit_margin, 2),
                    'gross_margin_pct': round(gross_margin, 2),
                    'operating_margin_pct': round(operating_margin, 2),
                    'roe_quality': roe_quality,
                    'margin_quality': margin_quality,
                    'overall_profitability': 'EXCELLENT' if roe_quality in ['EXCELLENT', 'GOOD'] and margin_quality in ['EXCELLENT', 'GOOD'] else 'CONCERNING'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating profitability for {ticker}: {str(e)}"
        
        def calculate_dcf_intrinsic_value_tool(ticker: str) -> str:
            """Calculate FCF-based DCF intrinsic value"""
            try:
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                current_fcf = info.get('freeCashflow', 0)
                shares_outstanding = info.get('sharesOutstanding', 0)
                total_debt = info.get('totalDebt', 0)
                total_cash = info.get('totalCash', 0)
                net_debt = total_debt - total_cash
                current_price = info.get('currentPrice', 0)
                
                if current_fcf <= 0 or shares_outstanding == 0:
                    return f"Insufficient data for DCF calculation for {ticker}"
                
                # Estimate growth rates (use revenue growth as proxy)
                revenue_growth = info.get('revenueGrowth', 0.05)
                fcf_growth_rates = [
                    revenue_growth * 0.9,  # Year 1: 90% of revenue growth
                    revenue_growth * 0.8,  # Year 2: 80%
                    revenue_growth * 0.7,  # Year 3: 70%
                    revenue_growth * 0.6,  # Year 4: 60%
                    revenue_growth * 0.5   # Year 5: 50% (conservative)
                ]
                
                # Calculate DCF
                dcf_model = FCFDCFModel()
                result = dcf_model.calculate_with_market_price(
                    current_fcf=current_fcf,
                    fcf_growth_rates=fcf_growth_rates,
                    shares_outstanding=shares_outstanding,
                    current_price=current_price,
                    net_debt=net_debt
                )
                
                if not result:
                    return f"Error in DCF calculation for {ticker}"
                
                data = {
                    'ticker': ticker,
                    'intrinsic_value_per_share': round(result['intrinsic_value_per_share'], 2),
                    'current_market_price': round(current_price, 2),
                    'upside_potential_pct': round(result['upside_potential_pct'], 2),
                    'recommendation': result['recommendation'],
                    'is_undervalued': result['is_undervalued'],
                    'price_to_value_ratio': round(result['price_to_value_ratio'], 2),
                    'confidence_level': result['confidence_level']
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating DCF for {ticker}: {str(e)}"
        
        tools = [
            Tool(
                name="CalculateGrowthMetrics",
                func=calculate_growth_metrics_tool,
                description="Calculate revenue growth, earnings growth, and historical CAGR. Assesses growth quality and sustainability. Critical for growth stock evaluation."
            ),
            Tool(
                name="CalculateValuationRatios",
                func=calculate_valuation_ratios_tool,
                description="Calculate P/E, P/B, PEG, EV/EBITDA and other valuation ratios. Assesses whether stock is undervalued or overvalued relative to metrics."
            ),
            Tool(
                name="CalculateFinancialHealth",
                func=calculate_financial_health_tool,
                description="Calculate debt/equity, current ratio, quick ratio, interest coverage. Assesses balance sheet strength and financial stability."
            ),
            Tool(
                name="CalculateProfitability",
                func=calculate_profitability_tool,
                description="Calculate ROE, ROA, profit margins, gross margins. Assesses company's ability to generate profits from assets and equity."
            ),
            Tool(
                name="CalculateDCFIntrinsicValue",
                func=calculate_dcf_intrinsic_value_tool,
                description="Calculate intrinsic value using Free Cash Flow DCF model. Projects FCF, discounts to present value, compares with market price. Most accurate fair value estimation."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent using langgraph"""
        try:
            # System prompt for the agent
            system_prompt = """You are a Fundamentals Analyzer Agent specialized in deep financial analysis.
            
You have access to tools that can:
1. Analyze growth metrics (revenue growth, earnings growth)
2. Calculate valuation ratios (P/E, P/B, PEG)
3. Assess financial health (debt/equity, current ratio)
4. Evaluate profitability (ROE, margins)
5. Calculate intrinsic value using DCF methodology

When analyzing fundamentals:
1. Calculate growth metrics using GrowthMetrics tool
2. Evaluate valuation ratios using ValuationRatios tool
3. Check financial health using FinancialHealth tool
4. Review profitability using Profitability tool
5. Estimate intrinsic value using DCFValuation tool

Provide data-driven analysis with clear investment implications."""
            
            # Create agent using langgraph prebuilt
            agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=system_prompt
            )
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating Fundamentals Analyzer Agent: {str(e)}")
            raise
    
    def analyze_comprehensive_fundamentals(self, ticker: str) -> Dict:
        """
        Comprehensive fundamental analysis
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete fundamental analysis
        """
        try:
            query = f"Perform comprehensive fundamental analysis for {ticker}. Calculate all key metrics: growth, valuation ratios, financial health, profitability, and DCF intrinsic value. Provide a clear investment recommendation based on fundamentals."
            
            # New langgraph API uses messages format
            result = self.agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            output = result["messages"][-1].content if result.get("messages") else "No response"
            
            return {
                'ticker': ticker,
                'fundamental_analysis': output,
                'agent': 'FundamentalsAnalyzerAgent',
                'model': 'gemini-2.5-flash'
            }
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {str(e)}")
            return {'error': str(e)}
    
    def compare_peer_fundamentals(self, tickers: List[str]) -> Dict:
        """
        Compare fundamental metrics across peer stocks
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            Peer comparison analysis
        """
        try:
            tickers_str = ", ".join(tickers)
            query = f"Compare fundamental metrics across these peer stocks: {tickers_str}. Focus on valuation (PE, PEG), growth rates, profitability (ROE), and financial health. Rank them from best to worst investment based on fundamentals."
            
            # New langgraph API uses messages format
            result = self.agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            output = result["messages"][-1].content if result.get("messages") else "No response"
            
            return {
                'tickers': tickers,
                'peer_comparison': output,
                'agent': 'FundamentalsAnalyzerAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing peer fundamentals: {str(e)}")
            return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    print("=== Fundamentals Analyzer Agent Test ===\n")
    
    # Initialize agent
    agent = FundamentalsAnalyzerAgent()
    
    # Test comprehensive analysis
    print("Analyzing fundamentals for AAPL...")
    result = agent.analyze_comprehensive_fundamentals("AAPL")
    print(f"\nFundamental Analysis:\n{result['fundamental_analysis']}")
    
    # Test peer comparison
    print("\n\nComparing tech peers...")
    peers = agent.compare_peer_fundamentals(["AAPL", "MSFT", "GOOGL"])
    print(f"\nPeer Comparison:\n{peers['peer_comparison']}")
