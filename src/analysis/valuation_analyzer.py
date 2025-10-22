"""
Comprehensive Stock Valuation Analyzer
Implements key valuation metrics for fair value assessment and undervaluation detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.logging_config import get_logger
from src.data.fetchers import YFinanceFetcher

logger = get_logger(__name__)


class ValuationAnalyzer:
    """
    Advanced valuation analysis using multiple financial metrics
    """
    
    def __init__(self):
        self.logger = logger
        self.fetcher = YFinanceFetcher()
        
        # Valuation thresholds (industry-agnostic defaults)
        self.thresholds = {
            'pe_undervalued': 15,      # P/E < 15 suggests undervaluation
            'pb_undervalued': 1.5,     # P/B < 1.5 for value stocks
            'ps_growth_reasonable': 3,  # P/S < 3 for growth stocks
            'peg_undervalued': 1.0,    # PEG < 1.0 ideal
            'peg_fair': 1.5,           # PEG < 1.5 reasonable
            'debt_equity_safe': 1.0,   # D/E < 1.0 conservative
            'debt_equity_max': 2.0,    # D/E < 2.0 acceptable
            'current_ratio_min': 1.5,  # Current ratio > 1.5 healthy
            'roe_excellent': 15,       # ROE > 15% excellent
            'roe_good': 10,           # ROE > 10% good
            'fcf_yield_high': 10,     # FCF yield > 10% very attractive
            'fcf_yield_good': 5,      # FCF yield > 5% attractive
            'ev_ebitda_undervalued': 10,  # EV/EBITDA < 10 undervalued
            'dividend_yield_high': 4,  # Dividend yield > 4% high
        }
    
    def fetch_enhanced_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch comprehensive stock data including additional valuation metrics
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Enhanced DataFrame with valuation metrics
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                self.logger.warning(f"No historical data for {ticker}")
                return None
            
            # Calculate additional metrics
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            market_cap = info.get('marketCap', 0)
            enterprise_value = info.get('enterpriseValue', 0)
            
            # Calculate volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized %
            
            # Enhanced dataset
            data = {
                'ticker': ticker,
                'current_price': current_price,
                'market_cap': market_cap,
                
                # Price ratios
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'peg_ratio': info.get('pegRatio', 0),
                
                # Financial health
                'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                
                # Profitability
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0,
                
                # Growth metrics
                'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
                
                # Cash flow
                'operating_cash_flow': info.get('operatingCashflow', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'fcf_yield': (info.get('freeCashflow', 0) / market_cap * 100) if market_cap > 0 else 0,
                
                # Valuation metrics
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                'ev_revenue': info.get('enterpriseToRevenue', 0),
                
                # Dividend
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
                
                # Risk metrics
                'beta': info.get('beta', 0),
                'volatility': volatility,
                
                # Market data
                'eps': info.get('trailingEps', 0),
                'book_value': info.get('bookValue', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            df = pd.DataFrame([data])
            self.logger.info(f"Enhanced data fetched for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching enhanced data for {ticker}: {str(e)}")
            return None
    
    def calculate_valuation_score(self, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive valuation score (0-100)
        
        Args:
            data: DataFrame with stock metrics
            
        Returns:
            Dictionary with scores and analysis
        """
        try:
            row = data.iloc[0]
            scores = {}
            total_weight = 0
            weighted_score = 0
            
            # P/E Score (weight: 20%)
            pe_ratio = row['pe_ratio']
            if pe_ratio > 0:
                if pe_ratio < self.thresholds['pe_undervalued']:
                    pe_score = 100
                elif pe_ratio < 25:
                    pe_score = 75
                elif pe_ratio < 35:
                    pe_score = 50
                else:
                    pe_score = 25
                scores['pe_score'] = pe_score
                weighted_score += pe_score * 0.20
                total_weight += 0.20
            
            # P/B Score (weight: 15%)
            pb_ratio = row['pb_ratio']
            if pb_ratio > 0:
                if pb_ratio < 1:
                    pb_score = 100
                elif pb_ratio < self.thresholds['pb_undervalued']:
                    pb_score = 80
                elif pb_ratio < 3:
                    pb_score = 60
                else:
                    pb_score = 30
                scores['pb_score'] = pb_score
                weighted_score += pb_score * 0.15
                total_weight += 0.15
            
            # PEG Score (weight: 20%)
            peg_ratio = row['peg_ratio']
            if peg_ratio > 0:
                if peg_ratio < self.thresholds['peg_undervalued']:
                    peg_score = 100
                elif peg_ratio < self.thresholds['peg_fair']:
                    peg_score = 75
                elif peg_ratio < 2:
                    peg_score = 50
                else:
                    peg_score = 25
                scores['peg_score'] = peg_score
                weighted_score += peg_score * 0.20
                total_weight += 0.20
            
            # Financial Health Score (weight: 15%)
            debt_equity = row['debt_equity']
            current_ratio = row['current_ratio']
            
            health_score = 50  # Base score
            if debt_equity < self.thresholds['debt_equity_safe']:
                health_score += 25
            elif debt_equity < self.thresholds['debt_equity_max']:
                health_score += 10
            else:
                health_score -= 20
                
            if current_ratio > self.thresholds['current_ratio_min']:
                health_score += 25
            elif current_ratio > 1.0:
                health_score += 10
            else:
                health_score -= 15
                
            health_score = max(0, min(100, health_score))
            scores['health_score'] = health_score
            weighted_score += health_score * 0.15
            total_weight += 0.15
            
            # Profitability Score (weight: 15%)
            roe = row['roe']
            if roe > self.thresholds['roe_excellent']:
                prof_score = 100
            elif roe > self.thresholds['roe_good']:
                prof_score = 75
            elif roe > 5:
                prof_score = 50
            elif roe > 0:
                prof_score = 25
            else:
                prof_score = 0
                
            scores['profitability_score'] = prof_score
            weighted_score += prof_score * 0.15
            total_weight += 0.15
            
            # FCF Yield Score (weight: 15%)
            fcf_yield = row['fcf_yield']
            if fcf_yield > self.thresholds['fcf_yield_high']:
                fcf_score = 100
            elif fcf_yield > self.thresholds['fcf_yield_good']:
                fcf_score = 80
            elif fcf_yield > 0:
                fcf_score = 60
            else:
                fcf_score = 30
                
            scores['fcf_score'] = fcf_score
            weighted_score += fcf_score * 0.15
            total_weight += 0.15
            
            # Calculate final score
            final_score = weighted_score / total_weight if total_weight > 0 else 50
            
            return {
                'overall_score': round(final_score, 1),
                'component_scores': scores,
                'total_weight': total_weight,
                'assessment': self._get_valuation_assessment(final_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating valuation score: {str(e)}")
            return {'overall_score': 50, 'error': str(e)}
    
    def _get_valuation_assessment(self, score: float) -> str:
        """Get textual assessment based on score"""
        if score >= 80:
            return "SIGNIFICANTLY UNDERVALUED"
        elif score >= 70:
            return "UNDERVALUED"
        elif score >= 60:
            return "FAIRLY VALUED"
        elif score >= 40:
            return "SLIGHTLY OVERVALUED"
        else:
            return "OVERVALUED"
    
    def analyze_stock(self, ticker: str) -> Dict:
        """
        Comprehensive valuation analysis for a single stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete analysis dictionary
        """
        try:
            # Fetch enhanced data
            data = self.fetch_enhanced_data(ticker)
            if data is None:
                return {'error': f'Could not fetch data for {ticker}'}
            
            row = data.iloc[0]
            
            # Calculate valuation score
            valuation_result = self.calculate_valuation_score(data)
            
            # Key metrics summary
            key_metrics = {
                'pe_ratio': row['pe_ratio'],
                'pb_ratio': row['pb_ratio'],
                'ps_ratio': row['ps_ratio'],
                'peg_ratio': row['peg_ratio'],
                'debt_equity': row['debt_equity'],
                'roe': row['roe'],
                'fcf_yield': row['fcf_yield'],
                'dividend_yield': row['dividend_yield']
            }
            
            # Fair value estimation (simplified)
            fair_value = self._estimate_fair_value(row)
            
            # Risk assessment
            risk_level = self._assess_risk_level(row)
            
            return {
                'ticker': ticker,
                'current_price': row['current_price'],
                'fair_value_estimate': fair_value,
                'valuation_score': valuation_result['overall_score'],
                'assessment': valuation_result['assessment'],
                'key_metrics': key_metrics,
                'risk_level': risk_level,
                'component_scores': valuation_result.get('component_scores', {}),
                'recommendation': self._get_recommendation(valuation_result['overall_score'], risk_level),
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def _estimate_fair_value(self, row: pd.Series) -> float:
        """
        Simple fair value estimation using multiple approaches
        """
        try:
            current_price = row['current_price']
            pe_ratio = row['pe_ratio']
            pb_ratio = row['pb_ratio']
            
            # P/E based fair value (assuming industry average P/E of 18)
            if pe_ratio > 0 and row['eps'] > 0:
                fair_pe = 18
                pe_fair_value = row['eps'] * fair_pe
            else:
                pe_fair_value = current_price
            
            # P/B based fair value (assuming fair P/B of 2.5)
            if pb_ratio > 0 and row['book_value'] > 0:
                fair_pb = 2.5
                pb_fair_value = row['book_value'] * fair_pb
            else:
                pb_fair_value = current_price
            
            # Average fair value
            fair_value = (pe_fair_value + pb_fair_value + current_price) / 3
            return round(fair_value, 2)
            
        except Exception:
            return row['current_price']
    
    def _assess_risk_level(self, row: pd.Series) -> str:
        """Assess risk level based on financial metrics"""
        risk_score = 0
        
        # Beta risk
        beta = row['beta']
        if beta < 0.8:
            risk_score += 0
        elif beta < 1.2:
            risk_score += 1
        else:
            risk_score += 2
        
        # Debt risk
        debt_equity = row['debt_equity']
        if debt_equity < 0.5:
            risk_score += 0
        elif debt_equity < 1.0:
            risk_score += 1
        else:
            risk_score += 2
        
        # Volatility risk
        volatility = row['volatility']
        if volatility < 20:
            risk_score += 0
        elif volatility < 40:
            risk_score += 1
        else:
            risk_score += 2
        
        # Return risk level
        if risk_score <= 1:
            return "LOW"
        elif risk_score <= 3:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_recommendation(self, valuation_score: float, risk_level: str) -> str:
        """Generate investment recommendation"""
        if valuation_score >= 80 and risk_level == "LOW":
            return "STRONG BUY"
        elif valuation_score >= 70 and risk_level in ["LOW", "MEDIUM"]:
            return "BUY"
        elif valuation_score >= 60:
            return "HOLD"
        elif valuation_score >= 40:
            return "WEAK HOLD"
        else:
            return "SELL"
    
    def compare_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """
        Compare valuation metrics across multiple stocks
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for ticker in tickers:
            analysis = self.analyze_stock(ticker)
            if 'error' not in analysis:
                results.append({
                    'ticker': ticker,
                    'current_price': analysis['current_price'],
                    'fair_value': analysis['fair_value_estimate'],
                    'valuation_score': analysis['valuation_score'],
                    'assessment': analysis['assessment'],
                    'risk_level': analysis['risk_level'],
                    'recommendation': analysis['recommendation'],
                    'pe_ratio': analysis['key_metrics']['pe_ratio'],
                    'peg_ratio': analysis['key_metrics']['peg_ratio'],
                    'roe': analysis['key_metrics']['roe']
                })
        
        if results:
            df = pd.DataFrame(results)
            # Sort by valuation score (best opportunities first)
            df = df.sort_values('valuation_score', ascending=False)
            return df
        else:
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    analyzer = ValuationAnalyzer()
    
    # Single stock analysis
    print("=== Single Stock Analysis ===")
    result = analyzer.analyze_stock('AAPL')
    print(f"Ticker: {result['ticker']}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Fair Value: ${result['fair_value_estimate']:.2f}")
    print(f"Valuation Score: {result['valuation_score']}/100")
    print(f"Assessment: {result['assessment']}")
    print(f"Recommendation: {result['recommendation']}")
    
    # Compare multiple stocks
    print("\n=== Stock Comparison ===")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    comparison = analyzer.compare_stocks(tickers)
    if not comparison.empty:
        print(comparison[['ticker', 'valuation_score', 'assessment', 'recommendation']].to_string(index=False))