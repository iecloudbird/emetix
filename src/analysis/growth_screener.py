"""
Growth Stock Screener for Undervalued Opportunities
Identifies stocks with strong revenue growth but lagged price performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from config.logging_config import get_logger
from src.data.fetchers import YFinanceFetcher

logger = get_logger(__name__)


class GrowthScreener:
    """
    Screener for identifying undervalued growth stocks (GARP strategy)
    """
    
    def __init__(self):
        self.logger = logger
        self.fetcher = YFinanceFetcher()
        
        # Default screening criteria
        self.default_criteria = {
            'min_revenue_growth': 15.0,      # Minimum revenue growth %
            'max_ytd_return': 5.0,           # Maximum YTD return %
            'max_peg_ratio': 1.5,            # Maximum PEG ratio
            'min_market_cap': 1_000_000_000, # Minimum market cap ($1B)
            'max_debt_equity': 2.0,          # Maximum debt-to-equity
            'min_roe': 10.0,                 # Minimum ROE %
            'min_gross_margin': 20.0,        # Minimum gross margin %
            'max_pe_ratio': 30.0,            # Maximum P/E ratio
        }
    
    def fetch_growth_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch comprehensive growth and performance data for screening
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with growth metrics and performance data
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical data for performance calculation
            hist_1y = stock.history(period="1y")
            hist_ytd = stock.history(period="ytd")
            
            if hist_1y.empty or hist_ytd.empty:
                return None
            
            # Calculate returns
            current_price = hist_1y['Close'].iloc[-1]
            ytd_start_price = hist_ytd['Close'].iloc[0]
            year_ago_price = hist_1y['Close'].iloc[0]
            
            ytd_return = ((current_price - ytd_start_price) / ytd_start_price) * 100
            one_year_return = ((current_price - year_ago_price) / year_ago_price) * 100
            
            # Get financial metrics
            data = {
                'ticker': ticker,
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                
                # Growth metrics
                'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
                'revenue_growth_qoq': info.get('quarterlyRevenueGrowth', 0) * 100 if info.get('quarterlyRevenueGrowth') else 0,
                'earnings_growth_qoq': info.get('quarterlyEarningsGrowth', 0) * 100 if info.get('quarterlyEarningsGrowth') else 0,
                
                # Performance metrics
                'ytd_return': ytd_return,
                'one_year_return': one_year_return,
                
                # Valuation metrics
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'pb_ratio': info.get('priceToBook', 0),
                
                # Financial health
                'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                'current_ratio': info.get('currentRatio', 0),
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                
                # Profitability
                'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0,
                'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                
                # Cash flow
                'free_cash_flow': info.get('freeCashflow', 0),
                'operating_cash_flow': info.get('operatingCashflow', 0),
                
                # Additional info
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'beta': info.get('beta', 0),
                'analyst_target_price': info.get('targetMeanPrice', 0),
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching growth data for {ticker}: {str(e)}")
            return None
    
    def screen_stock(self, ticker: str, criteria: Optional[Dict] = None) -> Dict:
        """
        Screen a single stock against growth criteria
        
        Args:
            ticker: Stock ticker symbol
            criteria: Custom screening criteria (uses defaults if None)
            
        Returns:
            Screening result with pass/fail and metrics
        """
        if criteria is None:
            criteria = self.default_criteria
        
        data = self.fetch_growth_data(ticker)
        if not data:
            return {'ticker': ticker, 'passed': False, 'error': 'Data fetch failed'}
        
        # Apply screening criteria
        checks = {}
        
        # Growth checks
        checks['revenue_growth'] = data['revenue_growth'] >= criteria['min_revenue_growth']
        checks['market_cap'] = data['market_cap'] >= criteria['min_market_cap']
        
        # Performance checks (lagged price)
        checks['ytd_return'] = data['ytd_return'] <= criteria['max_ytd_return']
        
        # Valuation checks
        if data['peg_ratio'] > 0:
            checks['peg_ratio'] = data['peg_ratio'] <= criteria['max_peg_ratio']
        else:
            checks['peg_ratio'] = True  # Skip if PEG not available
            
        if data['pe_ratio'] > 0:
            checks['pe_ratio'] = data['pe_ratio'] <= criteria['max_pe_ratio']
        else:
            checks['pe_ratio'] = True  # Skip if P/E not available
        
        # Financial health checks
        checks['debt_equity'] = data['debt_equity'] <= criteria['max_debt_equity']
        checks['roe'] = data['roe'] >= criteria['min_roe']
        checks['gross_margin'] = data['gross_margin'] >= criteria['min_gross_margin']
        
        # Cash flow check
        checks['positive_fcf'] = data['free_cash_flow'] > 0
        
        # Overall pass/fail
        passed = all(checks.values())
        
        # Calculate opportunity score (0-100)
        score = self._calculate_opportunity_score(data)
        
        return {
            'ticker': ticker,
            'passed': passed,
            'score': score,
            'data': data,
            'checks': checks,
            'reasons': self._get_fail_reasons(checks) if not passed else []
        }
    
    def _calculate_opportunity_score(self, data: Dict) -> float:
        """
        Calculate opportunity score based on growth and valuation metrics
        """
        score = 50  # Base score
        
        # Revenue growth bonus (0-25 points)
        revenue_growth = data['revenue_growth']
        if revenue_growth > 30:
            score += 25
        elif revenue_growth > 20:
            score += 20
        elif revenue_growth > 15:
            score += 15
        elif revenue_growth > 10:
            score += 10
        
        # Price performance bonus (0-20 points for underperformance)
        ytd_return = data['ytd_return']
        if ytd_return < -20:
            score += 20
        elif ytd_return < -10:
            score += 15
        elif ytd_return < 0:
            score += 10
        elif ytd_return < 5:
            score += 5
        
        # Valuation bonus (0-20 points)
        peg_ratio = data['peg_ratio']
        if peg_ratio > 0:
            if peg_ratio < 0.5:
                score += 20
            elif peg_ratio < 1.0:
                score += 15
            elif peg_ratio < 1.5:
                score += 10
        
        # Profitability bonus (0-15 points)
        roe = data['roe']
        if roe > 20:
            score += 15
        elif roe > 15:
            score += 12
        elif roe > 10:
            score += 8
        
        # Financial health bonus (0-10 points)
        debt_equity = data['debt_equity']
        if debt_equity < 0.5:
            score += 10
        elif debt_equity < 1.0:
            score += 5
        
        # Margin expansion bonus (0-10 points)
        gross_margin = data['gross_margin']
        if gross_margin > 50:
            score += 10
        elif gross_margin > 40:
            score += 7
        elif gross_margin > 30:
            score += 5
        
        return min(100, max(0, score))
    
    def _get_fail_reasons(self, checks: Dict) -> List[str]:
        """Get list of reasons why stock failed screening"""
        reasons = []
        
        if not checks['revenue_growth']:
            reasons.append("Insufficient revenue growth")
        if not checks['ytd_return']:
            reasons.append("Strong YTD performance (not lagged)")
        if not checks['peg_ratio']:
            reasons.append("High PEG ratio (overvalued growth)")
        if not checks['pe_ratio']:
            reasons.append("High P/E ratio")
        if not checks['debt_equity']:
            reasons.append("High debt levels")
        if not checks['roe']:
            reasons.append("Low profitability (ROE)")
        if not checks['gross_margin']:
            reasons.append("Low gross margins")
        if not checks['positive_fcf']:
            reasons.append("Negative free cash flow")
        if not checks['market_cap']:
            reasons.append("Market cap too small")
        
        return reasons
    
    def find_undervalued_growth_stocks(
        self,
        tickers: Optional[List[str]] = None,
        criteria: Optional[Dict] = None,
        min_score: float = 70.0
    ) -> pd.DataFrame:
        """
        Screen multiple stocks for undervalued growth opportunities
        
        Args:
            tickers: List of tickers to screen (uses S&P 500 sample if None)
            criteria: Custom screening criteria
            min_score: Minimum opportunity score
            
        Returns:
            DataFrame with ranked opportunities
        """
        if tickers is None:
            # Default S&P 500 sample for screening
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'CRM', 'ADBE', 'NFLX', 'PYPL', 'SHOP', 'SQ', 'ZOOM',
                'DOCU', 'OKTA', 'SNOW', 'PLTR', 'COIN', 'RBLX',
                'UBER', 'LYFT', 'ABNB', 'DASH', 'PINS', 'SNAP', 'TWTR',
                'ROKU', 'NET', 'DDOG', 'CRWD', 'ZS', 'VEEV', 'WDAY'
            ]
        
        results = []
        
        self.logger.info(f"Screening {len(tickers)} stocks for growth opportunities...")
        
        for ticker in tickers:
            try:
                result = self.screen_stock(ticker, criteria)
                
                if result['passed'] and result['score'] >= min_score:
                    data = result['data']
                    results.append({
                        'ticker': ticker,
                        'score': result['score'],
                        'revenue_growth': data['revenue_growth'],
                        'ytd_return': data['ytd_return'],
                        'one_year_return': data['one_year_return'],
                        'peg_ratio': data['peg_ratio'],
                        'pe_ratio': data['pe_ratio'],
                        'roe': data['roe'],
                        'gross_margin': data['gross_margin'],
                        'debt_equity': data['debt_equity'],
                        'market_cap': data['market_cap'],
                        'current_price': data['current_price'],
                        'sector': data['sector'],
                        'target_price': data['analyst_target_price'],
                        'upside_potential': ((data['analyst_target_price'] - data['current_price']) / data['current_price'] * 100) if data['analyst_target_price'] > 0 else 0
                    })
                    
            except Exception as e:
                self.logger.error(f"Error screening {ticker}: {str(e)}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            # Sort by opportunity score (best first)
            df = df.sort_values('score', ascending=False)
            
            self.logger.info(f"Found {len(df)} undervalued growth opportunities")
            return df
        else:
            self.logger.warning("No stocks passed the growth screening criteria")
            return pd.DataFrame()
    
    def analyze_growth_opportunity(self, ticker: str) -> Dict:
        """
        Detailed analysis of a growth opportunity
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Comprehensive analysis dictionary
        """
        data = self.fetch_growth_data(ticker)
        if not data:
            return {'error': f'Could not fetch data for {ticker}'}
        
        # Screen the stock
        screening_result = self.screen_stock(ticker)
        
        # Calculate additional insights
        growth_momentum = self._assess_growth_momentum(data)
        valuation_attractiveness = self._assess_valuation_attractiveness(data)
        risk_factors = self._identify_risk_factors(data)
        
        return {
            'ticker': ticker,
            'current_price': data['current_price'],
            'opportunity_score': screening_result['score'],
            'passed_screening': screening_result['passed'],
            'growth_momentum': growth_momentum,
            'valuation_attractiveness': valuation_attractiveness,
            'risk_factors': risk_factors,
            'key_metrics': {
                'revenue_growth': data['revenue_growth'],
                'ytd_return': data['ytd_return'],
                'peg_ratio': data['peg_ratio'],
                'roe': data['roe'],
                'gross_margin': data['gross_margin']
            },
            'investment_thesis': self._generate_investment_thesis(data, screening_result),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _assess_growth_momentum(self, data: Dict) -> str:
        """Assess growth momentum based on multiple metrics"""
        revenue_growth = data['revenue_growth']
        earnings_growth = data['earnings_growth']
        
        if revenue_growth > 25 and earnings_growth > 20:
            return "STRONG"
        elif revenue_growth > 15 and earnings_growth > 10:
            return "MODERATE"
        elif revenue_growth > 10:
            return "WEAK"
        else:
            return "DECLINING"
    
    def _assess_valuation_attractiveness(self, data: Dict) -> str:
        """Assess valuation attractiveness"""
        peg_ratio = data['peg_ratio']
        pe_ratio = data['pe_ratio']
        
        if peg_ratio > 0 and peg_ratio < 1.0 and pe_ratio < 25:
            return "VERY ATTRACTIVE"
        elif peg_ratio < 1.5 and pe_ratio < 30:
            return "ATTRACTIVE"
        elif peg_ratio < 2.0:
            return "FAIR"
        else:
            return "EXPENSIVE"
    
    def _identify_risk_factors(self, data: Dict) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if data['debt_equity'] > 1.5:
            risks.append("High debt levels")
        if data['current_ratio'] < 1.0:
            risks.append("Liquidity concerns")
        if data['operating_margin'] < 5:
            risks.append("Low operating margins")
        if data['beta'] > 1.5:
            risks.append("High volatility")
        if data['ytd_return'] < -30:
            risks.append("Significant price decline")
        
        return risks
    
    def _generate_investment_thesis(self, data: Dict, screening_result: Dict) -> str:
        """Generate investment thesis"""
        if not screening_result['passed']:
            return "Does not meet growth screening criteria"
        
        score = screening_result['score']
        revenue_growth = data['revenue_growth']
        ytd_return = data['ytd_return']
        
        thesis = f"Growth opportunity with {revenue_growth:.1f}% revenue growth and {ytd_return:.1f}% YTD return. "
        
        if score >= 80:
            thesis += "Strong buy candidate with excellent growth and attractive valuation."
        elif score >= 70:
            thesis += "Good opportunity with solid fundamentals and reasonable valuation."
        else:
            thesis += "Moderate opportunity requiring further analysis."
        
        return thesis


# Example usage
if __name__ == "__main__":
    screener = GrowthScreener()
    
    # Screen for opportunities
    print("=== Growth Stock Screening ===")
    opportunities = screener.find_undervalued_growth_stocks(min_score=60.0)
    
    if not opportunities.empty:
        print(f"Found {len(opportunities)} opportunities:")
        print(opportunities[['ticker', 'score', 'revenue_growth', 'ytd_return', 'peg_ratio']].head(10).to_string(index=False))
        
        # Detailed analysis of top opportunity
        if len(opportunities) > 0:
            top_pick = opportunities.iloc[0]['ticker']
            print(f"\n=== Detailed Analysis: {top_pick} ===")
            analysis = screener.analyze_growth_opportunity(top_pick)
            print(f"Opportunity Score: {analysis['opportunity_score']}")
            print(f"Growth Momentum: {analysis['growth_momentum']}")
            print(f"Valuation: {analysis['valuation_attractiveness']}")
            print(f"Investment Thesis: {analysis['investment_thesis']}")
    else:
        print("No growth opportunities found with current criteria.")