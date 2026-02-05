"""
Comprehensive Stock Valuation Analyzer
Implements key valuation metrics for fair value assessment and undervaluation detection
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from config.logging_config import get_logger
from config.settings import MODELS_DIR
from src.data.fetchers import YFinanceFetcher
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.data.processors.time_series_processor import TimeSeriesProcessor

logger = get_logger(__name__)


class ValuationAnalyzer:
    """
    Advanced valuation analysis using multiple financial metrics
    """
    
    def __init__(self):
        self.logger = logger
        self.fetcher = YFinanceFetcher()
        self.ts_processor = TimeSeriesProcessor()
        
        # Initialize LSTM-DCF model (lazy loading)
        self.lstm_model = None
        self._load_lstm_model()
        
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
    
    def _load_lstm_model(self):
        """Load LSTM-DCF growth model for fair value prediction using from_checkpoint"""
        try:
            model_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
            if model_path.exists():
                # Use from_checkpoint for automatic v1/v2 detection
                self.lstm_model, metadata = LSTMDCFModel.from_checkpoint(str(model_path))
                
                # Extract metadata - handle both v1 and v2 scaler formats
                self.lstm_scaler = metadata.get('feature_scaler') or metadata.get('scaler')
                self.lstm_target_scaler = metadata.get('target_scaler')
                self.lstm_feature_cols = metadata.get('feature_cols', [])
                self.lstm_sequence_length = metadata.get('sequence_length', 8)
                self.lstm_model_version = metadata.get('model_version', 'v1')
                
                self.logger.info(f"✅ LSTM-DCF {self.lstm_model_version} model loaded (seq_len={self.lstm_sequence_length})")
                return
            
            self.logger.warning("⚠️ No LSTM-DCF model found, will use traditional DCF only")
            self.lstm_model = None
            self.lstm_scaler = None
            
        except Exception as e:
            self.logger.error(f"❌ Error loading LSTM-DCF model: {e}")
            self.lstm_model = None
            self.lstm_scaler = None
    
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
            
            # Fair value estimation (returns dict with LSTM-DCF and traditional)
            fair_value_result = self._estimate_fair_value(row)
            fair_value = fair_value_result['fair_value']
            
            # Risk assessment
            risk_level = self._assess_risk_level(row)
            
            return {
                'ticker': ticker,
                'current_price': row['current_price'],
                'fair_value_estimate': fair_value,
                'lstm_dcf_fair_value': fair_value_result.get('lstm_dcf_fair_value'),
                'traditional_fair_value': fair_value_result.get('traditional_fair_value'),
                'lstm_predicted_growth': fair_value_result.get('lstm_predicted_growth'),
                'fair_value_method': fair_value_result.get('method', 'Traditional DCF'),
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
    
    def _estimate_fair_value(self, row: pd.Series) -> Dict[str, float]:
        """
        Enhanced fair value estimation using LSTM-DCF growth model vs traditional DCF
        
        Returns:
            Dict with lstm_dcf_fair_value, traditional_fair_value, and selected fair_value
        """
        try:
            current_price = row['current_price']
            ticker = row['ticker']
            
            # Traditional DCF calculation (baseline)
            traditional_fv = self._calculate_traditional_dcf(row)
            
            # LSTM-DCF calculation (if model available) - returns (fair_value, growth_rate)
            lstm_result = self._calculate_lstm_dcf_fair_value(ticker, row) if self.lstm_model else (None, None)
            lstm_fv, lstm_growth_rate = lstm_result if lstm_result else (None, None)
            
            # Select final fair value
            if lstm_fv and lstm_fv > 0:
                # Use LSTM if available and reasonable (within 5x of current price)
                if lstm_fv / current_price <= 5.0:
                    selected_fv = lstm_fv
                    method = "LSTM-DCF"
                else:
                    # If LSTM predicts extreme value, blend with traditional
                    selected_fv = (lstm_fv * 0.6 + traditional_fv * 0.4)
                    method = "Blended (LSTM 60% + Traditional 40%)"
            else:
                selected_fv = traditional_fv
                method = "Traditional DCF"
            
            return {
                'fair_value': round(selected_fv, 2),
                'lstm_dcf_fair_value': round(lstm_fv, 2) if lstm_fv else None,
                'traditional_fair_value': round(traditional_fv, 2),
                'lstm_predicted_growth': round(lstm_growth_rate, 2) if lstm_growth_rate else None,
                'method': method
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating fair value: {e}")
            return {
                'fair_value': row['current_price'],
                'lstm_dcf_fair_value': None,
                'traditional_fair_value': row['current_price'],
                'method': "Fallback (Current Price)"
            }
    
    def _calculate_traditional_dcf(self, row: pd.Series) -> float:
        """
        Traditional DCF using P/E and P/B multiples
        """
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
        return (pe_fair_value + pb_fair_value + current_price) / 3
    
    def _calculate_lstm_dcf_fair_value(self, ticker: str, row: pd.Series) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate fair value using LSTM-DCF growth rate predictions
        
        Uses correct quarterly financial statement features that the model was trained on:
        - revenue, capex, da, fcf, operating_cf, ebitda, total_assets, net_income
        - operating_income, operating_margin, net_margin, fcf_margin, ebitda_margin
        - revenue_per_asset, fcf_per_asset, ebitda_per_asset
        
        Uses Gordon Growth Model: Fair Value = FCF * (1 + g) / (WACC - g)
        Where g = LSTM predicted growth rate
        
        Returns:
            Tuple of (fair_value, predicted_growth_rate_percentage)
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get quarterly financial statements (what the model was trained on)
            income_stmt = stock.quarterly_income_stmt
            cash_flow = stock.quarterly_cashflow
            balance_sheet = stock.quarterly_balance_sheet
            
            if income_stmt.empty or cash_flow.empty:
                self.logger.debug(f"{ticker}: No quarterly financials available")
                return None, None
            
            n_quarters = min(len(income_stmt.columns), len(cash_flow.columns), 8)
            
            if n_quarters < 4:
                self.logger.debug(f"{ticker}: Not enough quarterly data ({n_quarters} quarters)")
                return None, None
            
            def safe_get(df, names, col, default=0):
                if not isinstance(names, list):
                    names = [names]
                for name in names:
                    try:
                        if col and name in df.index:
                            val = df.loc[name, col]
                            if pd.notna(val):
                                return float(val)
                    except:
                        pass
                return default
            
            features_list = []
            
            for i in range(n_quarters):
                try:
                    col_is = income_stmt.columns[i] if i < len(income_stmt.columns) else None
                    col_cf = cash_flow.columns[i] if i < len(cash_flow.columns) else None
                    col_bs = balance_sheet.columns[i] if i < len(balance_sheet.columns) else None
                    
                    # Income statement items
                    revenue = safe_get(income_stmt, 'Total Revenue', col_is, 0)
                    net_income = safe_get(income_stmt, 'Net Income', col_is, 0)
                    operating_income = safe_get(income_stmt, 'Operating Income', col_is, 0)
                    ebitda = safe_get(income_stmt, ['EBITDA', 'Normalized EBITDA'], col_is, 0)
                    
                    # Cash flow items
                    operating_cf = safe_get(cash_flow, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'], col_cf, 0)
                    capex = abs(safe_get(cash_flow, 'Capital Expenditure', col_cf, 0))
                    da = safe_get(cash_flow, ['Depreciation And Amortization', 'Depreciation'], col_cf, 0)
                    
                    fcf = operating_cf - capex
                    
                    # Balance sheet
                    total_assets = safe_get(balance_sheet, 'Total Assets', col_bs, 1)
                    
                    # Calculate margins
                    operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
                    net_margin = (net_income / revenue * 100) if revenue > 0 else 0
                    fcf_margin = (fcf / revenue * 100) if revenue > 0 else 0
                    ebitda_margin = (ebitda / revenue * 100) if revenue > 0 else 0
                    
                    # Asset efficiency ratios
                    revenue_per_asset = revenue / total_assets if total_assets > 0 else 0
                    fcf_per_asset = fcf / total_assets if total_assets > 0 else 0
                    ebitda_per_asset = ebitda / total_assets if total_assets > 0 else 0
                    
                    # Feature vector matching training order
                    quarter_features = [
                        revenue, capex, da, fcf, operating_cf, ebitda,
                        total_assets, net_income, operating_income,
                        operating_margin, net_margin, fcf_margin, ebitda_margin,
                        revenue_per_asset, fcf_per_asset, ebitda_per_asset
                    ]
                    
                    features_list.append(quarter_features)
                    
                except Exception as e:
                    self.logger.debug(f"{ticker} Q{i} feature extraction error: {e}")
                    continue
            
            if len(features_list) < 4:
                return None, None
            
            # Reverse to chronological order (oldest first, like training)
            features_list = features_list[::-1]
            
            # Pad to sequence length if needed
            seq_len = getattr(self, 'lstm_sequence_length', 60)
            while len(features_list) < seq_len:
                features_list.append(features_list[-1])
            features_list = features_list[-seq_len:]
            
            features_array = np.array(features_list, dtype=np.float32)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply the SAVED StandardScaler from training
            if hasattr(self, 'lstm_scaler') and self.lstm_scaler is not None:
                features_scaled = self.lstm_scaler.transform(features_array)
            else:
                features_scaled = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
            
            X_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.lstm_model(X_tensor)
            
            # Model outputs [revenue_growth, fcf_growth] as percentages
            if prediction.shape[1] == 2:
                predicted_growth = prediction[0, 1].item()  # Use FCF growth
            else:
                predicted_growth = prediction[0, 0].item()
            
            # Convert from percentage to decimal and clip
            predicted_growth_rate = predicted_growth / 100.0
            predicted_growth_rate = np.clip(predicted_growth_rate, -0.30, 0.50)
            
            # Gordon Growth Model DCF
            fcf = row['free_cash_flow']
            shares_outstanding = row['market_cap'] / row['current_price'] if row['current_price'] > 0 else 1
            fcf_per_share = fcf / shares_outstanding if shares_outstanding > 0 else 0
            
            wacc = 0.10
            terminal_growth = min(max(predicted_growth_rate * 0.3, 0.01), 0.03)
            
            if fcf_per_share > 0 and predicted_growth_rate < wacc - terminal_growth:
                fair_value = (fcf_per_share * (1 + predicted_growth_rate)) / (wacc - terminal_growth)
                self.logger.info(f"LSTM-DCF {ticker}: Growth={predicted_growth_rate:.2%}, FCF/share=${fcf_per_share:.2f}, FV=${fair_value:.2f}")
                return fair_value, predicted_growth_rate * 100
            else:
                # EPS-based fallback
                eps = row['eps']
                if eps > 0:
                    peg_adjusted_pe = 15 * (1 + predicted_growth_rate)
                    peg_adjusted_pe = np.clip(peg_adjusted_pe, 8, 30)
                    fair_value = eps * peg_adjusted_pe
                    self.logger.info(f"LSTM-DCF {ticker}: Using EPS method, Growth={predicted_growth_rate:.2%}, FV=${fair_value:.2f}")
                    return fair_value, predicted_growth_rate * 100
            
            return None, None
            
        except Exception as e:
            self.logger.warning(f"LSTM-DCF calculation failed for {ticker}: {e}")
            return None, None
    
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