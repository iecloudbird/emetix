"""
Real-World LSTM-DCF Model Testing Script
Tests trained LSTM-DCF model on actual stocks to validate performance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.data.processors.time_series_processor import TimeSeriesProcessor
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class LSTMDCFRealWorldTester:
    """Test LSTM-DCF model on real stocks with comprehensive analysis"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize tester with trained model
        
        Args:
            model_path: Path to trained LSTM model (defaults to lstm_dcf_enhanced.pth)
        """
        self.model_path = model_path or str(MODELS_DIR / "lstm_dcf_enhanced.pth")
        self.model = None
        self.processor = TimeSeriesProcessor(sequence_length=60)
        
        # Load trained model
        self._load_model()
    
    def _load_model(self):
        """Load trained LSTM-DCF model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, weights_only=False, map_location='cpu')
            
            # Infer architecture from weights if hyperparameters not saved
            if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
                # New format with hyperparameters
                input_size = checkpoint['hyperparameters']['input_size']
                hidden_size = checkpoint['hyperparameters']['hidden_size']
                num_layers = checkpoint['hyperparameters']['num_layers']
                dropout = checkpoint['hyperparameters'].get('dropout', 0.2)
                output_size = checkpoint['hyperparameters'].get('output_size', 2)
                
                # Check for model_state_dict (new format) or state_dict (Lightning format)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Fallback: assume checkpoint is the state dict
                    state_dict = checkpoint
            else:
                # Simple state_dict format - infer from weights
                logger.info("⚠️  No hyperparameters found, inferring from weight shapes...")
                state_dict = checkpoint
                
                # Infer from weight shapes
                # LSTM weight_ih_l0 shape: (4*hidden_size, input_size)
                lstm_ih_shape = state_dict['lstm.weight_ih_l0'].shape
                hidden_size = lstm_ih_shape[0] // 4  # 512 / 4 = 128
                input_size = lstm_ih_shape[1]  # 12
                
                # Count layers (l0, l1, l2, ...)
                num_layers = 0
                for key in state_dict.keys():
                    if 'lstm.weight_ih_l' in key:
                        layer_num = int(key.split('_l')[-1])
                        num_layers = max(num_layers, layer_num + 1)
                
                # Output size from FC layer
                output_size = state_dict['fc.weight'].shape[0]
                dropout = 0.2
            
            # Initialize model with inferred/loaded hyperparameters
            self.model = LSTMDCFModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_size=output_size
            )
            
            # Load state dict
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully from {self.model_path}")
            logger.info(f"   Input Size: {input_size}")
            logger.info(f"   Output Size: {output_size}")
            logger.info(f"   Hidden Size: {hidden_size}")
            logger.info(f"   Num Layers: {num_layers}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def fetch_financial_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch financial data for stock from stored CSV files
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with financial metrics or None if error
        """
        try:
            from config.settings import RAW_DATA_DIR
            
            # Load from stored financial statements
            income_path = RAW_DATA_DIR / "financial_statements" / f"{ticker}_income.csv"
            cashflow_path = RAW_DATA_DIR / "financial_statements" / f"{ticker}_cashflow.csv"
            balance_path = RAW_DATA_DIR / "financial_statements" / f"{ticker}_balance.csv"
            
            if not (income_path.exists() and cashflow_path.exists() and balance_path.exists()):
                logger.warning(f"⚠️  No stored data for {ticker}, trying yfinance...")
                return self._fetch_from_yfinance(ticker)
            
            # Load CSV files
            income = pd.read_csv(income_path)
            cashflow = pd.read_csv(cashflow_path)
            balance = pd.read_csv(balance_path)
            
            # Ensure we have date column
            if 'fiscalDateEnding' in income.columns:
                income['date'] = pd.to_datetime(income['fiscalDateEnding'])
                cashflow['date'] = pd.to_datetime(cashflow['fiscalDateEnding'])
                balance['date'] = pd.to_datetime(balance['fiscalDateEnding'])
            else:
                logger.warning(f"⚠️  Date column not found for {ticker}")
                return None
            
            # Sort by date
            income = income.sort_values('date')
            cashflow = cashflow.sort_values('date')
            balance = balance.sort_values('date')
            
            # Extract key metrics (matching training data format)
            data = []
            
            for idx in range(min(len(income), len(cashflow), len(balance))):
                try:
                    row = {
                        'date': income.iloc[idx]['date'],
                        'revenue': float(income.iloc[idx].get('totalRevenue', 0)),
                        'operating_income': float(income.iloc[idx].get('operatingIncome', 0)),
                        'net_income': float(income.iloc[idx].get('netIncome', 0)),
                        'operating_cf': float(cashflow.iloc[idx].get('operatingCashflow', 0)),
                        'capex': abs(float(cashflow.iloc[idx].get('capitalExpenditures', 0))),
                        'total_assets': float(balance.iloc[idx].get('totalAssets', 0)),
                        'ebitda': float(income.iloc[idx].get('ebitda', 0)),
                        'da': float(income.iloc[idx].get('depreciationAndAmortization', 0)),
                    }
                    
                    # Calculate derived metrics
                    row['fcf'] = row['operating_cf'] - row['capex']
                    row['operating_margin'] = (row['operating_income'] / row['revenue'] * 100) if row['revenue'] != 0 else 0
                    row['net_margin'] = (row['net_income'] / row['revenue'] * 100) if row['revenue'] != 0 else 0
                    row['fcf_margin'] = (row['fcf'] / row['revenue'] * 100) if row['revenue'] != 0 else 0
                    row['ebitda_margin'] = (row['ebitda'] / row['revenue'] * 100) if row['revenue'] != 0 else 0
                    
                    # Normalized by assets
                    row['revenue_per_asset'] = row['revenue'] / row['total_assets'] if row['total_assets'] != 0 else 0
                    row['fcf_per_asset'] = row['fcf'] / row['total_assets'] if row['total_assets'] != 0 else 0
                    row['ebitda_per_asset'] = row['ebitda'] / row['total_assets'] if row['total_assets'] != 0 else 0
                    
                    data.append(row)
                except Exception as e:
                    logger.debug(f"Skipping row {idx}: {e}")
                    continue
            
            if len(data) < 60:
                logger.warning(f"⚠️  Only {len(data)} quarters available for {ticker} (need 60)")
                return None
            
            df = pd.DataFrame(data)
            df = df.sort_values('date')
            
            logger.info(f"✅ Loaded {len(df)} quarters of data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {ticker}: {e}")
            return None
    
    def _fetch_from_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fallback: Fetch from yfinance (limited data)"""
        logger.info(f"   Attempting yfinance fallback for {ticker}...")
        return None  # yfinance only has 5 quarters, not enough
    
    def prepare_sequence(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        Prepare sequence for LSTM model
        
        Args:
            df: Financial data DataFrame
            
        Returns:
            Tensor of shape (1, 60, features) or None
        """
        try:
            # Select features (must match training - 16 features)
            feature_cols = [
                'revenue', 'capex', 'da', 'fcf', 
                'operating_cf', 'ebitda', 'total_assets',
                'net_income', 'operating_income',
                'operating_margin', 'net_margin', 'fcf_margin',
                'ebitda_margin', 'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
            ]
            
            # Take last 60 quarters
            recent_data = df[feature_cols].iloc[-60:]
            
            if len(recent_data) < 60:
                logger.warning(f"⚠️  Insufficient data: {len(recent_data)} < 60")
                return None
            
            # Convert to numpy and normalize
            X = recent_data.values.astype(np.float32)
            
            # Simple standardization
            mean = np.mean(X, axis=0, keepdims=True)
            std = np.std(X, axis=0, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            X_normalized = (X - mean) / std
            
            # Convert to tensor
            sequence = torch.tensor(X_normalized, dtype=torch.float32).unsqueeze(0)  # (1, 60, features)
            
            return sequence
            
        except Exception as e:
            logger.error(f"❌ Error preparing sequence: {e}")
            return None
    
    def predict_growth(self, ticker: str) -> Optional[Dict]:
        """
        Predict growth rates for stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with predictions or None
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 Analyzing {ticker}")
        logger.info(f"{'='*80}")
        
        # Fetch data
        df = self.fetch_financial_data(ticker)
        if df is None:
            return None
        
        # Prepare sequence
        sequence = self.prepare_sequence(df)
        if sequence is None:
            return None
        
        # Predict
        with torch.no_grad():
            prediction = self.model(sequence)
            
            if prediction.shape[-1] == 2:
                revenue_growth, fcf_growth = prediction[0].cpu().numpy()
                ebitda_growth = None
            elif prediction.shape[-1] == 3:
                revenue_growth, fcf_growth, ebitda_growth = prediction[0].cpu().numpy()
            else:
                # Single output - treat as FCF growth
                fcf_growth = prediction[0].cpu().item()
                revenue_growth = fcf_growth  # Use same for both
                ebitda_growth = None
        
        # Calculate annual growth from quarterly
        annual_revenue_growth = ((1 + revenue_growth/100) ** 4 - 1) * 100
        annual_fcf_growth = ((1 + fcf_growth/100) ** 4 - 1) * 100 if fcf_growth is not None else None
        
        result = {
            'ticker': ticker,
            'quarterly_revenue_growth': revenue_growth,
            'quarterly_fcf_growth': fcf_growth,
            'annual_revenue_growth': annual_revenue_growth,
            'annual_fcf_growth': annual_fcf_growth,
            'current_fcf': df['fcf'].iloc[-1],
            'current_revenue': df['revenue'].iloc[-1]
        }
        
        logger.info(f"\n📊 Growth Predictions:")
        logger.info(f"   Revenue Growth: {revenue_growth:.2f}% QoQ → {annual_revenue_growth:.2f}% Annual")
        if fcf_growth is not None:
            logger.info(f"   FCF Growth: {fcf_growth:.2f}% QoQ → {annual_fcf_growth:.2f}% Annual")
        logger.info(f"   Current FCF: ${result['current_fcf']/1e9:.2f}B")
        logger.info(f"   Current Revenue: ${result['current_revenue']/1e9:.2f}B")
        
        return result
    
    def calculate_dcf_valuation(self, ticker: str, growth_result: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate DCF fair value using LSTM predictions with improved handling
        
        Args:
            ticker: Stock ticker
            growth_result: Growth predictions from LSTM
            df: Historical financial data for averaging
            
        Returns:
            DCF valuation results
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('currentPrice', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            if shares_outstanding == 0:
                logger.warning(f"⚠️  No shares outstanding data for {ticker}")
                return None
            
            # DCF parameters
            wacc = 0.08  # 8% WACC
            terminal_growth = 0.03  # 3% terminal growth
            
            # Use trailing 4-quarter TOTAL FCF (TTM) for annual DCF
            # NOT average per quarter - DCF expects annual figures!
            trailing_fcf_ttm = df['fcf'].iloc[-4:].sum()
            current_fcf = trailing_fcf_ttm if not pd.isna(trailing_fcf_ttm) else growth_result['current_fcf'] * 4
            
            # Check for negative/zero FCF
            if current_fcf <= 0:
                logger.warning(f"⚠️  Negative/Zero FCF (${current_fcf/1e9:.2f}B) - Using P/S fallback valuation")
                return self._calculate_ps_valuation(ticker, growth_result, info, current_price, shares_outstanding)
            
            # Get growth rate with hybrid approach for tech stocks
            raw_fcf_growth = growth_result['annual_fcf_growth'] / 100 if growth_result['annual_fcf_growth'] else growth_result['annual_revenue_growth'] / 100
            revenue_growth = growth_result['annual_revenue_growth'] / 100
            
            # Hybrid Growth Approach: For high-quality profitable companies with positive revenue growth
            # Model was trained on mixed data (58% negative FCF growth) and predicts conservatively
            # For mature profitable companies, use revenue growth as proxy with quality-based conversion
            
            if revenue_growth > 0.02 and current_fcf > 5e9:  # Revenue growing > 2% and FCF > $5B (large cap)
                # Quality-based FCF conversion rates (higher quality = higher conversion)
                fcf_margin = (current_fcf / growth_result['current_revenue']) * 100
                
                if fcf_margin > 20:  # High-quality (AAPL, MSFT-like): 20%+ FCF margin
                    fcf_conversion_rate = 0.85  # Expect FCF to track revenue closely
                elif fcf_margin > 10:  # Good quality: 10-20% FCF margin
                    fcf_conversion_rate = 0.75
                else:  # Average quality: < 10% FCF margin
                    fcf_conversion_rate = 0.60
                
                fcf_growth_floor = revenue_growth * fcf_conversion_rate
                
                # Apply floor if predicted growth is unrealistically low
                if raw_fcf_growth < fcf_growth_floor:
                    fcf_growth = fcf_growth_floor
                    logger.info(f"💡 Applied quality-adjusted growth (FCF Margin: {fcf_margin:.1f}%): " +
                              f"FCF {raw_fcf_growth*100:.1f}% → {fcf_growth*100:.1f}% " +
                              f"(Revenue {revenue_growth*100:.1f}% × {fcf_conversion_rate*100:.0f}%)")
                else:
                    fcf_growth = raw_fcf_growth
            elif revenue_growth > 0.02 and current_fcf > 0:  # Smaller profitable companies
                # Use more conservative 50% conversion for smaller caps
                fcf_growth_floor = revenue_growth * 0.50
                if raw_fcf_growth < fcf_growth_floor:
                    fcf_growth = fcf_growth_floor
                    logger.info(f"💡 Applied hybrid growth: FCF {raw_fcf_growth*100:.1f}% → {fcf_growth*100:.1f}%")
                else:
                    fcf_growth = raw_fcf_growth
            else:
                fcf_growth = raw_fcf_growth
            
            # Final cap at reasonable maximum for DCF stability (apply AFTER quality adjustment)
            original_fcf_growth = fcf_growth
            fcf_growth = max(-0.10, min(0.20, fcf_growth))  # Cap between -10% and +20% (tighter cap)
            
            if fcf_growth != original_fcf_growth:
                logger.warning(f"⚠️  Final cap applied: FCF growth {original_fcf_growth*100:.1f}% → {fcf_growth*100:.1f}% for DCF stability")
            
            # Project 10-year FCF
            fcf_forecasts = []
            fcf = current_fcf
            
            for year in range(1, 11):
                fcf = fcf * (1 + fcf_growth)
                pv_fcf = fcf / (1 + wacc) ** year
                fcf_forecasts.append(pv_fcf)
            
            # Terminal value
            terminal_fcf = fcf * (1 + terminal_growth)
            terminal_value = terminal_fcf / (wacc - terminal_growth)
            pv_terminal = terminal_value / (1 + wacc) ** 10
            
            # Fair value
            enterprise_value = sum(fcf_forecasts) + pv_terminal
            fair_value_per_share = enterprise_value / shares_outstanding
            
            # Margin of safety
            margin_of_safety = ((fair_value_per_share - current_price) / current_price) * 100
            
            result = {
                'ticker': ticker,
                'current_price': current_price,
                'fair_value': fair_value_per_share,
                'margin_of_safety': margin_of_safety,
                'enterprise_value': enterprise_value,
                'terminal_value': pv_terminal,
                'pv_fcf_10y': sum(fcf_forecasts),
                'fcf_growth_used': fcf_growth * 100
            }
            
            logger.info(f"\n💰 DCF Valuation:")
            logger.info(f"   Current Price: ${current_price:.2f}")
            logger.info(f"   Fair Value: ${fair_value_per_share:.2f}")
            logger.info(f"   Margin of Safety: {margin_of_safety:+.1f}%")
            logger.info(f"   Enterprise Value: ${enterprise_value/1e9:.2f}B")
            logger.info(f"   10Y PV FCF: ${sum(fcf_forecasts)/1e9:.2f}B")
            logger.info(f"   Terminal Value: ${pv_terminal/1e9:.2f}B")
            
            # Signal
            if margin_of_safety > 20:
                signal = "🟢 STRONG BUY"
            elif margin_of_safety > 10:
                signal = "🟢 BUY"
            elif margin_of_safety > 0:
                signal = "🟡 HOLD (Undervalued)"
            elif margin_of_safety > -10:
                signal = "🟡 HOLD (Fair)"
            else:
                signal = "🔴 SELL (Overvalued)"
            
            logger.info(f"\n   Signal: {signal}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating DCF: {e}")
            return None
    
    def _calculate_ps_valuation(self, ticker: str, growth_result: Dict, info: dict, 
                                current_price: float, shares_outstanding: int) -> Optional[Dict]:
        """
        Fallback valuation using Price-to-Sales for negative FCF companies
        
        Args:
            ticker: Stock ticker
            growth_result: Growth predictions
            info: Stock info from yfinance
            current_price: Current stock price
            shares_outstanding: Shares outstanding
            
        Returns:
            P/S based valuation
        """
        try:
            # Get revenue
            revenue = growth_result['current_revenue']
            revenue_growth = growth_result['annual_revenue_growth'] / 100
            
            # Industry average P/S multiples (approximate)
            industry_ps_multiples = {
                'Aerospace': 1.5,
                'Technology': 5.0,
                'Healthcare': 3.0,
                'Financial': 2.0,
                'Consumer': 1.8,
                'Industrial': 1.2
            }
            
            # Get sector from info
            sector = info.get('sector', 'Industrial')
            ps_multiple = industry_ps_multiples.get(sector, 2.0)  # Default 2.0x
            
            # Adjust P/S for growth (higher growth = higher multiple)
            if revenue_growth > 0.15:
                ps_multiple *= 1.3
            elif revenue_growth > 0.10:
                ps_multiple *= 1.15
            elif revenue_growth < 0:
                ps_multiple *= 0.7
            
            # Calculate fair value
            enterprise_value = revenue * ps_multiple
            fair_value_per_share = enterprise_value / shares_outstanding
            margin_of_safety = ((fair_value_per_share - current_price) / current_price) * 100
            
            logger.info(f"\n💰 P/S Valuation (Negative FCF):")
            logger.info(f"   Revenue: ${revenue/1e9:.2f}B")
            logger.info(f"   P/S Multiple: {ps_multiple:.1f}x (Sector: {sector})")
            logger.info(f"   Current Price: ${current_price:.2f}")
            logger.info(f"   Fair Value: ${fair_value_per_share:.2f}")
            logger.info(f"   Margin of Safety: {margin_of_safety:+.1f}%")
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'fair_value': fair_value_per_share,
                'margin_of_safety': margin_of_safety,
                'enterprise_value': enterprise_value,
                'valuation_method': 'P/S (Negative FCF)',
                'ps_multiple': ps_multiple,
                'revenue_growth_used': revenue_growth * 100
            }
            
        except Exception as e:
            logger.error(f"❌ Error in P/S valuation: {e}")
            return None
    
    def analyze_stock(self, ticker: str) -> Dict:
        """
        Complete analysis of single stock
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Complete analysis results
        """
        # Fetch data first
        df = self.fetch_financial_data(ticker)
        if df is None:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'Failed to fetch data'}
        
        # Predict growth
        growth_result = self.predict_growth(ticker)
        if growth_result is None:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'Failed to predict growth'}
        
        # Calculate DCF (pass DataFrame for trailing averages)
        dcf_result = self.calculate_dcf_valuation(ticker, growth_result, df)
        if dcf_result is None:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'Failed to calculate DCF'}
        
        # Combine results
        return {
            'ticker': ticker,
            'status': 'success',
            'growth': growth_result,
            'valuation': dcf_result
        }
    
    def batch_analyze(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze multiple stocks and return summary
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for ticker in tickers:
            try:
                result = self.analyze_stock(ticker)
                
                if result['status'] == 'success':
                    results.append({
                        'Ticker': ticker,
                        'Current Price': f"${result['valuation']['current_price']:.2f}",
                        'Fair Value': f"${result['valuation']['fair_value']:.2f}",
                        'Margin of Safety': f"{result['valuation']['margin_of_safety']:+.1f}%",
                        'Revenue Growth': f"{result['growth']['annual_revenue_growth']:.1f}%",
                        'FCF Growth': f"{result['growth']['annual_fcf_growth']:.1f}%" if result['growth']['annual_fcf_growth'] else "N/A",
                        'Status': '✅'
                    })
                else:
                    results.append({
                        'Ticker': ticker,
                        'Current Price': 'N/A',
                        'Fair Value': 'N/A',
                        'Margin of Safety': 'N/A',
                        'Revenue Growth': 'N/A',
                        'FCF Growth': 'N/A',
                        'Status': f"❌ {result.get('reason', 'Error')}"
                    })
            except Exception as e:
                logger.error(f"❌ Error analyzing {ticker}: {e}")
                results.append({
                    'Ticker': ticker,
                    'Current Price': 'N/A',
                    'Fair Value': 'N/A',
                    'Margin of Safety': 'N/A',
                    'Revenue Growth': 'N/A',
                    'FCF Growth': 'N/A',
                    'Status': f"❌ {str(e)[:30]}"
                })
        
        df = pd.DataFrame(results)
        return df


def main():
    """Main test execution"""
    logger.info("\n" + "="*80)
    logger.info("🧪 LSTM-DCF REAL-WORLD TESTING")
    logger.info("="*80 + "\n")
    
    # Initialize tester
    tester = LSTMDCFRealWorldTester()
    
    # Test stocks (mix of sectors) - using stocks we have data for
    test_stocks = [
        'AAPL',   # Tech - Large Cap
        'MSFT',   # Tech - Large Cap
        'AMZN',   # Tech/E-commerce
        'AMD',    # Semiconductors
        'JPM',    # Financials (using BAC if JPM not available)
        'JNJ',    # Healthcare (using ABT if not available)
        'PG',     # Consumer Defensive (using similar)
        'BA',     # Aerospace
        'ADBE',   # Software
        'ACN'     # Consulting/IT Services
    ]
    
    # Single stock detailed analysis
    logger.info("\n" + "="*80)
    logger.info("📊 DETAILED SINGLE STOCK ANALYSIS")
    logger.info("="*80)
    
    result = tester.analyze_stock('AAPL')
    
    # Batch analysis
    logger.info("\n\n" + "="*80)
    logger.info("📊 BATCH ANALYSIS SUMMARY")
    logger.info("="*80 + "\n")
    
    summary_df = tester.batch_analyze(test_stocks)
    
    logger.info("\n" + str(summary_df.to_string(index=False)))
    
    # Save results
    output_path = PROCESSED_DATA_DIR / "lstm_dcf_test_results.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Results saved to: {output_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ TESTING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
