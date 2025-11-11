"""
Reverse DCF Validator
=====================
Tests if market price implies reasonable growth rates by solving DCF backward.

Usage:
    python scripts/consensus/test_reverse_dcf.py PYPL
    python scripts/consensus/test_reverse_dcf.py AAPL MSFT GOOGL --batch
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
import argparse
from typing import Dict, Optional
import torch

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from config.settings import MODELS_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class ReverseDCFValidator:
    """
    Reverse DCF: Solve for implied growth rate given market price
    """
    
    def __init__(self, wacc: float = 0.08, terminal_growth: float = 0.03):
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.lstm_model = self._load_lstm_model()
        self.scaler = None
        
    def _load_lstm_model(self) -> Optional[LSTMDCFModel]:
        """Load trained LSTM model"""
        try:
            model_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
            if not model_path.exists():
                logger.warning(f"LSTM model not found at {model_path}")
                return None
            
            checkpoint = torch.load(str(model_path), weights_only=False)
            
            model = LSTMDCFModel(
                input_size=checkpoint['hyperparameters']['input_size'],
                hidden_size=checkpoint['hyperparameters']['hidden_size'],
                num_layers=checkpoint['hyperparameters']['num_layers'],
                output_size=checkpoint['hyperparameters']['output_size'],
                dropout=checkpoint['hyperparameters'].get('dropout', 0.2)
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.scaler = checkpoint.get('scaler')
            
            logger.info(f"✅ LSTM model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return None
    
    def get_current_fcf(self, ticker: str) -> Optional[float]:
        """
        Get latest Free Cash Flow from financial statements
        """
        try:
            # Try to load from cached financial statements
            cashflow_path = RAW_DATA_DIR / 'financial_statements' / f'{ticker}_cashflow.csv'
            
            if cashflow_path.exists():
                df = pd.read_csv(cashflow_path)
                df['date'] = pd.to_datetime(df['fiscalDateEnding'])
                df = df.sort_values('date', ascending=False)
                
                operating_cf = pd.to_numeric(df['operatingCashflow'].iloc[0], errors='coerce')
                capex = pd.to_numeric(df['capitalExpenditures'].iloc[0], errors='coerce')
                
                if pd.notna(operating_cf) and pd.notna(capex):
                    fcf = operating_cf - abs(capex)
                    logger.info(f"{ticker} - Latest FCF: ${fcf/1e9:.2f}B")
                    return fcf
            
            # Fallback to yfinance
            logger.warning(f"{ticker} - Using yfinance fallback for FCF")
            stock = yf.Ticker(ticker)
            cashflow = stock.quarterly_cashflow
            
            if cashflow.empty:
                logger.error(f"{ticker} - No cash flow data available")
                return None
            
            if 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                operating_cf = cashflow.loc['Operating Cash Flow'].iloc[0]
                capex = cashflow.loc['Capital Expenditure'].iloc[0]
                fcf = operating_cf - abs(capex)
                return fcf
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching FCF for {ticker}: {e}")
            return None
    
    def get_lstm_growth_prediction(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get LSTM-predicted growth rates
        """
        if self.lstm_model is None:
            logger.warning("LSTM model not loaded, skipping prediction")
            return None
        
        try:
            # TODO: Implement sequence preparation from financial statements
            # For now, return None - full implementation in analyze_stock_consensus.py
            logger.warning(f"{ticker} - LSTM prediction requires full sequence preparation")
            return None
            
        except Exception as e:
            logger.error(f"LSTM prediction error for {ticker}: {e}")
            return None
    
    def calculate_npv_at_growth(
        self,
        fcf_current: float,
        growth_rate: float,
        years: int = 10
    ) -> float:
        """
        Calculate NPV of FCF stream at given growth rate
        
        Args:
            fcf_current: Current FCF
            growth_rate: Annual growth rate (e.g., 0.08 for 8%)
            years: Forecast period (default 10)
        
        Returns:
            Net Present Value
        """
        pv_sum = 0
        
        # Forecast explicit period
        for t in range(1, years + 1):
            fcf_t = fcf_current * (1 + growth_rate) ** t
            pv_fcf = fcf_t / (1 + self.wacc) ** t
            pv_sum += pv_fcf
        
        # Terminal value
        fcf_terminal = fcf_current * (1 + growth_rate) ** years * (1 + self.terminal_growth)
        terminal_value = fcf_terminal / (self.wacc - self.terminal_growth)
        pv_terminal = terminal_value / (1 + self.wacc) ** years
        
        return pv_sum + pv_terminal
    
    def solve_implied_growth(
        self,
        ticker: str,
        market_price: float,
        shares_outstanding: float,
        fcf_current: float
    ) -> Optional[float]:
        """
        Solve for implied growth rate given market price
        
        Returns:
            Implied growth rate (annual) or None if unsolvable
        """
        market_cap = market_price * shares_outstanding
        
        # Define objective function
        def objective(g):
            npv = self.calculate_npv_at_growth(fcf_current, g)
            return npv - market_cap
        
        try:
            # Binary search for growth rate
            # Search range: -50% to +50% annual growth
            implied_growth = brentq(objective, -0.5, 0.5, maxiter=100)
            return implied_growth
            
        except ValueError as e:
            # No solution in range (e.g., negative FCF, extreme overvaluation)
            logger.warning(f"{ticker} - Could not solve for implied growth: {e}")
            
            # Try to determine if overvalued or undervalued
            npv_at_zero = self.calculate_npv_at_growth(fcf_current, 0)
            if npv_at_zero < market_cap:
                logger.info(f"{ticker} - Market expects >50% growth (extremely overvalued)")
                return None  # Overvalued
            else:
                logger.info(f"{ticker} - Market expects <-50% decline (distressed)")
                return None  # Distressed
    
    def validate_stock(
        self,
        ticker: str,
        lstm_growth: Optional[float] = None
    ) -> Dict:
        """
        Full reverse DCF validation for a stock
        
        Args:
            ticker: Stock ticker symbol
            lstm_growth: Optional LSTM-predicted growth (annual %)
        
        Returns:
            Validation results dictionary
        """
        logger.info(f"{'='*80}")
        logger.info(f"Reverse DCF Validation: {ticker}")
        logger.info(f"{'='*80}")
        
        result = {
            'ticker': ticker,
            'success': False,
            'error': None
        }
        
        try:
            # 1. Get market data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            shares_outstanding = info.get('sharesOutstanding')
            
            if not current_price or not shares_outstanding:
                raise ValueError("Missing price or shares outstanding data")
            
            market_cap = current_price * shares_outstanding
            
            # 2. Get current FCF
            fcf_current = self.get_current_fcf(ticker)
            if not fcf_current or fcf_current <= 0:
                raise ValueError(f"Invalid FCF: {fcf_current}")
            
            # 3. Solve for implied growth
            implied_growth = self.solve_implied_growth(
                ticker, current_price, shares_outstanding, fcf_current
            )
            
            if implied_growth is None:
                result['error'] = "Could not solve for implied growth (extreme valuation)"
                return result
            
            # 4. Compare with LSTM if available
            validation_flag = "N/A"
            growth_diff = None
            
            if lstm_growth is not None:
                lstm_annual = lstm_growth / 100  # Convert % to decimal
                growth_diff = abs(implied_growth - lstm_annual)
                
                if growth_diff < 0.05:
                    validation_flag = "✅ OK"
                elif growth_diff < 0.10:
                    validation_flag = "⚠️ CAUTION"
                else:
                    validation_flag = "❌ WARNING"
            
            # 5. Calculate margin of safety (if LSTM available)
            margin_of_safety = None
            fair_value = None
            
            if lstm_growth is not None:
                lstm_annual = lstm_growth / 100
                npv_lstm = self.calculate_npv_at_growth(fcf_current, lstm_annual)
                fair_value = npv_lstm / shares_outstanding
                margin_of_safety = (fair_value - current_price) / current_price * 100
            
            # 6. Compile results
            result.update({
                'success': True,
                'current_price': current_price,
                'market_cap': market_cap,
                'fcf_current': fcf_current,
                'fcf_current_b': fcf_current / 1e9,
                'implied_growth': implied_growth * 100,  # Convert to %
                'lstm_growth': lstm_growth,
                'growth_diff': growth_diff * 100 if growth_diff else None,
                'validation_flag': validation_flag,
                'fair_value': fair_value,
                'margin_of_safety': margin_of_safety,
                'wacc': self.wacc * 100,
                'terminal_growth': self.terminal_growth * 100
            })
            
            # 7. Print results
            self._print_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {ticker}: {e}")
            result['error'] = str(e)
            return result
    
    def _print_results(self, result: Dict):
        """Pretty print validation results"""
        print(f"\n{'='*80}")
        print(f"📊 {result['ticker']} - Reverse DCF Validation Results")
        print(f"{'='*80}\n")
        
        print(f"💰 Market Data:")
        print(f"   Current Price:    ${result['current_price']:.2f}")
        print(f"   Market Cap:       ${result['market_cap']/1e9:.2f}B")
        print(f"   Current FCF:      ${result['fcf_current_b']:.2f}B")
        
        print(f"\n📈 Growth Analysis:")
        print(f"   Implied Growth:   {result['implied_growth']:.2f}%")
        
        if result['lstm_growth'] is not None:
            print(f"   LSTM Growth:      {result['lstm_growth']:.2f}%")
            print(f"   Difference:       {result['growth_diff']:.2f}%")
            print(f"   Validation:       {result['validation_flag']}")
        
        if result.get('fair_value'):
            print(f"\n💎 Valuation:")
            print(f"   Fair Value:       ${result['fair_value']:.2f}")
            print(f"   Margin of Safety: {result['margin_of_safety']:.1f}%")
        
        print(f"\n⚙️  Assumptions:")
        print(f"   WACC:             {result['wacc']:.1f}%")
        print(f"   Terminal Growth:  {result['terminal_growth']:.1f}%")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Reverse DCF Validator")
    parser.add_argument(
        'tickers',
        nargs='+',
        help='Stock ticker(s) to validate'
    )
    parser.add_argument(
        '--wacc',
        type=float,
        default=0.08,
        help='Weighted Average Cost of Capital (default: 8%%)'
    )
    parser.add_argument(
        '--terminal-growth',
        type=float,
        default=0.03,
        help='Terminal growth rate (default: 3%%)'
    )
    parser.add_argument(
        '--lstm-growth',
        type=float,
        default=None,
        help='LSTM-predicted growth rate for comparison (annual %%)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: process multiple tickers'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to CSV file'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ReverseDCFValidator(
        wacc=args.wacc,
        terminal_growth=args.terminal_growth
    )
    
    results = []
    
    # Process tickers
    for ticker in args.tickers:
        result = validator.validate_stock(ticker, lstm_growth=args.lstm_growth)
        results.append(result)
        
        if not args.batch and len(args.tickers) > 1:
            input("\nPress Enter to continue to next ticker...")
    
    # Save results if requested
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Results saved to {args.output}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*80}")
    print(f"📊 Validation Summary: {successful}/{len(results)} successful")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
