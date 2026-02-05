"""
LSTM-DCF v2 Inference Processor
===============================
Prepares data for v2 model inference using quarterly financial data.

The v2 model was trained on quarterly fundamentals, NOT daily prices.
This processor fetches real-time quarterly data and prepares it for inference.

Required Features (16):
    revenue, capex, da, fcf, operating_cf, ebitda, total_assets,
    net_income, operating_income, operating_margin, net_margin, fcf_margin,
    ebitda_margin, revenue_per_asset, fcf_per_asset, ebitda_per_asset

Author: Emetix ML Pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, Dict, Any
import yfinance as yf

from config.logging_config import get_logger
from config.settings import MODELS_DIR

logger = get_logger(__name__)


# v2 model feature columns (must match training)
V2_FEATURE_COLS = [
    'revenue', 'capex', 'da', 'fcf', 'operating_cf', 'ebitda', 'total_assets',
    'net_income', 'operating_income', 'operating_margin', 'net_margin', 'fcf_margin',
    'ebitda_margin', 'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
]


class LSTMV2Processor:
    """
    Data processor for LSTM-DCF v2 inference.
    
    Uses quarterly financial statements instead of daily prices.
    Designed to match the training pipeline exactly.
    """
    
    def __init__(self, sequence_length: int = 8):
        """
        Initialize v2 processor.
        
        Args:
            sequence_length: Number of quarters for sequence (default 8 = 2 years)
        """
        self.sequence_length = sequence_length
        self.feature_cols = V2_FEATURE_COLS
        
        logger.info(f"LSTM V2 Processor initialized (seq_len={sequence_length})")
    
    def fetch_quarterly_fundamentals(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch quarterly financial data from yfinance.
        
        yfinance provides quarterly data which is more accessible than
        Finnhub for real-time inference.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with quarterly fundamentals or None on error
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements (quarterly)
            # yfinance format: rows=metrics, columns=dates
            income_stmt = stock.quarterly_income_stmt
            cash_flow = stock.quarterly_cash_flow
            balance_sheet = stock.quarterly_balance_sheet
            
            if income_stmt is None or income_stmt.empty:
                logger.warning(f"No income statement data for {ticker}")
                return None
            
            # Transpose so dates are rows (dates as index, metrics as columns)
            income_df = income_stmt.T.sort_index()  # Sort by date ascending
            cash_df = cash_flow.T.sort_index() if cash_flow is not None and not cash_flow.empty else pd.DataFrame()
            balance_df = balance_sheet.T.sort_index() if balance_sheet is not None and not balance_sheet.empty else pd.DataFrame()
            
            # Build fundamentals DataFrame with date index
            df = pd.DataFrame(index=income_df.index)
            df['date'] = income_df.index
            
            # Income Statement metrics
            df['revenue'] = self._safe_get_series(income_df, ['Total Revenue', 'Revenue', 'Operating Revenue'])
            df['net_income'] = self._safe_get_series(income_df, ['Net Income', 'Net Income Common Stockholders'])
            df['operating_income'] = self._safe_get_series(income_df, ['Operating Income', 'EBIT', 'Total Operating Income As Reported'])
            df['ebitda'] = self._safe_get_series(income_df, ['EBITDA', 'Normalized EBITDA'])
            
            # Cash Flow metrics
            if not cash_df.empty:
                df['operating_cf'] = self._safe_get_series(cash_df, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities', 'Changes In Cash'])
                df['capex'] = self._safe_get_series(cash_df, ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of Ppe']).abs()
                df['da'] = self._safe_get_series(cash_df, ['Depreciation And Amortization', 'Depreciation', 'Depreciation Amortization Depletion'])
                df['fcf'] = self._safe_get_series(cash_df, ['Free Cash Flow'])
            else:
                df['operating_cf'] = 0.0
                df['capex'] = 0.0
                df['da'] = 0.0
                df['fcf'] = 0.0
            
            # Balance Sheet
            if not balance_df.empty:
                df['total_assets'] = self._safe_get_series(balance_df, ['Total Assets'])
            else:
                df['total_assets'] = 1.0  # Avoid division by zero
            
            # Calculate FCF if not available directly
            if df['fcf'].isna().all() or (df['fcf'] == 0).all():
                df['fcf'] = df['operating_cf'] - df['capex']
            
            # Calculate derived metrics (margins)
            df['operating_margin'] = (df['operating_income'] / df['revenue'].replace(0, np.nan)) * 100
            df['net_margin'] = (df['net_income'] / df['revenue'].replace(0, np.nan)) * 100
            df['fcf_margin'] = (df['fcf'] / df['revenue'].replace(0, np.nan)) * 100
            df['ebitda_margin'] = (df['ebitda'] / df['revenue'].replace(0, np.nan)) * 100
            
            # Calculate asset ratios
            df['revenue_per_asset'] = df['revenue'] / df['total_assets'].replace(0, np.nan)
            df['fcf_per_asset'] = df['fcf'] / df['total_assets'].replace(0, np.nan)
            df['ebitda_per_asset'] = df['ebitda'] / df['total_assets'].replace(0, np.nan)
            
            # Clean up - reset index so we have numeric index
            df = df.reset_index(drop=True)
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            logger.info(f"Fetched {len(df)} quarters of data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching quarterly data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _safe_get_series(self, df: pd.DataFrame, columns: list) -> pd.Series:
        """Safely get a column as Series, trying multiple column names."""
        for col in columns:
            if col in df.columns:
                return df[col].fillna(0)
        return pd.Series(0.0, index=df.index)
    
    def prepare_inference_sequence(
        self,
        ticker: str,
        feature_scaler: Any = None,
        min_quarters: int = 4,
    ) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Prepare a sequence for v2 model inference.
        
        Args:
            ticker: Stock symbol
            feature_scaler: Fitted scaler from model checkpoint
            min_quarters: Minimum quarters needed (will pad if < sequence_length)
            
        Returns:
            Tuple of (input_tensor, metadata_dict) or None on error
        """
        # Fetch quarterly data
        df = self.fetch_quarterly_fundamentals(ticker)
        
        if df is None:
            return None
        
        # Check we have minimum quarters
        if len(df) < min_quarters:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} quarters (need at least {min_quarters})")
            return None
        
        # Select features in correct order
        try:
            features = df[self.feature_cols].values
        except KeyError as e:
            logger.error(f"Missing features for {ticker}: {e}")
            missing = [c for c in self.feature_cols if c not in df.columns]
            logger.error(f"Missing columns: {missing}")
            return None
        
        # Handle case where we have fewer quarters than sequence_length
        actual_quarters = len(features)
        if actual_quarters < self.sequence_length:
            # Pad with zeros at the beginning (oldest data)
            padding = np.zeros((self.sequence_length - actual_quarters, features.shape[1]))
            features = np.vstack([padding, features])
            logger.info(f"Padded {ticker} from {actual_quarters} to {self.sequence_length} quarters")
        else:
            # Use only the last sequence_length quarters
            features = features[-self.sequence_length:]
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features if scaler provided
        if feature_scaler is not None:
            try:
                features = feature_scaler.transform(features)
            except Exception as e:
                logger.warning(f"Could not apply scaler: {e}. Using unscaled features.")
        
        # Convert to tensor (batch_size=1, seq_len, features)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Metadata for interpretation
        metadata = {
            'ticker': ticker,
            'quarters_used': self.sequence_length,
            'latest_quarter': str(df['date'].iloc[-1]) if 'date' in df.columns else 'unknown',
            'feature_cols': self.feature_cols,
            'shape': tensor.shape
        }
        
        logger.info(f"Prepared inference tensor for {ticker}: shape={tensor.shape}")
        return tensor, metadata
    
    def interpret_prediction(
        self,
        prediction: torch.Tensor,
        target_scaler: Any = None,
        current_price: float = 0.0
    ) -> Dict[str, Any]:
        """
        Interpret v2 model prediction.
        
        v2 model outputs: [revenue_growth, fcf_growth] (percentages)
        
        Args:
            prediction: Model output tensor
            target_scaler: Fitted scaler for inverse transform
            current_price: Current stock price for valuation
            
        Returns:
            Dictionary with interpreted results
        """
        pred = prediction.detach().numpy().flatten()
        
        # Inverse transform if scaler provided
        if target_scaler is not None:
            try:
                pred = target_scaler.inverse_transform(pred.reshape(1, -1)).flatten()
            except Exception as e:
                logger.warning(f"Could not inverse transform prediction: {e}")
        
        revenue_growth = float(pred[0]) if len(pred) > 0 else 0.0
        fcf_growth = float(pred[1]) if len(pred) > 1 else 0.0
        
        # Calculate implied valuation based on growth forecasts
        # Simple DCF approximation: higher growth = higher fair value
        # Using Gordon Growth Model approximation
        discount_rate = 0.10  # 10% discount rate
        terminal_growth = min(revenue_growth / 100, 0.03)  # Cap at 3%
        
        # Simple valuation multiple based on growth
        growth_multiple = 1 + (revenue_growth + fcf_growth) / 200  # Average growth impact
        implied_fair_value = current_price * growth_multiple if current_price > 0 else 0
        
        return {
            'revenue_growth_forecast': revenue_growth,
            'fcf_growth_forecast': fcf_growth,
            'growth_multiple': growth_multiple,
            'implied_fair_value': implied_fair_value,
            'upside_percent': (growth_multiple - 1) * 100,
            'model': 'lstm_dcf_v2'
        }


# Factory function for compatibility
def create_v2_processor(model_metadata: Dict = None) -> LSTMV2Processor:
    """
    Create a v2 processor configured from model metadata.
    
    Args:
        model_metadata: Metadata from model checkpoint
        
    Returns:
        Configured LSTMV2Processor
    """
    seq_length = 8  # v2 default
    
    if model_metadata:
        seq_length = model_metadata.get('sequence_length', 8)
    
    return LSTMV2Processor(sequence_length=seq_length)


if __name__ == "__main__":
    # Test the processor
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LSTM v2 processor")
    parser.add_argument("ticker", type=str, help="Stock ticker to test")
    args = parser.parse_args()
    
    processor = LSTMV2Processor(sequence_length=8)
    
    # Test fetch
    df = processor.fetch_quarterly_fundamentals(args.ticker)
    if df is not None:
        print(f"\n✓ Fetched {len(df)} quarters for {args.ticker}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nLatest quarter:")
        print(df.tail(1).T)
    else:
        print(f"\n✗ Failed to fetch data for {args.ticker}")
