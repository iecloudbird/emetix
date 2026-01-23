"""
LSTM-DCF 10-Year Backtest (LEGACY - Stock Return Based)
=======================================================

⚠️ NOTE (Jan 2026): This script evaluates LSTM predictions against STOCK RETURNS.
For the recommended FCF-based evaluation (comparing predicted FCF growth to actual 
FCF growth), use: evaluate_fcf_predictions.py

This legacy script is kept for reference and for comparing model performance
against market benchmarks (SPY).

Methodology:
1. For each historical point (e.g., 2015-01-01):
   - Use LSTM to predict growth rates (revenue_growth, fcf_growth)
   - Apply Gordon Growth DCF to calculate fair value
   - Track actual price trajectory over next 10 years
   
2. Compare predictions vs reality:
   - Predicted Fair Value vs Actual Price (10 years later)
   - Direction accuracy (did we predict up/down correctly?)
   - Magnitude accuracy (how close was the predicted return?)
   - Investment performance (if we bought based on LSTM-DCF)

Usage:
    # Full backtest (2015-2025, all available stocks)
    python scripts/evaluation/backtest_lstm_dcf_10year.py
    
    # Specific tickers
    python scripts/evaluation/backtest_lstm_dcf_10year.py --tickers AAPL MSFT GOOGL
    
    # Shorter window (5 years)
    python scripts/evaluation/backtest_lstm_dcf_10year.py --years 5
    
    # Quick test (3 stocks)
    python scripts/evaluation/backtest_lstm_dcf_10year.py --quick
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
import time
warnings.filterwarnings('ignore')


def yf_with_retry(ticker_symbol: str, max_retries: int = 3, initial_delay: float = 5.0):
    """
    Create a yfinance Ticker with retry logic for rate limiting.
    
    Args:
        ticker_symbol: Stock ticker (e.g., 'AAPL', 'SPY')
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (will exponentially increase)
        
    Returns:
        yfinance Ticker object, or raises Exception after all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Test the ticker by accessing info (this triggers the actual API call)
            _ = ticker.info.get('symbol', ticker_symbol)
            return ticker
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            if 'rate' in error_msg or 'too many' in error_msg or '429' in error_msg:
                if attempt < max_retries:
                    print(f"  ⏳ Rate limited on {ticker_symbol}, waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"Rate limited after {max_retries} retries for {ticker_symbol}")
            else:
                # Non-rate-limiting error, don't retry
                raise e
    
    raise last_exception if last_exception else Exception(f"Failed to fetch {ticker_symbol}")

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.valuation.fcf_dcf_model import FCFDCFModel
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class LSTMDCFBacktester:
    """
    Backtest LSTM-DCF predictions against actual 10-year price trajectories
    """
    
    def __init__(
        self,
        model_path: str = None,
        lookback_years: int = 10,
        projection_years: int = 5,
        discount_rate: float = 0.10,
        margin_of_safety: float = 0.20
    ):
        """
        Args:
            model_path: Path to trained LSTM model
            lookback_years: How many years back to test (default 10)
            projection_years: DCF projection horizon (default 5)
            discount_rate: WACC for DCF (default 10%)
            margin_of_safety: Required MoS for buy signal (default 20%)
        """
        self.model_path = model_path or str(MODELS_DIR / "lstm_dcf_enhanced.pth")
        self.lookback_years = lookback_years
        self.projection_years = projection_years
        self.margin_of_safety = margin_of_safety
        
        # Models
        self.lstm_model = None
        self.dcf_model = FCFDCFModel(
            discount_rate=discount_rate,
            terminal_growth_rate=0.025,
            projection_years=projection_years
        )
        
        # Feature columns (must match LSTM training)
        self.feature_cols = [
            'revenue', 'capex', 'da', 'fcf', 
            'operating_cf', 'ebitda', 'total_assets',
            'net_income', 'operating_income',
            'operating_margin', 'net_margin', 'fcf_margin',
            'ebitda_margin', 'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
        ]
        
        # Results storage
        self.backtest_results = []
        
        # Scaler from training (will be loaded with model)
        self.scaler = None  # For v1 models
        self.feature_scaler = None  # For v2 models
        self.target_scaler = None  # For v2 models
        self.model_version = 'v1'  # Will be set based on checkpoint
        
        self._load_lstm_model()
    
    def _load_lstm_model(self):
        """Load trained LSTM model (supports both v1 and v2 architectures)"""
        try:
            checkpoint = torch.load(self.model_path, weights_only=False, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
                hp = checkpoint['hyperparameters']
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
                
                # Detect model version based on checkpoint contents
                if 'feature_scaler' in checkpoint:
                    self.model_version = 'v2'
                    self.feature_scaler = checkpoint['feature_scaler']
                    self.target_scaler = checkpoint['target_scaler']
                    logger.info(f"✅ Detected v2 model with feature_scaler and target_scaler")
                    
                    # V2 uses a different architecture (with BatchNorm)
                    # Import and use v2 model class
                    from scripts.lstm.train_lstm_dcf_v2 import LSTMDCFModelV2
                    self.lstm_model = LSTMDCFModelV2(
                        input_size=hp['input_size'],
                        hidden_size=hp['hidden_size'],
                        num_layers=hp['num_layers'],
                        dropout=hp.get('dropout', 0.3),
                        output_size=hp.get('output_size', 2)
                    )
                else:
                    self.model_version = 'v1'
                    if 'scaler' in checkpoint:
                        self.scaler = checkpoint['scaler']
                        logger.info(f"✅ Detected v1 model with scaler")
                    else:
                        logger.warning(f"⚠️  No scaler in checkpoint - will fall back to per-stock normalization")
                    
                    self.lstm_model = LSTMDCFModel(
                        input_size=hp['input_size'],
                        hidden_size=hp['hidden_size'],
                        num_layers=hp['num_layers'],
                        dropout=hp.get('dropout', 0.2),
                        output_size=hp.get('output_size', 2)
                    )
                
                self.lstm_model.load_state_dict(state_dict)
                self.lstm_model.eval()
                
                logger.info(f"✅ LSTM model ({self.model_version}) loaded: {hp['input_size']} features → {hp['output_size']} outputs")
            else:
                raise ValueError("Model checkpoint missing hyperparameters")
                
        except Exception as e:
            logger.error(f"❌ Failed to load LSTM model: {e}")
            raise
    
    def load_historical_fundamentals(self) -> pd.DataFrame:
        """Load quarterly fundamentals for all stocks"""
        data_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
        
        if not data_path.exists():
            logger.warning(f"⚠️  Training data not found at {data_path}")
            logger.info("Attempting to load from raw data...")
            return self._load_from_raw_data()
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"✅ Loaded {len(df)} quarterly records from {len(df['ticker'].unique())} tickers")
        
        return df
    
    def _load_from_raw_data(self) -> pd.DataFrame:
        """Fallback: Load from raw fundamentals directory"""
        fundamentals_dir = RAW_DATA_DIR / "fundamentals"
        
        if not fundamentals_dir.exists():
            raise FileNotFoundError(f"No data found at {fundamentals_dir}")
        
        all_data = []
        for csv_file in fundamentals_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['ticker'] = csv_file.stem
                all_data.append(df)
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {csv_file.name}: {e}")
        
        if not all_data:
            raise ValueError("No fundamental data found")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        logger.info(f"✅ Loaded {len(combined_df)} records from raw data")
        return combined_df
    
    def get_historical_point_data(
        self,
        ticker: str,
        target_date: pd.Timestamp,
        df: pd.DataFrame,
        sequence_length: int = 60
    ) -> Optional[Dict]:
        """
        Get data needed for LSTM prediction at a historical point in time
        
        Args:
            ticker: Stock ticker
            target_date: The date we're making the prediction FROM
            df: Full historical fundamentals dataframe
            sequence_length: Number of quarters needed (60 default, but will use min 20)
            
        Returns:
            Dict with features, fcf, shares, price, or None if insufficient data
        """
        # Get ticker data up to target_date (use only past data)
        ticker_data = df[
            (df['ticker'] == ticker) & 
            (df['date'] <= target_date)
        ].sort_values('date')
        
        # Need at least 20 quarters minimum (5 years), prefer 60 (15 years)
        MIN_SEQUENCE = 20
        
        if len(ticker_data) < MIN_SEQUENCE:
            return None
        
        # Use all available history up to sequence_length
        actual_sequence = min(len(ticker_data), sequence_length)
        
        # Get last N quarters as features
        feature_window = ticker_data.iloc[-actual_sequence:][self.feature_cols].values
        
        # Pad with zeros if we have less than 60 quarters
        if actual_sequence < sequence_length:
            padding = np.zeros((sequence_length - actual_sequence, len(self.feature_cols)))
            feature_window = np.vstack([padding, feature_window])
            logger.debug(f"{ticker} @ {target_date.date()}: Using {actual_sequence}/{sequence_length} quarters (padded)")
        
        # Get latest fundamentals (at target_date)
        latest = ticker_data.iloc[-1]
        
        # Get shares outstanding from yfinance (since not in training data)
        shares_outstanding = self._get_shares_outstanding(ticker, target_date)
        if shares_outstanding is None or shares_outstanding <= 0:
            return None
        
        return {
            'ticker': ticker,
            'date': target_date,
            'features': feature_window,  # (60, 16) - padded if necessary
            'current_fcf': latest.get('fcf', 0),
            'shares_outstanding': shares_outstanding,
            'net_debt': 0  # Simplified: ignore debt for backtest (conservative)
        }
    
    def _get_shares_outstanding(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """
        Get shares outstanding from yfinance
        Note: yfinance gives current shares, not historical (limitation)
        Falls back to market cap approximation if shares not available
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            shares = info.get('sharesOutstanding', None)
            
            if shares and shares > 0:
                return float(shares)
            
            # Fallback 1: Try from balance sheet
            try:
                bs = stock.quarterly_balance_sheet
                if not bs.empty and 'Share Issued' in bs.index:
                    shares_from_bs = bs.loc['Share Issued'].iloc[0]
                    if shares_from_bs > 0:
                        return float(shares_from_bs)
            except:
                pass
            
            # Fallback 2: Approximate from market cap and current price
            try:
                market_cap = info.get('marketCap', None)
                
                # Get historical price at date
                hist = stock.history(start=date - timedelta(days=7), end=date + timedelta(days=7))
                if not hist.empty and market_cap and market_cap > 0:
                    # Convert hist index to timezone-naive
                    if hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)
                    if date.tz is not None:
                        date_naive = date.tz_localize(None)
                    else:
                        date_naive = date
                    
                    time_diffs = np.abs((hist.index - date_naive).total_seconds())
                    closest_idx = time_diffs.argmin()
                    price = float(hist['Close'].iloc[closest_idx])
                    
                    if price > 0:
                        approx_shares = market_cap / price
                        logger.debug(f"Using approximated shares for {ticker}: {approx_shares:,.0f}")
                        return approx_shares
            except:
                pass
            
            logger.warning(f"⚠️  Could not get shares outstanding for {ticker}")
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get shares for {ticker}: {e}")
            return None
    
    def predict_growth_rates(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict growth rates using LSTM
        
        Args:
            features: (60, 16) array of quarterly features
            
        Returns:
            (revenue_growth, fcf_growth) as percentages (e.g., 0.15 = 15%)
        """
        # Normalize features based on model version
        if self.model_version == 'v2' and self.feature_scaler is not None:
            # V2 uses RobustScaler for features
            original_shape = features.shape
            features_flat = features.reshape(-1, features.shape[-1])
            features_normalized = self.feature_scaler.transform(features_flat)
            features_normalized = features_normalized.reshape(original_shape)
            logger.debug("Using v2 feature_scaler for normalization")
        elif self.scaler is not None:
            # V1 uses StandardScaler
            original_shape = features.shape
            features_flat = features.reshape(-1, features.shape[-1])
            features_normalized = self.scaler.transform(features_flat)
            features_normalized = features_normalized.reshape(original_shape)
            logger.debug("Using v1 scaler for normalization")
        else:
            # Fallback to per-stock normalization (NOT RECOMMENDED - causes mode collapse)
            logger.warning("⚠️  Using per-stock normalization - predictions may be unreliable!")
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)
            std[std == 0] = 1
            features_normalized = (features - mean) / std
        
        # Convert to tensor
        X = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)  # (1, 60, 16)
        
        # Predict
        with torch.no_grad():
            prediction = self.lstm_model(X)
            growth_rates = prediction[0].cpu().numpy()  # (2,) [revenue_growth, fcf_growth]
        
        # For v2 models, inverse transform the scaled predictions
        if self.model_version == 'v2' and self.target_scaler is not None:
            growth_rates = self.target_scaler.inverse_transform(growth_rates.reshape(1, -1))[0]
        
        revenue_growth = float(growth_rates[0])
        fcf_growth = float(growth_rates[1])
        
        # Check if model output is in percentage form (e.g., 15.59 instead of 0.1559)
        # If values are > 1 or < -1, they're likely in wrong scale
        if abs(revenue_growth) > 1:
            logger.debug(f"Revenue growth {revenue_growth:.2f} appears to be in wrong scale, dividing by 100")
            revenue_growth = revenue_growth / 100
        if abs(fcf_growth) > 1:
            logger.debug(f"FCF growth {fcf_growth:.2f} appears to be in wrong scale, dividing by 100")
            fcf_growth = fcf_growth / 100
        
        # No clipping - let the model express full range of predictions
        # Model is trained on winsorized data (5th-95th percentile) so extreme values are rare
        
        return revenue_growth, fcf_growth
    
    def calculate_lstm_dcf_fair_value(
        self,
        current_fcf: float,
        fcf_growth_rate: float,
        shares_outstanding: float,
        net_debt: float
    ) -> Optional[float]:
        """
        Calculate fair value using LSTM-predicted growth rate
        
        Args:
            current_fcf: Current FCF ($)
            fcf_growth_rate: LSTM-predicted FCF growth (e.g., 0.15 = 15%/year)
            shares_outstanding: Total shares
            net_debt: Total debt - cash
            
        Returns:
            Fair value per share, or None if calculation fails
        """
        if shares_outstanding is None or shares_outstanding <= 0:
            return None
        
        if current_fcf <= 0:
            # Use conservative approach for negative FCF
            current_fcf = abs(current_fcf) * 0.3
        
        # Project FCF using LSTM growth rate (assume declining growth)
        growth_rates = []
        for year in range(self.projection_years):
            # Decay growth rate over time (e.g., 15% → 12% → 9% → 6% → 4%)
            decay_factor = 0.85 ** year
            annual_growth = fcf_growth_rate * decay_factor
            growth_rates.append(annual_growth)
        
        # Calculate DCF
        dcf_result = self.dcf_model.calculate_intrinsic_value(
            current_fcf=current_fcf,
            fcf_growth_rates=growth_rates,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt
        )
        
        if dcf_result is None:
            return None
        
        return dcf_result['intrinsic_value_per_share']
    
    def get_historical_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Get actual stock price at a specific date"""
        try:
            # Ensure date is timezone-naive for yfinance
            if date.tz is not None:
                date = date.tz_localize(None)
            
            stock = yf.Ticker(ticker)
            
            # Fetch price within 7-day window around target date
            start_date = date - timedelta(days=7)
            end_date = date + timedelta(days=7)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Convert hist index to timezone-naive if needed
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            
            # Get closest date using np.abs
            time_diffs = np.abs((hist.index - date).total_seconds())
            closest_idx = time_diffs.argmin()
            return float(hist['Close'].iloc[closest_idx])
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch price for {ticker} at {date}: {e}")
            return None
    
    def get_future_price(
        self,
        ticker: str,
        start_date: pd.Timestamp,
        years_ahead: int
    ) -> Optional[float]:
        """Get stock price N years in the future"""
        future_date = start_date + timedelta(days=years_ahead * 365)
        return self.get_historical_price(ticker, future_date)
    
    def backtest_single_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[Dict]:
        """
        Backtest one ticker across all historical points
        
        Args:
            ticker: Stock ticker
            df: Full fundamentals dataframe
            start_date: Start of backtest period
            end_date: End of backtest period (must be lookback_years before today)
            
        Returns:
            List of backtest results
        """
        results = []
        
        # Get available quarters for this ticker
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        # Get all quarters in backtest period
        test_quarters = ticker_data[
            (ticker_data['date'] >= start_date) & 
            (ticker_data['date'] <= end_date)
        ]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Backtesting {ticker}: {len(test_quarters)} historical points")
        logger.info(f"{'='*80}")
        
        for idx, (_, quarter_row) in enumerate(test_quarters.iterrows(), 1):
            prediction_date = quarter_row['date']
            
            # Get historical data at this point
            try:
                point_data = self.get_historical_point_data(ticker, prediction_date, df)
            except Exception as e:
                logger.debug(f"⏭️  Skipped {ticker} @ {prediction_date}: Error getting data - {e}")
                continue
            
            if point_data is None:
                logger.debug(f"⏭️  Skipped {ticker} @ {prediction_date}: Insufficient data")
                continue
            
            # Get price at prediction date
            current_price = self.get_historical_price(ticker, prediction_date)
            if current_price is None:
                logger.debug(f"⏭️  Skipped {ticker} @ {prediction_date}: No price data")
                continue
            
            # Predict growth rates using LSTM
            revenue_growth, fcf_growth = self.predict_growth_rates(point_data['features'])
            
            # Calculate LSTM-DCF fair value
            lstm_fair_value = self.calculate_lstm_dcf_fair_value(
                current_fcf=point_data['current_fcf'],
                fcf_growth_rate=fcf_growth,
                shares_outstanding=point_data['shares_outstanding'],
                net_debt=point_data['net_debt']
            )
            
            if lstm_fair_value is None or lstm_fair_value <= 0:
                logger.debug(f"⏭️  Skipped {ticker} @ {prediction_date}: Invalid fair value")
                continue
            
            # Calculate margin of safety
            mos = (lstm_fair_value - current_price) / current_price
            
            # Get future price (N years later)
            future_price = self.get_future_price(ticker, prediction_date, self.lookback_years)
            
            if future_price is None:
                logger.debug(f"⏭️  Skipped {ticker} @ {prediction_date}: No future price")
                continue
            
            # Calculate actual returns
            actual_return = (future_price - current_price) / current_price
            predicted_return = mos
            
            # Determine if prediction was correct
            direction_correct = (predicted_return > 0 and actual_return > 0) or \
                               (predicted_return < 0 and actual_return < 0)
            
            # Investment decision (buy if MoS > threshold)
            should_buy = mos > self.margin_of_safety
            
            # Investment outcome (if we followed the signal)
            if should_buy:
                investment_return = actual_return
            else:
                investment_return = 0  # Hold cash (0% return)
            
            result = {
                'ticker': ticker,
                'prediction_date': prediction_date,
                'current_price': current_price,
                'lstm_revenue_growth': revenue_growth,
                'lstm_fcf_growth': fcf_growth,
                'lstm_fair_value': lstm_fair_value,
                'margin_of_safety': mos,
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'future_price': future_price,
                'years_ahead': self.lookback_years,
                'direction_correct': direction_correct,
                'should_buy': should_buy,
                'investment_return': investment_return,
                'current_fcf': point_data['current_fcf'],
                'shares_outstanding': point_data['shares_outstanding']
            }
            
            results.append(result)
            
            # Log progress every 10 predictions
            if idx % 10 == 0 or idx == len(test_quarters):
                logger.info(f"  Progress: {idx}/{len(test_quarters)} quarters | "
                          f"Valid predictions: {len(results)}")
        
        logger.info(f"✅ {ticker}: {len(results)} valid predictions completed\n")
        
        return results
    
    def run_backtest(
        self,
        tickers: List[str] = None,
        max_tickers: int = None
    ) -> pd.DataFrame:
        """
        Run full backtest across multiple tickers
        
        Args:
            tickers: List of tickers to test (None = all available)
            max_tickers: Limit number of tickers (for quick testing)
            
        Returns:
            DataFrame with all backtest results
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 LSTM-DCF 10-YEAR BACKTEST")
        logger.info("="*80)
        logger.info(f"Lookback Period: {self.lookback_years} years")
        logger.info(f"DCF Projection: {self.projection_years} years")
        logger.info(f"Margin of Safety: {self.margin_of_safety:.0%}")
        logger.info(f"Discount Rate: {self.dcf_model.discount_rate:.0%}")
        
        # Load historical fundamentals
        df = self.load_historical_fundamentals()
        
        # Determine backtest date range
        today = pd.Timestamp.now()
        end_date = today - timedelta(days=self.lookback_years * 365)  # Must have N years of future data
        start_date = end_date - timedelta(days=10 * 365)  # Test over 10-year historical window
        
        logger.info(f"\nBacktest Window: {start_date.date()} to {end_date.date()}")
        logger.info(f"(Predictions made during this period, validated {self.lookback_years} years later)")
        
        # Get tickers to test
        if tickers is None:
            available_tickers = df['ticker'].unique()
            tickers = list(available_tickers)
        
        if max_tickers:
            tickers = tickers[:max_tickers]
        
        logger.info(f"\nTickers to backtest: {len(tickers)}")
        logger.info(f"Tickers: {', '.join(tickers[:10])}" + 
                   (f"... and {len(tickers)-10} more" if len(tickers) > 10 else ""))
        
        # Run backtest for each ticker
        all_results = []
        
        for ticker_idx, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{ticker_idx}/{len(tickers)}] Testing {ticker}...")
            
            try:
                ticker_results = self.backtest_single_ticker(
                    ticker=ticker,
                    df=df,
                    start_date=start_date,
                    end_date=end_date
                )
                
                all_results.extend(ticker_results)
                
            except Exception as e:
                logger.error(f"❌ Failed to backtest {ticker}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        if results_df.empty:
            logger.error("❌ No valid backtest results generated!")
            return results_df
        
        # Save results
        output_path = PROCESSED_DATA_DIR / "backtesting" / f"lstm_dcf_{self.lookback_years}year_backtest.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"\n✅ Backtest complete! Results saved to:")
        logger.info(f"   {output_path}")
        
        return results_df
    
    def get_spy_return_for_period(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[float]:
        """
        Get SPY return for the EXACT same period as a stock prediction.
        
        This ensures a fair apples-to-apples comparison.
        
        Args:
            start_date: Entry date
            end_date: Exit date (start_date + lookback_years)
            
        Returns:
            SPY return as decimal (e.g., 0.50 = 50% return), or None if unavailable
        """
        try:
            # Ensure dates are timezone-naive
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            spy = yf.Ticker("SPY")
            
            # Get SPY data with 7-day buffer on each end
            hist = spy.history(
                start=start_date - timedelta(days=7),
                end=end_date + timedelta(days=7)
            )
            
            if hist.empty:
                return None
            
            # Make index timezone-naive
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            
            # Find closest dates to start and end
            start_diffs = np.abs((hist.index - start_date).total_seconds())
            end_diffs = np.abs((hist.index - end_date).total_seconds())
            
            start_idx = start_diffs.argmin()
            end_idx = end_diffs.argmin()
            
            start_price = float(hist['Close'].iloc[start_idx])
            end_price = float(hist['Close'].iloc[end_idx])
            
            return (end_price - start_price) / start_price
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch SPY return: {e}")
            return None
    
    def add_spy_benchmark_to_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add SPY benchmark returns to each prediction for fair comparison.
        
        Uses the EXACT same entry/exit dates as each stock prediction.
        Handles rate limiting by pre-fetching SPY data for the entire period.
        """
        if results_df.empty:
            return results_df
        
        logger.info("\n📊 Adding SPY benchmark returns (matching holding periods)...")
        
        # Pre-fetch all SPY data at once to avoid rate limiting
        min_date = results_df['prediction_date'].min() - timedelta(days=30)
        max_date = results_df['prediction_date'].max() + timedelta(days=self.lookback_years * 365 + 30)
        
        logger.info(f"  Pre-fetching SPY data from {min_date.date()} to {max_date.date()}...")
        
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=min_date, end=max_date)
            
            if spy_hist.empty:
                logger.warning("⚠️  Could not fetch SPY data - skipping benchmark")
                results_df['spy_return'] = None
                results_df['excess_return'] = None
                results_df['beat_spy'] = None
                return results_df
            
            # Make index timezone-naive
            if spy_hist.index.tz is not None:
                spy_hist.index = spy_hist.index.tz_localize(None)
            
            logger.info(f"  ✅ SPY data loaded: {len(spy_hist)} trading days")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch SPY data: {e}")
            results_df['spy_return'] = None
            results_df['excess_return'] = None
            results_df['beat_spy'] = None
            return results_df
        
        # Calculate SPY return for each prediction using pre-fetched data
        spy_returns = []
        
        for idx, row in results_df.iterrows():
            start_date = row['prediction_date']
            end_date = start_date + timedelta(days=self.lookback_years * 365)
            
            # Ensure timezone-naive
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            try:
                # Find closest dates in pre-fetched data
                start_diffs = np.abs((spy_hist.index - start_date).total_seconds())
                end_diffs = np.abs((spy_hist.index - end_date).total_seconds())
                
                start_idx = start_diffs.argmin()
                end_idx = end_diffs.argmin()
                
                start_price = float(spy_hist['Close'].iloc[start_idx])
                end_price = float(spy_hist['Close'].iloc[end_idx])
                
                spy_return = (end_price - start_price) / start_price
                spy_returns.append(spy_return)
            except Exception as e:
                spy_returns.append(None)
            
            if idx % 100 == 0:
                logger.info(f"  Progress: {idx}/{len(results_df)} SPY returns calculated")
        
        results_df['spy_return'] = spy_returns
        
        # Calculate excess return (alpha) - handle None values
        results_df['excess_return'] = results_df.apply(
            lambda r: (r['actual_return'] - r['spy_return']) 
                      if r['spy_return'] is not None and r['actual_return'] is not None 
                      else None, axis=1
        )
        results_df['beat_spy'] = results_df.apply(
            lambda r: r['actual_return'] > r['spy_return'] 
                      if r['spy_return'] is not None and r['actual_return'] is not None 
                      else None, axis=1
        )
        
        logger.info(f"✅ SPY benchmark added to {len(results_df)} predictions")
        
        return results_df

    def analyze_results(self, results_df: pd.DataFrame):
        """
        Analyze and print backtest performance metrics
        
        Args:
            results_df: DataFrame from run_backtest()
        """
        if results_df.empty:
            logger.error("No results to analyze!")
            return
        
        logger.info("\n" + "="*80)
        logger.info("📈 BACKTEST PERFORMANCE ANALYSIS")
        logger.info("="*80)
        
        # Overall metrics
        total_predictions = len(results_df)
        unique_tickers = results_df['ticker'].nunique()
        
        logger.info(f"\n📊 Dataset:")
        logger.info(f"  Total Predictions: {total_predictions:,}")
        logger.info(f"  Unique Tickers: {unique_tickers}")
        logger.info(f"  Time Span: {results_df['prediction_date'].min().date()} to "
                   f"{results_df['prediction_date'].max().date()}")
        
        # Direction accuracy
        direction_correct = results_df['direction_correct'].sum()
        direction_accuracy = direction_correct / total_predictions
        
        logger.info(f"\n🎯 Prediction Accuracy:")
        logger.info(f"  Direction Accuracy: {direction_accuracy:.1%} ({direction_correct}/{total_predictions})")
        
        # Buy signal performance
        buy_signals = results_df[results_df['should_buy']]
        if len(buy_signals) > 0:
            buy_wins = (buy_signals['actual_return'] > 0).sum()
            buy_win_rate = buy_wins / len(buy_signals)
            avg_buy_return = buy_signals['actual_return'].mean()
            
            logger.info(f"\n💰 Investment Performance (Buy Signals Only):")
            logger.info(f"  Total Buy Signals: {len(buy_signals)} ({len(buy_signals)/total_predictions:.1%})")
            logger.info(f"  Win Rate: {buy_win_rate:.1%} ({buy_wins}/{len(buy_signals)})")
            logger.info(f"  Average Return: {avg_buy_return:.1%}")
            logger.info(f"  Median Return: {buy_signals['actual_return'].median():.1%}")
            logger.info(f"  Best Return: {buy_signals['actual_return'].max():.1%} "
                       f"({buy_signals.loc[buy_signals['actual_return'].idxmax(), 'ticker']})")
            logger.info(f"  Worst Return: {buy_signals['actual_return'].min():.1%} "
                       f"({buy_signals.loc[buy_signals['actual_return'].idxmin(), 'ticker']})")
        
        # SPY BENCHMARK COMPARISON (if available)
        if 'spy_return' in results_df.columns and results_df['spy_return'].notna().any():
            valid_spy = results_df[results_df['spy_return'].notna()].copy()
            
            logger.info(f"\n📊 SPY BENCHMARK COMPARISON (Matched Holding Periods):")
            logger.info(f"  Predictions with SPY data: {len(valid_spy)}/{total_predictions}")
            
            avg_spy_return = valid_spy['spy_return'].mean()
            avg_strategy_return = valid_spy['actual_return'].mean()
            beat_spy_count = valid_spy['beat_spy'].sum() if valid_spy['beat_spy'].notna().any() else 0
            beat_spy_rate = beat_spy_count / len(valid_spy) if len(valid_spy) > 0 else 0
            
            logger.info(f"  Average Strategy Return: {avg_strategy_return:.1%}")
            logger.info(f"  Average SPY Return (same periods): {avg_spy_return:.1%}")
            logger.info(f"  Alpha (Excess Return): {avg_strategy_return - avg_spy_return:.1%}")
            logger.info(f"  Beat SPY Rate: {beat_spy_rate:.1%} ({int(beat_spy_count)}/{len(valid_spy)})")
            
            # By year cohort analysis
            valid_spy['year'] = valid_spy['prediction_date'].dt.year
            yearly_comparison = valid_spy.groupby('year').agg({
                'actual_return': 'mean',
                'spy_return': 'mean',
                'beat_spy': 'mean'
            }).round(3)
            
            logger.info(f"\n📅 Cohort Analysis (by entry year):")
            for year, row in yearly_comparison.iterrows():
                beat_marker = "✅" if row['actual_return'] > row['spy_return'] else "❌"
                logger.info(f"  {year}: Strategy {row['actual_return']*100:.1f}% vs SPY {row['spy_return']*100:.1f}% {beat_marker}")
        
        # Portfolio simulation (follow all buy signals)
        total_investment_return = results_df['investment_return'].mean()
        logger.info(f"\n📊 Portfolio Simulation:")
        logger.info(f"  Strategy: Buy on MoS > {self.margin_of_safety:.0%}, Hold otherwise")
        logger.info(f"  Average Return per Opportunity: {total_investment_return:.1%}")
        logger.info(f"  (vs Buy-and-Hold everything: {results_df['actual_return'].mean():.1%})")
        
        # Prediction error
        prediction_error = (results_df['predicted_return'] - results_df['actual_return']).abs().mean()
        logger.info(f"\n📉 Prediction Error:")
        logger.info(f"  Mean Absolute Error: {prediction_error:.1%}")
        
        # Risk-Adjusted Metrics
        logger.info(f"\n📊 RISK-ADJUSTED METRICS:")
        
        # Calculate annualized metrics for buy signals
        if len(buy_signals) > 0:
            returns = buy_signals['actual_return'].values
            
            # Annualize returns (assuming lookback_years holding period)
            annualized_returns = (1 + returns) ** (1 / self.lookback_years) - 1
            
            # Risk-free rate (approximate)
            risk_free_rate = 0.02  # 2% annual
            
            # Sharpe Ratio = (Avg Return - Risk Free Rate) / Std Dev
            avg_annual_return = annualized_returns.mean()
            std_annual_return = annualized_returns.std()
            sharpe_ratio = (avg_annual_return - risk_free_rate) / std_annual_return if std_annual_return > 0 else 0
            
            # Sortino Ratio = (Avg Return - Risk Free Rate) / Downside Std Dev
            negative_returns = annualized_returns[annualized_returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.01
            sortino_ratio = (avg_annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # Max Drawdown (simplified - worst single-period return)
            max_drawdown = returns.min()
            
            # Win/Loss Ratio
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else avg_win / 0.01
            
            # Calmar Ratio = Annualized Return / Max Drawdown
            calmar_ratio = avg_annual_return / abs(max_drawdown) if max_drawdown < 0 else avg_annual_return / 0.01
            
            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f} (>1.0 is good, >2.0 is excellent)")
            logger.info(f"  Sortino Ratio: {sortino_ratio:.2f} (higher = better downside protection)")
            logger.info(f"  Max Drawdown: {max_drawdown:.1%}")
            logger.info(f"  Win/Loss Ratio: {win_loss_ratio:.2f}x")
            logger.info(f"  Calmar Ratio: {calmar_ratio:.2f}")
            logger.info(f"  Annualized Return: {avg_annual_return:.1%}")
            logger.info(f"  Return Volatility: {std_annual_return:.1%}")
        
        # Growth rate statistics
        logger.info(f"\n🌱 LSTM Growth Rate Predictions:")
        logger.info(f"  Revenue Growth: {results_df['lstm_revenue_growth'].mean():.1%} avg, "
                   f"{results_df['lstm_revenue_growth'].median():.1%} median")
        logger.info(f"  FCF Growth: {results_df['lstm_fcf_growth'].mean():.1%} avg, "
                   f"{results_df['lstm_fcf_growth'].median():.1%} median")
        
        # Valuation statistics
        avg_mos = results_df['margin_of_safety'].mean()
        logger.info(f"\n💎 Valuation Metrics:")
        logger.info(f"  Average Margin of Safety: {avg_mos:.1%}")
        logger.info(f"  Median Margin of Safety: {results_df['margin_of_safety'].median():.1%}")
        
        # Top 10 best predictions
        logger.info(f"\n🏆 Top 10 Best Predictions:")
        top_10 = buy_signals.nlargest(10, 'actual_return')[
            ['ticker', 'prediction_date', 'current_price', 'future_price', 
             'predicted_return', 'actual_return']
        ]
        
        for idx, row in top_10.iterrows():
            logger.info(f"  {row['ticker']:5s} @ {row['prediction_date'].date()}: "
                       f"${row['current_price']:.2f} → ${row['future_price']:.2f} "
                       f"(Predicted: {row['predicted_return']:+.1%}, Actual: {row['actual_return']:+.1%})")
        
        logger.info("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Backtest LSTM-DCF predictions against 10-year price trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to backtest (default: all available)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Lookback period in years (default: 10)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with 3 stocks only'
    )
    
    parser.add_argument(
        '--projection-years',
        type=int,
        default=5,
        help='DCF projection horizon (default: 5)'
    )
    
    parser.add_argument(
        '--discount-rate',
        type=float,
        default=0.10,
        help='WACC for DCF (default: 0.10 = 10%%)'
    )
    
    parser.add_argument(
        '--mos-threshold',
        type=float,
        default=0.20,
        help='Margin of safety threshold for buy signal (default: 0.20 = 20%%)'
    )
    
    parser.add_argument(
        '--skip-spy',
        action='store_true',
        help='Skip SPY benchmark comparison (faster, but no alpha calculation)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model path (default: models/lstm_dcf_enhanced.pth). Use "v2" for lstm_dcf_v2.pth'
    )
    
    args = parser.parse_args()
    
    # Handle model path shortcuts
    if args.model == 'v2':
        args.model = str(MODELS_DIR / "lstm_dcf_v2.pth")
    elif args.model is None:
        args.model = str(MODELS_DIR / "lstm_dcf_enhanced.pth")
    
    # Quick mode: Test only 3 stocks
    if args.quick:
        if args.tickers is None:
            args.tickers = ['AAPL', 'MSFT', 'JNJ']
        else:
            args.tickers = args.tickers[:3]
        logger.info("🚀 Quick mode: Testing 3 stocks only")
    
    # Initialize backtester
    backtester = LSTMDCFBacktester(
        model_path=args.model,
        lookback_years=args.years,
        projection_years=args.projection_years,
        discount_rate=args.discount_rate,
        margin_of_safety=args.mos_threshold
    )
    
    # Run backtest
    results_df = backtester.run_backtest(
        tickers=args.tickers,
        max_tickers=3 if args.quick else None
    )
    
    # Add SPY benchmark if requested
    if not results_df.empty and not args.skip_spy:
        results_df = backtester.add_spy_benchmark_to_results(results_df)
        
        # Save updated results with SPY data
        output_path = PROCESSED_DATA_DIR / "backtesting" / f"lstm_dcf_{args.years}year_backtest.csv"
        results_df.to_csv(output_path, index=False)
    
    # Analyze results
    if not results_df.empty:
        backtester.analyze_results(results_df)
    else:
        logger.error("❌ Backtest failed - no results generated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
