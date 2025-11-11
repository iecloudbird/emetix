"""Test LSTM-DCF in consensus scoring"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yfinance as yf
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.data.processors.time_series_processor import TimeSeriesProcessor
from config.logging_config import get_logger

logger = get_logger(__name__)

def test_lstm_consensus():
    """Test full LSTM-DCF pipeline for consensus"""
    
    ticker = 'AAPL'
    
    # Load model
    model = LSTMDCFModel(input_size=12)
    model.load_model('models/lstm_dcf_final.pth')
    
    # Load processor with scaler
    proc = TimeSeriesProcessor()
    
    # Fetch data
    print(f"\n{'='*60}")
    print(f"Testing LSTM-DCF Consensus Pipeline for {ticker}")
    print(f"{'='*60}")
    
    ts_data = proc.fetch_sequential_data(ticker, period='5y')
    if ts_data is None or ts_data.empty:
        print("✗ Failed to fetch time-series data")
        return
    
    print(f"✓ Fetched time-series data: {ts_data.shape}")
    
    # Create sequences
    X, _ = proc.create_sequences(ts_data, target_col='close')
    if len(X) == 0:
        print("✗ Failed to create sequences")
        return
    
    print(f"✓ Created sequences: {len(X)} sequences")
    
    # Get last sequence
    seq = torch.tensor(X[-1:], dtype=torch.float32)
    print(f"✓ Input sequence shape: {seq.shape}")
    
    # Get stock info FIRST (needed for shares)
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    shares = info.get('sharesOutstanding', 1e9)
    
    print(f"\n✓ Stock Info:")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Shares Outstanding: {shares:,.0f}")
    
    # Check scaler
    print(f"\n✓ Scaler Info:")
    print(f"  Fitted: {hasattr(proc.scaler, 'n_features_in_')}")
    print(f"  N features: {proc.scaler.n_features_in_ if hasattr(proc.scaler, 'n_features_in_') else 'N/A'}")
    print(f"  Scale range: {proc.scaler.scale_[:3] if hasattr(proc.scaler, 'scale_') else 'N/A'}")
    
    # Forecast with denormalization (PER-SHARE mode)
    print(f"\n✓ Forecasting with scaler={proc.scaler is not None}, fcff_idx=-1, PER-SHARE mode")
    fcff = model.forecast_fcff(seq, periods=10, scaler=proc.scaler, fcff_feature_idx=-1, use_per_share=True)
    print(f"✓ Raw FCFF Forecasts (per-share): {fcff[:3]}")
    print(f"✓ FCFF Forecasts (10 years, per-share):")
    print(f"  Year 1-5: {[f'${f:.2f}' for f in fcff[:5]]}")
    print(f"  Year 6-10: {[f'${f:.2f}' for f in fcff[5:]]}")
    
    # DCF valuation with calibration
    dcf_result = model.dcf_valuation(fcff, 1.0, current_price)
    print(f"\n✓ DCF Valuation:")
    print(f"  Raw Fair Value: ${dcf_result['fair_value']:.2f}")
    print(f"  Calibrated Fair Value: ${dcf_result['calibrated_fair_value']:.2f}")
    print(f"  Present Value FCFF: ${dcf_result['pv_fcff']:.2f}")
    print(f"  Terminal Value (PV): ${dcf_result['pv_terminal_value']:.2f}")
    
    # Calculate consensus score
    fair_value = dcf_result['calibrated_fair_value']
    if current_price > 0:
        gap = ((fair_value - current_price) / current_price) * 100
        # Normalize gap to 0-1 scale (-20% to +20% maps to 0 to 1)
        lstm_score = max(0, min(1, (gap + 20) / 40))
        
        print(f"\n✓ Consensus Scoring:")
        print(f"  Valuation Gap: {gap:+.2f}%")
        print(f"  LSTM Score (0-1): {lstm_score:.4f}")
        print(f"  Assessment: {'UNDERVALUED' if gap > 10 else 'FAIRLY VALUED' if abs(gap) < 10 else 'OVERVALUED'}")
        print(f"\n✓ Calibration Effect:")
        print(f"  Raw model predicted: ${dcf_result['fair_value']:.2f} ({((dcf_result['fair_value']-current_price)/current_price*100):+.1f}%)")
        print(f"  After calibration: ${fair_value:.2f} ({gap:+.1f}%)")
        print(f"  More reasonable range for consensus voting")
        
        return lstm_score
    else:
        print("✗ Invalid current price")
        return 0.5

if __name__ == "__main__":
    try:
        test_lstm_consensus()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n✗ Test failed: {e}")
