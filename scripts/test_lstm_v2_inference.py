"""
Quick test for LSTM-DCF v2 inference
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.processors.lstm_v2_processor import LSTMV2Processor
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from config.settings import MODELS_DIR
import yfinance as yf

def test_v2_inference(ticker: str = 'AAPL'):
    """Test v2 inference pipeline"""
    print(f"\n{'='*60}")
    print(f"Testing LSTM-DCF v2 Inference for {ticker}")
    print('='*60)
    
    # Load model
    model, metadata = LSTMDCFModel.from_checkpoint(str(MODELS_DIR / 'lstm_dcf_enhanced.pth'))
    print(f"\n✓ Model version: {metadata.get('model_version')}")
    print(f"✓ Sequence length: {metadata.get('sequence_length')}")
    print(f"✓ Feature columns: {len(metadata.get('feature_cols', []))} features")
    
    # Create processor
    processor = LSTMV2Processor(sequence_length=metadata.get('sequence_length', 8))
    
    # Get current price
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    print(f"✓ Current price: ${current_price:.2f}")
    
    # Prepare inference
    result = processor.prepare_inference_sequence(
        ticker, 
        feature_scaler=metadata.get('feature_scaler')
    )
    
    if result is None:
        print(f"\n✗ Failed to prepare inference sequence for {ticker}")
        return None
    
    tensor, meta = result
    print(f"\n✓ Input tensor shape: {tensor.shape}")
    print(f"✓ Quarters analyzed: {meta['quarters_used']}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model(tensor)
    
    print(f"✓ Prediction shape: {prediction.shape}")
    print(f"✓ Raw prediction: {prediction.numpy().flatten()}")
    
    # Interpret prediction
    interpretation = processor.interpret_prediction(
        prediction,
        target_scaler=metadata.get('target_scaler'),
        current_price=current_price
    )
    
    print(f"\n{'='*60}")
    print("LSTM-DCF v2 Results:")
    print('='*60)
    print(f"Revenue Growth Forecast: {interpretation['revenue_growth_forecast']:+.1f}%")
    print(f"FCF Growth Forecast: {interpretation['fcf_growth_forecast']:+.1f}%")
    print(f"Growth Multiple: {interpretation['growth_multiple']:.3f}")
    print(f"Implied Fair Value: ${interpretation['implied_fair_value']:.2f}")
    print(f"Upside: {interpretation['upside_percent']:+.1f}%")
    
    return interpretation


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str, nargs="?", default="AAPL")
    args = parser.parse_args()
    
    test_v2_inference(args.ticker)
