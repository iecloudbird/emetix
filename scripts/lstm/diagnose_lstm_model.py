"""
Simple LSTM-DCF Diagnostic Script
Debug what the model is actually predicting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import yfinance as yf

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from config.settings import MODELS_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def load_model():
    """Load trained LSTM model"""
    model_path = MODELS_DIR / "lstm_dcf_final.pth"
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    
    # Infer architecture
    lstm_ih_shape = checkpoint['lstm.weight_ih_l0'].shape
    hidden_size = lstm_ih_shape[0] // 4
    input_size = lstm_ih_shape[1]
    num_layers = sum(1 for key in checkpoint.keys() if 'lstm.weight_ih_l' in key)
    output_size = checkpoint['fc.weight'].shape[0]
    
    model = LSTMDCFModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
        output_size=output_size
    )
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Model Architecture:")
    print(f"   Input: {input_size} features")
    print(f"   Hidden: {hidden_size}")
    print(f"   Layers: {num_layers}")
    print(f"   Output: {output_size} (growth rate)")
    
    return model, input_size


def test_prediction(ticker='AAPL'):
    """Test a single prediction"""
    print(f"\n{'='*80}")
    print(f"Testing {ticker}")
    print(f"{'='*80}")
    
    model, input_size = load_model()
    
    # Load financial data
    income_path = RAW_DATA_DIR / "financial_statements" / f"{ticker}_income.csv"
    
    if not income_path.exists():
        print(f"❌ No data for {ticker}")
        return
    
    income = pd.read_csv(income_path)
    print(f"\n📊 Data Shape: {income.shape}")
    print(f"   Columns: {list(income.columns[:10])}")
    
    # Show sample of actual data
    if 'totalRevenue' in income.columns:
        print(f"\n💰 Recent Revenues (billions):")
        revenues = income['totalRevenue'].head(5).astype(float) / 1e9
        for idx, rev in enumerate(revenues):
            print(f"   Q{idx+1}: ${rev:.2f}B")
    
    # Create dummy input matching the model
    print(f"\n🧪 Creating test input with {input_size} features:")
    
    # Create random normalized sequence
    dummy_sequence = torch.randn(1, 60, input_size)
    
    # Predict
    with torch.no_grad():
        prediction = model(dummy_sequence)
        growth_rate = prediction[0].item()
    
    print(f"\n📈 Model Prediction:")
    print(f"   Quarterly Growth: {growth_rate:.2f}%")
    print(f"   Annual Growth: {((1 + growth_rate/100)**4 - 1)*100:.2f}%")
    
    # Test with actual data pattern
    if 'totalRevenue' in income.columns and len(income) >= 60:
        print(f"\n🔍 Testing with ACTUAL data pattern:")
        
        # Get actual revenues
        revenues = income['totalRevenue'].head(60).astype(float).values
        
        # Calculate actual growth
        actual_growth = ((revenues[-1] / revenues[0]) ** (1/15) - 1) * 100  # 15 years = 60 quarters
        print(f"   Actual 15Y CAGR: {actual_growth:.2f}%")
        
        # Simple pattern: just revenues repeated
        revenue_sequence = revenues.reshape(60, 1).repeat(input_size, axis=1)
        revenue_sequence = (revenue_sequence - revenue_sequence.mean()) / (revenue_sequence.std() + 1e-8)
        revenue_tensor = torch.tensor(revenue_sequence, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_from_actual = model(revenue_tensor)
            growth_from_actual = pred_from_actual[0].item()
        
        print(f"   Model Prediction: {growth_from_actual:.2f}% QoQ → {((1 + growth_from_actual/100)**4 - 1)*100:.2f}% Annual")
    
    # Get current price
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice', 0)
        print(f"\n💵 Current Price: ${current_price:.2f}")
    except:
        pass
    
    print(f"\n{'='*80}\n")


def main():
    """Run diagnostics"""
    print("\n" + "="*80)
    print("🔬 LSTM-DCF MODEL DIAGNOSTIC")
    print("="*80)
    
    # Test on a few stocks
    for ticker in ['AAPL', 'MSFT', 'AMZN']:
        test_prediction(ticker)
    
    print("="*80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
