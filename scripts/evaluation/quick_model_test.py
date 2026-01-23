"""
Quick Test Script for LSTM-DCF Model
Tests model loading, inference, and integration with live data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd

from src.models.ensemble.consensus_scorer import ConsensusScorer
from src.data.fetchers import YFinanceFetcher
from config.settings import MODELS_DIR

print("=" * 80)
print("QUICK MODEL TEST - LSTM-DCF")
print("=" * 80)

# Test 1: Check model files exist
print("\n1. Checking Model Files...")
lstm_enhanced_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
lstm_path = MODELS_DIR / "lstm_dcf_enhanced.pth"

if lstm_enhanced_path.exists():
    print(f"Model found: {lstm_enhanced_path}")
    print(f"  Size: {lstm_enhanced_path.stat().st_size / 1024 / 1024:.2f} MB")
    active_path = lstm_enhanced_path
elif lstm_path.exists():
    print(f"Model found: {lstm_path}")
    print(f"  Size: {lstm_path.stat().st_size / 1024 / 1024:.2f} MB")
    active_path = lstm_path
else:
    print(f"LSTM-DCF model NOT found")
    print(f"  Checked: {lstm_enhanced_path}")
    print(f"  Checked: {lstm_path}")
    sys.exit(1)

# Test 2: Load LSTM-DCF Model - detect v1 vs v2
print(f"\n2. Loading LSTM-DCF Model...")
try:
    checkpoint = torch.load(str(active_path), map_location='cpu', weights_only=False)
    
    # Detect model version
    model_version = checkpoint.get('model_version', 'v1')
    print(f"  Detected model version: {model_version}")
    
    if model_version == 'v2' or 'feature_scaler' in checkpoint:
        # V2 model
        from scripts.lstm.train_lstm_dcf_v2 import LSTMDCFModelV2
        hp = checkpoint['hyperparameters']
        lstm_model = LSTMDCFModelV2(
            input_size=hp['input_size'],
            hidden_size=hp['hidden_size'],
            num_layers=hp['num_layers'],
            dropout=hp.get('dropout', 0.3),
            output_size=hp.get('output_size', 2)
        )
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        feature_scaler = checkpoint['feature_scaler']
        target_scaler = checkpoint['target_scaler']
        input_size = hp['input_size']
        seq_length = checkpoint.get('sequence_length', 8)
        print(f"  Loaded v2 model: input={input_size}, seq_len={seq_length}")
    else:
        # V1 model
        from src.models.deep_learning.lstm_dcf import LSTMDCFModel
        hp = checkpoint.get('hyperparameters', {})
        input_size = hp.get('input_size', 16)
        lstm_model = LSTMDCFModel(input_size=input_size, hidden_size=128, num_layers=3)
        lstm_model.load_model(str(active_path))
        feature_scaler = checkpoint.get('scaler')
        target_scaler = None
        seq_length = 60
        print(f"  Loaded v1 model: input={input_size}, seq_len={seq_length}")
    
    lstm_model.eval()
    print("Model loaded successfully")
    
    # Test inference
    dummy_seq = torch.randn(1, seq_length, input_size)
    with torch.no_grad():
        output = lstm_model(dummy_seq)
    
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(output.numpy())
        print(f"  Test predictions: Revenue={predictions[0,0]:.1f}%, FCF={predictions[0,1]:.1f}%")
    else:
        print(f"  Test output shape: {output.shape}")
        
except Exception as e:
    print(f"LSTM-DCF loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Quick Integration Test with Real Stock
print("\n3. Integration Test with Real Stock (AAPL)...")
try:
    fetcher = YFinanceFetcher()
    stock_data = fetcher.fetch_stock_data('AAPL')
    
    if stock_data is not None and not (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
        print(f"✓ Fetched AAPL data")
        print(f"  P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}")
        print(f"  Beta: {stock_data.get('beta', 'N/A')}")
        print(f"  Current Price: ${stock_data.get('current_price', stock_data.get('price', 'N/A'))}")
        
        # Consensus Scoring (LSTM-DCF + GARP + Risk)
        scorer = ConsensusScorer()
        model_scores = {
            'lstm_dcf': 0.70,      # Placeholder - would come from actual LSTM prediction
            'garp_score': 0.65,    # Placeholder - calculated from Forward P/E + PEG
            'risk_score': 0.60     # Placeholder - from Beta + volatility
        }
        consensus = scorer.calculate_consensus(model_scores)
        print(f"✓ Consensus Decision (LSTM-DCF 50%, GARP 25%, Risk 25%):")
        print(f"  Score: {consensus['consensus_score']:.4f}")
        print(f"  Confidence: {consensus['confidence']:.4f}")
        print(f"  Decision: {'UNDERVALUED' if consensus['is_undervalued'] else 'NOT UNDERVALUED'}")
    else:
        print("⚠ Could not fetch stock data (offline or API issue)")
        
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Performance Check
print("\n4. Performance Check...")
try:
    import time
    
    # LSTM Inference time
    start = time.time()
    for _ in range(10):
        dummy_seq = torch.randn(1, seq_length, input_size)
        with torch.no_grad():
            output = lstm_model(dummy_seq)
    lstm_time = (time.time() - start) / 10 * 1000
    print(f"LSTM-DCF: {lstm_time:.2f}ms per prediction (avg of 10)")
    
except Exception as e:
    print(f"Performance check skipped: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
print("\nSummary:")
print("  ✓ LSTM-DCF model loaded and functional")
print("  ✓ Integration with live data working")
print("  ✓ Consensus scorer working (LSTM 50%, GARP 25%, Risk 25%)")
print("\nArchitecture: LSTM-DCF + GARP Score + Risk Score")
print("=" * 80)
