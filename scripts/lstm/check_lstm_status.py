"""
Quick check of LSTM Growth Forecaster training status
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR

print("\n" + "="*80)
print("LSTM GROWTH FORECASTER STATUS CHECK")
print("="*80)

# Check data
data_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "lstm_growth_training_data.csv"
if data_file.exists():
    df = pd.read_csv(data_file)
    print(f"\n✅ Training Data Found:")
    print(f"   File: {data_file.name}")
    print(f"   Records: {len(df):,}")
    print(f"   Stocks: {df['ticker'].nunique()}")
    print(f"   Columns: {list(df.columns)}")
else:
    print(f"\n❌ Training data not found!")

# Check model files
model_file = MODELS_DIR / "lstm_growth_forecaster.pth"
checkpoint_file = MODELS_DIR / "lstm_growth_forecaster_best.pth"

print(f"\n📦 Model Files:")
if model_file.exists():
    size = model_file.stat().st_size / 1024  # KB
    print(f"   ✅ lstm_growth_forecaster.pth ({size:.1f} KB)")
else:
    print(f"   ❌ lstm_growth_forecaster.pth (not found)")

if checkpoint_file.exists():
    size = checkpoint_file.stat().st_size / 1024  # KB
    print(f"   ✅ lstm_growth_forecaster_best.pth ({size:.1f} KB)")
else:
    print(f"   ⚠️  lstm_growth_forecaster_best.pth (not found - this is OK)")

# Check if we can load the model
if model_file.exists():
    print(f"\n🧪 Testing Model Load:")
    try:
        from src.models.deep_learning.lstm_growth_forecaster import LSTMGrowthForecaster
        
        model = LSTMGrowthForecaster(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=4
        )
        model.load_model(str(model_file))
        print(f"   ✅ Model loaded successfully!")
        print(f"   Architecture: {model.num_layers}-layer LSTM, hidden_size={model.hidden_size}")
        
        # Test prediction
        test_input = torch.randn(1, 20, 4)  # batch=1, seq=20, features=4
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        print(f"   ✅ Test prediction successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")

print("\n" + "="*80)
print("STATUS: Ready to use!" if model_file.exists() else "STATUS: Need to train model")
print("="*80)
