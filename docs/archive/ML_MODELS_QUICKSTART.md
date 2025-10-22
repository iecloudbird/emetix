# LSTM-DCF & RF Ensemble - Quick Start Guide

## Installation & Setup

### Step 1: Install Dependencies

First, install the new ML/DL packages:

```powershell
# Install PyTorch (CPU version for Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other ML packages
pip install pytorch-lightning scikit-learn scikit-optimize shap joblib statsmodels

# Install updated requirements
pip install -r requirements.txt
```

**Note**: If you have a GPU and CUDA installed, use the GPU version of PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Verify Installation

Test that the models are working:

```powershell
python scripts/test_ml_models.py
```

Expected output: All tests should PASS (some may skip if offline).

## Training Workflow

### Step 1: Fetch Training Data

Fetch historical time-series data for LSTM training:

```powershell
python scripts/fetch_historical_data.py
```

This will:

- Fetch fundamental data for 50 S&P 500 stocks
- Fetch 15-year historical prices for time-series analysis
- Create sequences for LSTM training
- Save data to `data/processed/training/lstm_training_data.csv`

**Expected time**: ~10-15 minutes (depends on network speed)

### Step 2: Train LSTM-DCF Model

Train the LSTM-DCF hybrid model:

```powershell
python scripts/train_lstm_dcf.py
```

This will:

- Load time-series sequences
- Train LSTM with early stopping
- Calculate DCF valuations
- Save model to `models/lstm_dcf_final.pth`

**Expected time**: ~30-60 minutes on CPU (5-10 minutes on GPU)

**Configuration**: Edit `config/model_config.yaml` to adjust:

- `lstm_dcf.training.max_epochs` (default: 100)
- `lstm_dcf.architecture.hidden_size` (default: 128)
- `lstm_dcf.dcf_params.wacc` (default: 0.08)

### Step 3: Train Random Forest Ensemble

Train the RF ensemble model:

```powershell
python scripts/train_rf_ensemble.py
```

This will:

- Fetch current stock data for 20 tickers
- Train RF regressor and classifier
- Save model to `models/rf_ensemble.pkl`
- Save feature importance to `models/rf_feature_importance.csv`

**Expected time**: ~2-5 minutes

## Using the Models

### Example: Stock Valuation with LSTM-DCF

```python
import torch
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.data.processors.time_series_processor import TimeSeriesProcessor
from config.settings import MODELS_DIR

# Load trained model
model = LSTMDCFModel(input_size=10)
model.load_model(str(MODELS_DIR / "lstm_dcf_final.pth"))

# Fetch time-series data
processor = TimeSeriesProcessor()
ts_data = processor.fetch_sequential_data('AAPL', period='5y')

# Create sequence
X, _ = processor.create_sequences(ts_data)
last_seq = torch.tensor(X[-1:], dtype=torch.float32)

# Predict fair value
import yfinance as yf
stock = yf.Ticker('AAPL')
current_price = stock.info['currentPrice']
shares = stock.info['sharesOutstanding']

result = model.predict_stock_value(last_seq, current_price, shares)

print(f"Current Price: ${result['current_price']:.2f}")
print(f"Fair Value: ${result['fair_value']:.2f}")
print(f"Gap: {result['fair_value_gap']:+.2f}%")
print(f"Undervalued: {result['is_undervalued']}")
```

### Example: RF Ensemble Prediction

```python
from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.data.fetchers import YFinanceFetcher
from config.settings import MODELS_DIR

# Load trained model
model = RFEnsembleModel()
model.load(str(MODELS_DIR / "rf_ensemble.pkl"))

# Fetch stock data
fetcher = YFinanceFetcher()
stock_data = fetcher.fetch_stock_data('MSFT')

# Prepare features and predict
X = model.prepare_features(stock_data)
result = model.predict_score(X)

print(f"Ensemble Score: {result['ensemble_score']:.4f}")
print(f"Is Undervalued: {result['is_undervalued']}")
```

### Example: Consensus Scoring

```python
from src.models.ensemble.consensus_scorer import ConsensusScorer

# Initialize scorer
scorer = ConsensusScorer()

# Combine scores from all models
model_scores = {
    'lstm_dcf': 0.75,        # From LSTM-DCF gap analysis
    'rf_ensemble': 0.68,     # From RF ensemble
    'linear_valuation': 0.72, # From existing linear model
    'risk_classifier': 0.65  # From existing risk model (inverted)
}

# Calculate consensus
result = scorer.calculate_consensus(model_scores)

print(f"Consensus Score: {result['consensus_score']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Decision: {'UNDERVALUED' if result['is_undervalued'] else 'NOT UNDERVALUED'}")

# Get detailed breakdown
breakdown = scorer.get_model_breakdown(model_scores)
for model, details in breakdown.items():
    print(f"{model}: {details['raw_score']:.4f} × {details['weight']:.2f} = {details['weighted_contribution']:.4f}")
```

## Integration with Existing Agents

The models are designed to integrate with your existing `ValuationAgent`. See implementation plan section "Phase 5: Agent Integration" for details.

Quick example tool integration:

```python
from langchain.tools import tool
from src.models.deep_learning.lstm_dcf import LSTMDCFModel

@tool
def lstm_dcf_valuation(ticker: str) -> str:
    """Perform LSTM-DCF valuation"""
    # Load model and predict
    # Return formatted string for LangChain
    pass
```

## Troubleshooting

### Common Issues

**1. PyTorch Import Error**

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Install PyTorch with the correct command for your system (see Step 1).

**2. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in `config/model_config.yaml`:

```yaml
lstm_dcf:
  training:
    batch_size: 16 # Reduce from 32
```

**3. No Training Data**

```
Training data not found: data/processed/training/lstm_training_data.csv
```

**Solution**: Run `python scripts/fetch_historical_data.py` first.

**4. API Rate Limiting**

```
Too many requests
```

**Solution**: The scripts include rate limiting. Wait a few minutes and retry.

## Model Files

After training, you should have:

```
models/
  ├── lstm_dcf_final.pth           # LSTM-DCF model weights
  ├── lstm_checkpoints/            # Training checkpoints
  │   └── lstm_dcf_epoch_XX.pth
  ├── rf_ensemble.pkl              # RF ensemble model
  └── rf_feature_importance.csv    # Feature rankings
```

## Configuration

All model hyperparameters are in `config/model_config.yaml`:

```yaml
lstm_dcf:
  architecture:
    hidden_size: 128 # LSTM hidden dimension
    num_layers: 3 # Number of LSTM layers
    dropout: 0.2 # Dropout rate

  training:
    batch_size: 32
    learning_rate: 0.001
    max_epochs: 100

  dcf_params:
    wacc: 0.08 # Discount rate
    terminal_growth: 0.03 # Terminal growth rate

rf_ensemble:
  n_estimators: 200 # Number of trees
  max_depth: 15 # Tree depth

consensus_weights:
  lstm_dcf: 0.40 # 40% weight
  rf_ensemble: 0.30 # 30% weight
  linear_valuation: 0.20 # 20% weight
  risk_classifier: 0.10 # 10% weight
```

## Next Steps

1. **Test the models**: Run `python scripts/test_ml_models.py`
2. **Fetch data**: Run `python scripts/fetch_historical_data.py`
3. **Train models**: Run training scripts in sequence
4. **Integrate with agents**: See `LSTM_DCF_RF_IMPLEMENTATION_PLAN.md` Phase 5
5. **Run analysis**: Use the models in your stock analysis workflow

## Performance Benchmarks

Expected performance on a typical laptop (CPU):

- **LSTM Training**: 30-60 minutes for 30 stocks
- **RF Training**: 2-5 minutes for 20 stocks
- **Inference**: <500ms per stock (cold start), <100ms (warm)
- **Memory**: ~4GB RAM during training, ~1GB during inference

## Support

For issues or questions:

- Check the implementation plan: `LSTM_DCF_RF_IMPLEMENTATION_PLAN.md`
- Review test output: `python scripts/test_ml_models.py`
- Check logs in `logs/` directory
