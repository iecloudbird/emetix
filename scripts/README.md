# Scripts

CLI tools for stock analysis, data collection, ML training, pipeline execution, and evaluation.

> **Important**: Always activate the virtual environment first: `.\venv\Scripts\Activate.ps1`

---

## Root Scripts

| Script | Purpose | Usage |
| ------ | ------- | ----- |
| `analyze_stock.py` | Main entry point — full or basic stock analysis | `python scripts/analyze_stock.py AAPL` |
| `quick_start_data_collection.py` | Fetch market data from all sources | `python scripts/quick_start_data_collection.py` |
| `backtest_validator.py` | Validate backtest data integrity | `python scripts/backtest_validator.py` |
| `validate_finnhub_format.py` | Validate Finnhub data format | `python scripts/validate_finnhub_format.py` |
| `diagnosis.py` | Diagnostic utilities | `python scripts/diagnosis.py` |
| `test_lstm_v2_inference.py` | Test LSTM V2 model inference | `python scripts/test_lstm_v2_inference.py` |
| `test_phase3_metrics.py` | Test Phase 3 pipeline metrics | `python scripts/test_phase3_metrics.py` |
| `test_risk_profile.py` | Test risk profile assessment | `python scripts/test_risk_profile.py` |

### Stock Analysis Examples

```bash
python scripts/analyze_stock.py AAPL              # Full AI-powered analysis
python scripts/analyze_stock.py AAPL --basic       # No AI agents (faster)
python scripts/analyze_stock.py AAPL MSFT --compare # Side-by-side comparison
```

---

## Pipeline Scripts (`scripts/pipeline/`)

Scripts for running the 3-stage Quality Growth Pipeline.

| Script | Stage | Purpose | Schedule |
| ------ | ----- | ------- | -------- |
| `weekly_attention_scan.py` | 1 | Scan ~5,800 stocks with 5 attention triggers | Weekly |
| `daily_qualified_update_v3.py` | 2 | Apply 5-pillar scoring to attention stocks | Daily |
| `stage3_curate_watchlist.py` | 3 | Classify and curate final watchlist | On-demand |
| `full_universe_scan.py` | All | End-to-end pipeline run | Ad-hoc |
| `run_pipeline_e2e.py` | All | End-to-end pipeline orchestrator | Ad-hoc |
| `populate_universe.py` | Setup | Populate stock universe in MongoDB | One-time |
| `diagnose_stock.py` | Debug | Diagnose pipeline status for a stock | Debug |

### Pipeline Workflow

```bash
# 1. Populate universe (first time only)
python scripts/pipeline/populate_universe.py

# 2. Weekly attention scan
python scripts/pipeline/weekly_attention_scan.py

# 3. Daily qualification
python scripts/pipeline/daily_qualified_update_v3.py

# 4. Curate final watchlist
python scripts/pipeline/stage3_curate_watchlist.py
```

---

## Data Collection Scripts (`scripts/data_collection/`)

| Script | Purpose |
| ------ | ------- |
| `fetch_historical_data.py` | Fetch historical price data |
| `fetch_enhanced_training_data.py` | Fetch enhanced training dataset |
| `fetch_comprehensive_training_data.py` | Comprehensive multi-source data |
| `collect_unified_training_data.py` | Unified training data collection |
| `build_enhanced_training_data.py` | Build enhanced training dataset |

---

## LSTM Training Scripts (`scripts/lstm/`)

| Script | Purpose |
| ------ | ------- |
| `train_lstm_dcf_v2.py` | Train LSTM-DCF V2 model (primary) |
| `train_lstm_dcf_enhanced.py` | Train enhanced LSTM-DCF variant |
| `clean_training_data.py` | Clean and preprocess training data |

### GPU Training

```bash
# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Train (auto-detects GPU)
python scripts/lstm/train_lstm_dcf_v2.py
```

RTX 3050: ~6 min | CPU: ~30–60 min

---

## Evaluation Scripts (`scripts/evaluation/`)

| Script | Purpose |
| ------ | ------- |
| `quick_model_test.py` | Quick model inference test |
| `evaluate_fcf_predictions.py` | Evaluate FCF prediction accuracy |
| `backtest_lstm_dcf_10year.py` | 10-year LSTM-DCF backtest |

---

## Consensus Scripts (`scripts/consensus/`)

| Script | Purpose |
| ------ | ------- |
| `test_reverse_dcf.py` | Test reverse DCF calculations |
