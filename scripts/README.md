# Scripts Directory Organization

This directory contains all production scripts organized by functionality.

## 📁 Directory Structure

```
scripts/
├── 📊 analyze_stock_consensus.py        # MAIN: End-to-end stock analysis
├── 🎯 build_ml_watchlist.py              # MAIN: Screen top opportunities
├── 📝 generate_assessment_report.py      # FYP report generation
├── 🧪 test_agent.py                      # Test risk agent
├── 🧪 test_multiagent_system.py          # Test multi-agent orchestration
├── 🧪 test_news_sentiment.py             # Test news API
├── 🧪 test_alpha_vantage.py              # Test Alpha Vantage API
│
├── 📂 lstm/                               # LSTM-DCF Related
│   ├── train_lstm_dcf_enhanced.py        # Train LSTM model (10-15 min)
│   ├── train_lstm_growth_forecaster.py   # Growth forecaster variant
│   └── check_lstm_status.py              # Check training progress
│
├── 📂 rf/                                 # Random Forest Related (DEPRECATED)
│   └── train_rf_risk_sentiment.py        # RF Risk+Sentiment (archived)
│
├── 📂 consensus/                          # Consensus System
│   └── test_reverse_dcf.py               # ⭐ NEW: Reverse DCF validator
│
├── 📂 data_collection/                    # Data Pipelines
│   ├── fetch_enhanced_training_data.py   # Fetch financial statements
│   ├── fetch_historical_data.py          # Fetch stock prices
│   └── build_enhanced_training_data.py   # Build 8,828-record dataset
│
├── 📂 evaluation/                         # Testing & Validation
│   ├── backtest_consensus_strategy.py    # ⭐ NEW: Backtest with Sharpe
│   ├── quick_model_test.py               # Fast model check
│   ├── validate_enhanced_data.py         # Data quality check
│   ├── inspect_dataset.py                # Dataset explorer
│   ├── evaluate_deep_learning_models.py  # LSTM evaluation
│   └── evaluate_traditional_models.py    # RF evaluation
│
└── 📂 archive/                            # Deprecated Scripts
    ├── fetch_lstm_training_data.py       # ❌ Old price-based
    ├── train_lstm_dcf.py                 # ❌ Old LSTM trainer
    ├── simple_train.py                   # ❌ Non-production
    ├── test_training_data.py             # ❌ Replaced
    ├── retry_failed_tickers.py           # ❌ Integrated
    ├── test_valuation_system.py          # ❌ Replaced by consensus
    ├── test_enhanced_agent.py            # ❌ Replaced by consensus
    └── analyze_stock.py                  # ❌ Non-consensus version
```

---

## 🚀 Most Frequently Used Commands

### **Quick Stock Analysis**

```powershell
# Single stock comprehensive analysis (70/20/10 consensus)
venv\Scripts\python.exe scripts\analyze_stock_consensus.py PYPL

# Multiple stocks comparison
venv\Scripts\python.exe scripts\analyze_stock_consensus.py AAPL MSFT GOOGL --compare

# Screen top 50 opportunities
venv\Scripts\python.exe scripts\analyze_stock_consensus.py --watchlist 50 --min-score 70
```

### **Model Training Pipeline**

```powershell
# 1. Fetch financial data (Alpha Vantage + Yahoo Finance)
venv\Scripts\python.exe scripts\data_collection\fetch_enhanced_training_data.py

# 2. Build training dataset (8,828 records, 117 stocks)
venv\Scripts\python.exe scripts\data_collection\build_enhanced_training_data.py

# 3. Train LSTM-DCF (10-15 mins on GPU)
venv\Scripts\python.exe scripts\lstm\train_lstm_dcf_enhanced.py

# 4. Train RF Risk+Sentiment (3-5 mins)
venv\Scripts\python.exe scripts\rf\train_rf_risk_sentiment.py

# 5. Validate models
venv\Scripts\python.exe scripts\evaluation\quick_model_test.py
```

### **Validation & Testing**

```powershell
# Reverse DCF validation (growth sanity check)
venv\Scripts\python.exe scripts\consensus\test_reverse_dcf.py PYPL

# Data quality check
venv\Scripts\python.exe scripts\evaluation\validate_enhanced_data.py

# Backtest strategy (2015-2025)
venv\Scripts\python.exe scripts\evaluation\backtest_consensus_strategy.py --start 2015-01-01 --quick

# Generate FYP assessment report
venv\Scripts\python.exe scripts\generate_assessment_report.py
```

---

## 📋 Script Details

### 🎯 **Main Analysis Scripts**

#### `analyze_stock_consensus.py` ⭐ PRIMARY

**Purpose**: End-to-end stock analysis with 70/20/10 weighted ensemble

**Features**:

- LSTM-DCF fair value (70%)
- RF Risk + Sentiment (20%)
- P/E Sanity Check (10%)
- Reverse DCF validation
- Margin of safety calculation

**Usage**:

```powershell
# Single stock
python scripts\analyze_stock_consensus.py AAPL

# Batch with comparison
python scripts\analyze_stock_consensus.py AAPL MSFT GOOGL --compare

# Watchlist screening
python scripts\analyze_stock_consensus.py --watchlist 50 --min-score 70 --output results.csv
```

**Output Example**:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PYPL - PayPal Holdings Inc.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Valuation
   Fair Value:   $92.40
   Current:      $68.50
   MoS:          34.8% ↗

✅ Consensus: 84/100 → STRONG BUY

⚖️  Breakdown
   LSTM-DCF:     84.8 (70%)
   Risk+Sent:    65.0 (20%)
   P/E Sanity:   82.0 (10%)
```

---

#### `build_ml_watchlist.py`

**Purpose**: Screen S&P 500 for GARP opportunities

**Criteria**:

- Revenue growth > 15%
- YTD return < 5%
- PEG ratio < 1.5
- Positive momentum

**Usage**:

```powershell
python scripts\build_ml_watchlist.py --min-mos 20 --max-risk Medium
```

---

### 🧠 **LSTM Scripts** (`scripts/lstm/`)

#### `train_lstm_dcf_enhanced.py` ⭐ CORE

**Purpose**: Train LSTM-DCF model on enhanced financial data

**Specs**:

- **Input**: 16 financial features
- **Output**: 2 growth rates (revenue%, fcf%)
- **Sequence**: 60 quarters (15 years)
- **Architecture**: 3-layer LSTM, 128 hidden
- **Training Time**: 10-15 mins (GPU) / 30-60 mins (CPU)

**Usage**:

```powershell
python scripts\lstm\train_lstm_dcf_enhanced.py

# With custom hyperparameters
python scripts\lstm\train_lstm_dcf_enhanced.py --hidden-size 256 --num-layers 4
```

**Outputs**:

- `models/lstm_dcf_enhanced.pth` (model + scaler + metadata)
- `lightning_logs/` (training logs)

---

#### `check_lstm_status.py`

**Purpose**: Monitor training progress

**Usage**:

```powershell
python scripts\lstm\check_lstm_status.py
```

---

### 🌲 **Random Forest Scripts** (`scripts/rf/`)

#### `train_rf_risk_sentiment.py` ⭐ NEW

**Purpose**: Train RF classifier with 14 features including news sentiment

**Features**:

- 6 Risk: beta, volatility, debt/equity, volume z-score, short %, RSI
- 4 Sentiment: mean, std, volume, relevance
- 4 Valuation: P/E, P/B, margin, ROE

**Target**: Risk class (Low / Medium / High) based on 6-month drawdown

**Usage**:

```powershell
# Train on S&P 500 sample
python scripts\rf\train_rf_risk_sentiment.py

# Use existing data
python scripts\rf\train_rf_risk_sentiment.py --use-existing-data

# Custom tickers
python scripts\rf\train_rf_risk_sentiment.py --tickers-file my_stocks.csv
```

**Training Time**: 3-5 minutes

**Outputs**:

- `models/rf_risk_sentiment.pkl`
- `models/rf_risk_sentiment_feature_importance.csv`

---

### ⚖️ **Consensus Scripts** (`scripts/consensus/`)

#### `test_reverse_dcf.py` ⭐ NEW

**Purpose**: Validate growth rates via reverse DCF

**Logic**:

1. Plug market price into DCF
2. Solve for **implied growth rate**
3. Compare with LSTM prediction
4. Flag if difference > 5%

**Usage**:

```powershell
# Single stock
python scripts\consensus\test_reverse_dcf.py PYPL

# With LSTM comparison
python scripts\consensus\test_reverse_dcf.py PYPL --lstm-growth 8.2

# Batch processing
python scripts\consensus\test_reverse_dcf.py AAPL MSFT GOOGL --batch --output validation.csv
```

**Output Example**:

```
📊 PYPL - Reverse DCF Validation Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Market Data:
   Current Price:    $68.50
   Market Cap:       $74.2B
   Current FCF:      $5.8B

📈 Growth Analysis:
   Implied Growth:   7.9%
   LSTM Growth:      8.2%
   Difference:       0.3%
   Validation:       ✅ OK

💎 Valuation:
   Fair Value:       $92.40
   Margin of Safety: 34.8%
```

---

### 📊 **Data Collection Scripts** (`scripts/data_collection/`)

#### `fetch_enhanced_training_data.py`

**Purpose**: Fetch financial statements from Alpha Vantage + Yahoo Finance

**Data Sources**:

- **Primary**: Alpha Vantage (INCOME_STATEMENT, CASH_FLOW, BALANCE_SHEET)
- **Fallback**: Yahoo Finance quarterly_financials

**Rate Limits**:

- Alpha Vantage: 25 calls/day
- Yahoo Finance: Unlimited

**Usage**:

```powershell
# Fetch all S&P 500
python scripts\data_collection\fetch_enhanced_training_data.py

# Specific tickers
python scripts\data_collection\fetch_enhanced_training_data.py --tickers AAPL MSFT GOOGL
```

**Outputs**: `data/raw/financial_statements/{TICKER}_{statement}.csv`

---

#### `build_enhanced_training_data.py`

**Purpose**: Combine financial statements into ML-ready dataset

**Process**:

1. Load income/cashflow/balance sheets
2. Calculate 29 features (core + margins + normalized + growth)
3. Concatenate all stocks
4. Save to CSV

**Usage**:

```powershell
python scripts\data_collection\build_enhanced_training_data.py
```

**Output**: `data/processed/training/lstm_dcf_training_enhanced.csv`

- 8,828 records
- 117 stocks
- 29 features

---

### 🧪 **Evaluation Scripts** (`scripts/evaluation/`)

#### `backtest_consensus_strategy.py` ⭐ NEW

**Purpose**: Backtest 70/20/10 strategy with portfolio metrics

**Strategy**:

- **Buy**: Consensus > 70, MoS > 10%
- **Sell**: Consensus < 40, MoS < -10%
- **Hold**: Otherwise

**Metrics**:

- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Holding period

**Usage**:

```powershell
# Full backtest (2015-2025)
python scripts\evaluation\backtest_consensus_strategy.py --start 2015-01-01 --end 2025-11-01

# Quick test (10 stocks)
python scripts\evaluation\backtest_consensus_strategy.py --quick

# Custom capital & rebalance
python scripts\evaluation\backtest_consensus_strategy.py --capital 50000 --rebalance monthly
```

**Output Example**:

```
📊 Backtest Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Period: 2015-01-01 to 2025-11-01
Duration: 10.8 years

💰 Returns:
   Initial Capital:     $100,000.00
   Final Value:         $287,340.00
   Total Return:        +187.34%
   Annualized Return:   +10.32%

📈 Risk Metrics:
   Sharpe Ratio:        1.42
   Max Drawdown:        -22.3%

📊 Trading Stats:
   Total Trades:        248
   Win Rate:            64.2%
   Avg Win:             +12.4%
   Avg Loss:            -7.8%
```

---

#### `quick_model_test.py`

**Purpose**: Fast model inference test

**Usage**:

```powershell
python scripts\evaluation\quick_model_test.py
```

**Checks**:

- Model loads successfully
- Inference speed (ms)
- Output shape correctness

---

#### `validate_enhanced_data.py`

**Purpose**: Quick data quality check (AAPL only)

**Usage**:

```powershell
python scripts\evaluation\validate_enhanced_data.py
```

**Validates**:

- 81 quarters of data
- No missing values
- Realistic growth rates
- Margin ranges

---

### 🧰 **Utility Scripts**

#### `test_agent.py`

Test risk agent with Groq LLM

#### `test_multiagent_system.py`

Test multi-agent orchestration

#### `test_news_sentiment.py`

Test Alpha Vantage NEWS_SENTIMENT API

#### `generate_assessment_report.py`

Generate FYP documentation report

---

## 🗑️ **Archived Scripts** (`scripts/archive/`)

These scripts are **obsolete** and moved to archive:

| Script                        | Reason                                                  |
| ----------------------------- | ------------------------------------------------------- |
| `fetch_lstm_training_data.py` | Used price-based FCFF_proxy (413x inflated)             |
| `train_lstm_dcf.py`           | Old LSTM trainer (replaced by enhanced version)         |
| `simple_train.py`             | Non-production test script                              |
| `test_training_data.py`       | Replaced by `validate_enhanced_data.py`                 |
| `retry_failed_tickers.py`     | Logic integrated into `build_enhanced_training_data.py` |
| `test_valuation_system.py`    | Replaced by `analyze_stock_consensus.py`                |
| `test_enhanced_agent.py`      | Replaced by `analyze_stock_consensus.py`                |
| `analyze_stock.py`            | Non-consensus version                                   |

**Note**: These are kept for reference only. Do not use in production.

---

## 🎯 **Development Workflow**

### Phase 1: Data Collection

```powershell
# Step 1: Fetch financial statements (Alpha Vantage + Yahoo Finance)
python scripts\data_collection\fetch_enhanced_training_data.py

# Step 2: Build training dataset (8,828 records)
python scripts\data_collection\build_enhanced_training_data.py

# Step 3: Validate data quality
python scripts\evaluation\validate_enhanced_data.py
```

### Phase 2: Model Training

```powershell
# Step 4: Train LSTM-DCF (10-15 mins)
python scripts\lstm\train_lstm_dcf_enhanced.py

# Step 5: Train RF Risk+Sentiment (3-5 mins)
python scripts\rf\train_rf_risk_sentiment.py

# Step 6: Quick model test
python scripts\evaluation\quick_model_test.py
```

### Phase 3: Validation

```powershell
# Step 7: Reverse DCF validation
python scripts\consensus\test_reverse_dcf.py PYPL

# Step 8: Backtest strategy
python scripts\evaluation\backtest_consensus_strategy.py --quick
```

### Phase 4: Production Analysis

```powershell
# Step 9: Analyze stocks
python scripts\analyze_stock_consensus.py AAPL MSFT GOOGL --compare

# Step 10: Build watchlist
python scripts\build_ml_watchlist.py --min-mos 20
```

---

## 📝 **Best Practices**

1. **Always use virtual environment**: `venv\Scripts\python.exe`
2. **Check model status before analysis**: `scripts\lstm\check_lstm_status.py`
3. **Validate data quality regularly**: `scripts\evaluation\validate_enhanced_data.py`
4. **Run quick tests after training**: `scripts\evaluation\quick_model_test.py`
5. **Backtest before production**: `scripts\evaluation\backtest_consensus_strategy.py`

---

## 🚨 **Troubleshooting**

### Import Errors

```powershell
# Lint warnings are normal (Pylance doesn't see venv)
# If actual runtime error:
pip install -r requirements.txt
```

### Model Not Found

```powershell
# Check if models exist
ls models\*.pth
ls models\*.pkl

# Retrain if missing
python scripts\lstm\train_lstm_dcf_enhanced.py
python scripts\rf\train_rf_risk_sentiment.py
```

### Data Missing

```powershell
# Rebuild training data
python scripts\data_collection\build_enhanced_training_data.py
```

### GPU Not Detected

```powershell
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Training will fall back to CPU automatically
```

---

## 📚 **Additional Resources**

- **Pipeline Documentation**: `EMETIX_ML_PIPELINES_FINAL.md`
- **Implementation Guide**: `LSTM_ENHANCED_IMPLEMENTATION.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **Quick Start**: `QUICKSTART.md`

---

**Last Updated**: November 10, 2025  
**Status**: ✅ LSTM Trained | 🔄 RF In Progress | 📋 Consensus Pending
