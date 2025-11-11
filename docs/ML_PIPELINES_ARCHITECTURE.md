# Machine Learning Pipelines & System Architecture

## JobHedge Investor - FYP Documentation

---

## 📊 MACHINE LEARNING PIPELINES

### Pipeline 1: LSTM-DCF (Deep Learning) Pipeline

**Purpose:** Forecast DCF component growth rates using time-series LSTM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LSTM-DCF TRAINING PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘

[1] DATA COLLECTION (Alpha Vantage API)
    ├── Alpha Vantage API (25 calls/day limit)
    │   ├── Income Statement (quarterly)
    │   ├── Cash Flow Statement (quarterly)
    │   └── Balance Sheet (quarterly)
    │
    └── AlphaVantageFinancialsFetcher
        ├── fetch_income_statement(ticker)
        ├── fetch_cash_flow(ticker)
        └── fetch_balance_sheet(ticker)

    Output: Raw CSV files
    ├── data/raw/financial_statements/{ticker}_income.csv
    ├── data/raw/financial_statements/{ticker}_cashflow.csv
    └── data/raw/financial_statements/{ticker}_balance.csv

    Script: scripts/fetch_lstm_training_data.py
    Command: python scripts/fetch_lstm_training_data.py --daily-limit 10
    Status: 86 stocks fetched, 6,501 quarters, 6,635 records

        ↓

[2] DATA PREPARATION
    ├── FinancialStatementsFetcher
    │   ├── fetch_quarterly_financials(ticker)
    │   └── Extract components:
    │       ├── Revenue (Total Revenue)
    │       ├── CapEx (Capital Expenditures)
    │       ├── D&A (Depreciation & Amortization)
    │       ├── EBIT (Operating Income)
    │       ├── Tax Rate (calculated)
    │       └── NOPAT = EBIT × (1 - tax_rate)
    │
    └── normalize_by_assets(df)
        └── Each metric / Total Assets

    Output: Normalized metrics per quarter
    ├── revenue_norm
    ├── capex_norm
    ├── da_norm
    └── nopat_norm

        ↓

[3] STANDARDIZATION
    ├── standardize_metrics(df)
    │   └── (metric - mean) / std
    │
    └── Output: Standardized features (mean=0, std=1)
        ├── revenue_std
        ├── capex_std
        ├── da_std
        └── nopat_std

    File: data/processed/lstm_dcf_training/lstm_growth_training_data.csv
    Shape: (6,635 records, 6 columns)
    Columns: [ticker, date, revenue_std, capex_std, da_std, nopat_std]

        ↓

[4] SEQUENCE CREATION
    ├── create_sequences(df, sequence_length=20)
    │   ├── Group by ticker
    │   ├── Sort by date
    │   └── Create overlapping windows:
    │       ├── Input: 20 quarters of [rev, capex, da, nopat]
    │       └── Target: Growth rates for next quarter
    │
    └── Output:
        ├── X: (num_sequences, 20, 4) - sequences
        └── y: (num_sequences, 4) - growth rate targets

    Split: 70% train, 15% val, 15% test

        ↓

[5] MODEL ARCHITECTURE
    ┌──────────────────────────────────────┐
    │      LSTMGrowthForecaster           │
    ├──────────────────────────────────────┤
    │ Input: (batch, 20, 4)               │
    │   ├── Sequence length: 20 quarters  │
    │   └── Features: 4 (rev,capex,da,no) │
    │                                      │
    │ LSTM Layer 1:                        │
    │   ├── Hidden size: 64                │
    │   ├── Dropout: 0.2                   │
    │   └── Bidirectional: False           │
    │                                      │
    │ LSTM Layer 2:                        │
    │   ├── Hidden size: 64                │
    │   └── Dropout: 0.2                   │
    │                                      │
    │ Fully Connected:                     │
    │   ├── Input: 64                      │
    │   └── Output: 4 (growth rates)       │
    │                                      │
    │ Output: (batch, 4)                   │
    │   ├── Revenue growth rate            │
    │   ├── CapEx growth rate              │
    │   ├── D&A growth rate                │
    │   └── NOPAT growth rate              │
    └──────────────────────────────────────┘

        ↓

[6] TRAINING
    ├── Loss: MSE (Mean Squared Error)
    ├── Optimizer: Adam (lr=0.001)
    ├── Scheduler: ReduceLROnPlateau
    ├── Epochs: 30-50
    ├── Batch size: 32
    └── Early stopping: patience=10

    Device: CUDA (GPU) or CPU
    Training time: ~5-10 mins (GPU) / ~30-60 mins (CPU)

    Script: scripts/train_lstm_growth_forecaster.py
    Command: python scripts/train_lstm_growth_forecaster.py --epochs 30

        ↓

[7] MODEL OUTPUT
    ├── models/lstm_growth_forecaster.pth (212 KB)
    │   ├── Model weights
    │   ├── Architecture: 2-layer LSTM, hidden_size=64
    │   └── Input: (batch, 20, 4) → Output: (batch, 4)
    │
    └── Evaluation metrics:
        ├── Test R² score
        ├── MSE per component
        └── Growth rate accuracy

        ↓

[8] INFERENCE (in analyze_stock.py)
    User input: ticker (e.g., "AAPL")
        ↓
    TimeSeriesProcessor.fetch_sequential_data(ticker, period='5y')
        ↓
    Prepare 20-quarter sequence
        ↓
    Model prediction: growth_rates = model(sequence)
        ↓
    Forecast 10-year FCFF using growth rates
        ↓
    DCF valuation: Fair Value per share

```

---

### Pipeline 2: Random Forest Ensemble Pipeline

**Purpose:** Multi-metric stock valuation using fundamental features

```
┌─────────────────────────────────────────────────────────────────────┐
│               RANDOM FOREST ENSEMBLE PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘

[1] DATA COLLECTION (Yahoo Finance)
    ├── YFinanceFetcher
    │   ├── fetch_stock_data(ticker)
    │   │   ├── Stock.info (fundamentals)
    │   │   └── Stock.history(period='1y') (prices)
    │   │
    │   └── Extract features:
    │       ├── P/E Ratio (trailingPE)
    │       ├── Forward P/E (forwardPE)
    │       ├── Debt/Equity (debtToEquity)
    │       ├── Current Ratio (currentRatio)
    │       ├── Market Cap (marketCap)
    │       ├── Beta (beta)
    │       ├── Dividend Yield (dividendYield)
    │       ├── EPS (trailingEps)
    │       ├── Revenue Growth (revenueGrowth)
    │       ├── Volatility (std of returns)
    │       └── Current Price (currentPrice)
    │
    └── Output: DataFrame with 12 features per stock

    Script: scripts/fetch_historical_data.py
    Stocks: 50 S&P 500 sample tickers
    No rate limit (yfinance is free)

        ↓

[2] FEATURE ENGINEERING
    ├── Raw features (12):
    │   ├── pe_ratio
    │   ├── forward_pe
    │   ├── debt_equity
    │   ├── current_ratio
    │   ├── market_cap
    │   ├── beta
    │   ├── dividend_yield
    │   ├── eps
    │   ├── revenue_growth
    │   ├── volatility
    │   ├── current_price
    │   └── ticker (for tracking)
    │
    └── Handle missing values:
        ├── Fill NaN with 0 or median
        └── Remove outliers (optional)

        ↓

[3] TARGET CREATION
    ├── Regression target:
    │   └── Future returns estimate
    │       └── (Can use historical returns or analyst estimates)
    │
    └── Classification target (optional):
        └── Risk levels: Low (0), Medium (1), High (2)
            └── Based on beta + volatility thresholds

        ↓

[4] TRAIN/TEST SPLIT
    ├── Train: 80% of stocks
    ├── Test: 20% of stocks
    └── Random state: 42 (reproducible)

        ↓

[5] MODEL ARCHITECTURE
    ┌──────────────────────────────────────┐
    │      RFEnsembleModel                │
    ├──────────────────────────────────────┤
    │ RandomForestRegressor                │
    │   ├── n_estimators: 200 trees        │
    │   ├── max_depth: 15                   │
    │   ├── min_samples_split: 5            │
    │   ├── min_samples_leaf: 2             │
    │   └── random_state: 42                │
    │                                      │
    │ Features: 12                          │
    │   ├── P/E Ratio (98.7% importance)   │
    │   ├── Revenue Growth (0.5%)           │
    │   ├── Beta (0.3%)                     │
    │   └── ... (remaining features)        │
    │                                      │
    │ Output: Valuation score (0-100)      │
    └──────────────────────────────────────┘

        ↓

[6] TRAINING
    ├── Algorithm: Random Forest (scikit-learn)
    ├── Training time: ~2-5 minutes
    ├── Cross-validation: 5-fold CV
    └── Hyperparameter tuning (optional):
        └── GridSearchCV or RandomizedSearchCV

    Script: scripts/train_rf_ensemble.py
    Command: python scripts/train_rf_ensemble.py

        ↓

[7] MODEL OUTPUT
    ├── models/rf_ensemble.pkl (210 KB, joblib format)
    │   ├── Trained RandomForestRegressor
    │   └── 200 decision trees
    │
    └── models/rf_feature_importance.csv
        ├── Feature ranking by importance
        └── Helps interpret model predictions

        ↓

[8] INFERENCE (in EnhancedValuationAgent)
    User input: ticker (e.g., "TSLA")
        ↓
    YFinanceFetcher.fetch_stock_data(ticker)
        ↓
    Extract 12 features
        ↓
    Model prediction: score = rf_model.predict(features)
        ↓
    Consensus scoring with other models
        ↓
    Final recommendation: Buy/Hold/Sell

```

---

### Pipeline 3: Consensus Scoring Pipeline

**Purpose:** Combine multiple models for robust valuation

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CONSENSUS SCORING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Stock ticker (e.g., "AAPL")
    │
    ├──► [Model 1] LSTM-DCF (40% weight)
    │    └── Growth rate forecasting → DCF valuation
    │
    ├──► [Model 2] RF Ensemble (30% weight)
    │    └── Multi-metric fundamental analysis
    │
    ├──► [Model 3] Linear Valuation (20% weight)
    │    └── Traditional regression on P/E, D/E, etc.
    │
    └──► [Model 4] Risk Classifier (10% weight)
         └── Beta + volatility risk assessment

         ↓

ConsensusScorer.calculate_consensus(scores_dict)
    ├── Weighted average of model scores
    ├── Agreement level (std deviation)
    └── Confidence score (0-100)

         ↓

OUTPUT: Consensus recommendation
    ├── Fair value estimate
    ├── Confidence level
    ├── Buy/Hold/Sell signal
    └── Risk-adjusted rating

```

---

## 🏗️ OVERALL SYSTEM ARCHITECTURE

### System Components Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                      JOBHEDGE INVESTOR SYSTEM                       │
└─────────────────────────────────────────────────────────────────────┘

[LAYER 1] USER INTERFACE
    ├── Command-Line Interface (CLI)
    │   └── scripts/analyze_stock.py
    │       ├── Single stock analysis
    │       ├── Batch analysis (multiple stocks)
    │       ├── Stock comparison
    │       └── Growth opportunity screening
    │
    └── Future: Web Dashboard (React + Flask API)
        ├── frontend/ (React UI)
        └── src/api/ (Flask/FastAPI backend)

            ↓

[LAYER 2] AI AGENTS (LangChain + Groq LLM)
    ├── SupervisorAgent
    │   └── Orchestrates multi-agent workflow
    │
    ├── EnhancedValuationAgent
    │   ├── Tools: 5 ML-powered valuation tools
    │   ├── Uses: LSTM-DCF, RF Ensemble, Linear models
    │   └── Output: Natural language analysis
    │
    ├── RiskAgent
    │   ├── Assesses stock risk (beta, volatility)
    │   └── Classification: Low/Medium/High risk
    │
    ├── ValuationAgent
    │   ├── Traditional DCF calculations
    │   └── Fair value estimation
    │
    ├── FundamentalsAnalyzerAgent
    │   ├── P/E, P/B, PEG analysis
    │   └── Financial health scoring
    │
    ├── SentimentAnalyzerAgent
    │   ├── News sentiment analysis
    │   └── Multi-source aggregation
    │
    ├── DataFetcherAgent
    │   └── Coordinates data retrieval
    │
    └── WatchlistManagerAgent
        └── Tracks and monitors stocks

            ↓

[LAYER 3] ANALYSIS MODULES
    ├── ValuationAnalyzer
    │   ├── 12+ valuation metrics
    │   ├── 0-100 scoring system
    │   ├── Fair value calculation
    │   └── Buy/Hold/Sell recommendation
    │
    └── GrowthScreener
        ├── GARP strategy (Growth at Reasonable Price)
        ├── Screening criteria:
        │   ├── Revenue growth >15%
        │   ├── YTD return <5%
        │   ├── PEG ratio <1.5
        │   └── Positive momentum
        └── Growth opportunity ranking

            ↓

[LAYER 4] MACHINE LEARNING MODELS
    ├── Deep Learning Models
    │   ├── LSTMDCFModel (lstm_dcf_final.pth, 1.29 MB)
    │   │   ├── 3-layer LSTM
    │   │   ├── Input: 12 features, 60-period sequences
    │   │   ├── Hidden size: 128
    │   │   └── Output: 10-year FCFF forecast
    │   │
    │   └── LSTMGrowthForecaster (lstm_growth_forecaster.pth, 212 KB)
    │       ├── 2-layer LSTM
    │       ├── Input: 4 features, 20-quarter sequences
    │       ├── Hidden size: 64
    │       └── Output: 4 growth rates
    │
    ├── Ensemble Models
    │   ├── RFEnsembleModel (rf_ensemble.pkl, 210 KB)
    │   │   ├── 200 decision trees
    │   │   ├── 12 fundamental features
    │   │   └── P/E 98.7% importance
    │   │
    │   └── ConsensusScorer
    │       └── Weighted voting (4 models)
    │
    └── Traditional Models
        ├── LinearValuationModel
        │   ├── Features: [pe_ratio, debt_equity, revenue_growth, beta]
        │   └── Target: Fair value estimation
        │
        ├── DCFModel
        │   └── Classic DCF: (EPS × (1+g)) / (r-g)
        │
        └── FCFDCFModel
            └── FCFF-based DCF with WACC discount

            ↓

[LAYER 5] DATA LAYER
    ├── Data Fetchers
    │   ├── YFinanceFetcher (Primary)
    │   │   ├── Stock fundamentals (P/E, beta, etc.)
    │   │   ├── Historical prices (OHLCV)
    │   │   ├── No rate limit
    │   │   └── Free, reliable
    │   │
    │   ├── AlphaVantageFinancialsFetcher (Secondary)
    │   │   ├── Quarterly financial statements
    │   │   ├── Income, Cash Flow, Balance Sheet
    │   │   ├── Rate limit: 25 calls/day
    │   │   └── Used for LSTM training data
    │   │
    │   ├── NewsSentimentFetcher (Supplementary)
    │   │   ├── Tier 1: Yahoo Finance, NewsAPI
    │   │   ├── Tier 2: Finnhub (fallback)
    │   │   ├── Tier 3: Google News RSS
    │   │   └── Auto-deduplication (85% similarity)
    │   │
    │   └── FinancialStatementsFetcher
    │       └── Yahoo Finance quarterly data processing
    │
    └── Data Processors
        └── TimeSeriesProcessor
            ├── Sequence generation for LSTM
            ├── 60-period windows (LSTM-DCF)
            ├── 20-quarter windows (Growth Forecaster)
            └── Feature scaling and normalization

            ↓

[LAYER 6] CONFIGURATION & UTILITIES
    ├── config/
    │   ├── settings.py (paths, constants)
    │   ├── logging_config.py (logging setup)
    │   └── model_config.yaml (ML hyperparameters)
    │
    └── utils/
        └── Helper functions

            ↓

[LAYER 7] DATA STORAGE
    ├── data/raw/
    │   ├── stocks/ (Yahoo Finance data)
    │   ├── financial_statements/ (Alpha Vantage)
    │   ├── timeseries/ (LSTM training data)
    │   └── fundamentals/
    │
    ├── data/processed/
    │   ├── training/ (ML training datasets)
    │   ├── lstm_dcf_training/
    │   │   ├── lstm_growth_training_data.csv (6,635 records)
    │   │   └── fetch_progress.json (tracking)
    │   └── features/ (engineered features)
    │
    ├── models/ (Trained models)
    │   ├── lstm_dcf_final.pth
    │   ├── lstm_growth_forecaster.pth
    │   ├── rf_ensemble.pkl
    │   └── lstm_checkpoints/
    │
    └── data/cache/ (API response caching)

```

---

## 📋 COMPLETE COMPONENT LIST

### 1. AI Agents (src/agents/)

- `SupervisorAgent` - Multi-agent orchestration
- `EnhancedValuationAgent` - ML-powered valuation (5 tools)
- `RiskAgent` - Risk assessment agent
- `ValuationAgent` - Traditional valuation
- `FundamentalsAnalyzerAgent` - Fundamental analysis
- `SentimentAnalyzerAgent` - News sentiment
- `DataFetcherAgent` - Data coordination
- `WatchlistManagerAgent` - Portfolio tracking

### 2. Analysis Modules (src/analysis/)

- `ValuationAnalyzer` - 12+ metric scoring system
- `GrowthScreener` - GARP opportunity finder

### 3. ML Models (src/models/)

**Deep Learning (src/models/deep_learning/)**

- `LSTMDCFModel` - 10-year FCFF forecasting
- `LSTMGrowthForecaster` - Growth rate prediction
- `TimeSeriesDataset` - PyTorch dataset handler

**Ensemble (src/models/ensemble/)**

- `RFEnsembleModel` - Random Forest valuation
- `ConsensusScorer` - Multi-model consensus

**Traditional (src/models/valuation/)**

- `LinearValuationModel` - Linear regression
- `DCFModel` - Classic DCF calculator
- `FCFDCFModel` - FCFF-based DCF

**Risk (src/models/risk/)**

- Risk classification utilities

### 4. Data Fetchers (src/data/fetchers/)

- `YFinanceFetcher` - Yahoo Finance API
- `AlphaVantageFinancialsFetcher` - Alpha Vantage API
- `FinancialStatementsFetcher` - Financial statements
- `NewsSentimentFetcher` - Multi-source news aggregation

### 5. Data Processors (src/data/processors/)

- `TimeSeriesProcessor` - LSTM sequence generation

### 6. Configuration (config/)

- `settings.py` - Paths and constants
- `logging_config.py` - Logging configuration
- `model_config.yaml` - ML hyperparameters

### 7. Scripts (scripts/)

- `analyze_stock.py` - Interactive stock analysis CLI
- `fetch_historical_data.py` - Bulk Yahoo Finance fetch
- `fetch_lstm_training_data.py` - Alpha Vantage daily collection
- `train_lstm_dcf.py` - Train LSTM-DCF model
- `train_lstm_growth_forecaster.py` - Train growth forecaster
- `train_rf_ensemble.py` - Train Random Forest
- `retry_failed_tickers.py` - Retry failed fetches
- `check_lstm_status.py` - Model status checker
- `inspect_dataset.py` - Dataset inspector
- `test_*.py` - Various testing scripts

### 8. Data Files

- `data/processed/lstm_dcf_training/lstm_growth_training_data.csv` (6,635 records)
- `models/lstm_dcf_final.pth` (1.29 MB)
- `models/lstm_growth_forecaster.pth` (212 KB)
- `models/rf_ensemble.pkl` (210 KB)
- `models/rf_feature_importance.csv`

---

## 🔄 DATA FLOW DIAGRAM

```
┌──────────┐
│   USER   │
└────┬─────┘
     │ Input: ticker ("AAPL")
     ↓
┌─────────────────┐
│  analyze_stock  │ (CLI Interface)
└────┬────────────┘
     ↓
┌──────────────────────────────────────────────┐
│           StockAnalysisTool                  │
├──────────────────────────────────────────────┤
│  1. Valuation Analysis (ValuationAnalyzer)  │
│  2. Growth Screening (GrowthScreener)        │
│  3. News Sentiment (NewsSentimentFetcher)    │
│  4. AI Analysis (optional, if GROQ key)      │
│     ├── ValuationAgent                       │
│     └── RiskAgent                            │
└──────────────────────────────────────────────┘
     │
     ├─────► YFinanceFetcher.fetch_stock_data(ticker)
     │       └── Returns: fundamentals DataFrame
     │
     ├─────► TimeSeriesProcessor.fetch_sequential_data(ticker)
     │       └── Returns: 60-period LSTM sequence
     │
     ├─────► NewsSentimentFetcher.fetch_all_news(ticker)
     │       └── Returns: sentiment score + articles
     │
     ↓
┌──────────────────────────────────────────────┐
│          ML MODEL INFERENCE                  │
├──────────────────────────────────────────────┤
│  [Parallel Execution]                        │
│                                              │
│  Model 1: LSTM-DCF                           │
│    ├── Input: (1, 60, 12)                    │
│    └── Output: 10-year FCFF forecast         │
│                                              │
│  Model 2: RF Ensemble                        │
│    ├── Input: 12 fundamental features        │
│    └── Output: Valuation score (0-100)       │
│                                              │
│  Model 3: Linear Valuation                   │
│    ├── Input: [pe, debt_equity, growth, etc] │
│    └── Output: Fair value estimate           │
│                                              │
│  Model 4: Risk Classifier                    │
│    ├── Input: beta, volatility               │
│    └── Output: Low/Medium/High risk          │
└──────────────────────────────────────────────┘
     │
     ↓
┌──────────────────────────────────────────────┐
│       ConsensusScorer                        │
│  Weighted average: 40% + 30% + 20% + 10%    │
└──────────────────────────────────────────────┘
     │
     ↓
┌──────────────────────────────────────────────┐
│           OUTPUT TO USER                     │
├──────────────────────────────────────────────┤
│  ✅ Valuation Score: 78/100                  │
│  📊 Fair Value: $185.50 (Current: $180.25)   │
│  📈 Growth Score: 85/100 (GARP candidate)    │
│  📰 Sentiment: Positive (0.72, 35 articles)  │
│  🤖 AI Analysis: "Strong buy with moderate   │
│      risk. P/E attractive at 24.5x..."       │
│  💡 Recommendation: BUY                       │
│  ⚠️  Risk Level: Medium (beta=1.15)          │
└──────────────────────────────────────────────┘
```

---

## 📈 TRAINING STATISTICS

### LSTM Growth Forecaster

- **Dataset:** 6,635 records, 86 stocks, 22 years (2003-2025)
- **Architecture:** 2-layer LSTM, hidden_size=64
- **Training:** 30 epochs, batch_size=32
- **Device:** CUDA (RTX 3050, 6 mins) or CPU (30-60 mins)
- **Model size:** 212 KB
- **Status:** ✅ Trained and ready

### Random Forest Ensemble

- **Dataset:** 50 S&P 500 stocks, 12 features
- **Architecture:** 200 trees, max_depth=15
- **Training:** 2-5 minutes (CPU)
- **Feature importance:** P/E 98.7%, Revenue Growth 0.5%
- **Model size:** 210 KB
- **Status:** ✅ Trained and ready

### LSTM-DCF (Main)

- **Dataset:** 111,294 records (from earlier training)
- **Architecture:** 3-layer LSTM, hidden_size=128
- **Features:** 12 (close, volume, fundamentals, technical)
- **Sequences:** 60-period windows
- **Training:** GPU-accelerated, validation loss 0.000092
- **Model size:** 1.29 MB
- **Status:** ✅ Trained and ready

---

## 🎯 KEY INTEGRATION POINTS

### 1. analyze_stock.py Integration

```python
# User runs:
python scripts/analyze_stock.py AAPL

# System executes:
1. YFinanceFetcher → fetch fundamentals
2. TimeSeriesProcessor → prepare LSTM sequences
3. LSTMGrowthForecaster → predict growth rates
4. RFEnsembleModel → score fundamentals
5. ConsensusScorer → combine results
6. NewsSentimentFetcher → get news sentiment
7. ValuationAgent (AI) → natural language analysis
8. Display comprehensive report
```

### 2. Multi-Agent System

```python
# SupervisorAgent orchestrates:
1. DataFetcherAgent → retrieve data
2. FundamentalsAnalyzerAgent → analyze fundamentals
3. SentimentAnalyzerAgent → news sentiment
4. EnhancedValuationAgent → ML valuations
5. RiskAgent → risk assessment
6. WatchlistManagerAgent → tracking
```

### 3. Model Consensus Flow

```python
# EnhancedValuationAgent calls:
1. tool_lstm_dcf_valuation(ticker) → 40% weight
2. tool_rf_multimetric_analysis(ticker) → 30% weight
3. tool_traditional_valuation(ticker) → 20% weight
4. (risk classifier) → 10% weight
5. ConsensusScorer.calculate_consensus(scores)
6. Return: consensus recommendation
```

---

## 📝 USAGE EXAMPLES

### Example 1: Single Stock Analysis

```bash
python scripts/analyze_stock.py AAPL
```

### Example 2: Compare Multiple Stocks

```bash
python scripts/analyze_stock.py AAPL MSFT GOOGL --compare
```

### Example 3: Find Growth Opportunities

```bash
python scripts/analyze_stock.py AAPL MSFT TSLA NVDA --opportunities
```

### Example 4: Retry Failed Data Collection

```bash
python scripts/retry_failed_tickers.py --batch-size 10
```

### Example 5: Train Models

```bash
# Collect data (run daily)
python scripts/fetch_lstm_training_data.py --daily-limit 10

# Train LSTM Growth Forecaster
python scripts/train_lstm_growth_forecaster.py --epochs 30

# Train Random Forest
python scripts/train_rf_ensemble.py
```

---

This documentation provides all the components and data flows needed to create:

1. **Machine Learning Pipeline Diagrams** - for the 2 main ML systems
2. **System Architecture Diagram** - showing all 7 layers and components
3. **Data Flow Diagrams** - showing how data moves through the system

Use this as your reference for drawing comprehensive system design diagrams! 🎨
