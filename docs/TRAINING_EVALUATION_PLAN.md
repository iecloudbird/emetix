# Training & Evaluation Plan for FYP Assessment

**Date:** October 23, 2025  
**Phase:** 6 → Phase 3 Preparation  
**Data Available:** 12 stocks (930 quarters) from Alpha Vantage

---

## Executive Summary

You have **TWO parallel training paths** available:

### Path A: Quick Training with 12 Stocks (TODAY) ⚡

- Train LSTM Growth Forecaster with current 12 stocks
- Fast iteration for FYP assessment
- **Timeline:** 2-3 hours

### Path B: Production Training with 136 Stocks (2-3 WEEKS) 🏭

- Continue daily data collection (~16 more days)
- Train with full dataset for production quality
- **Timeline:** 16 days collection + 1 day training

**Recommendation:** Do BOTH! Train now with 12 stocks for assessment, continue collecting for production.

---

## Current Model Inventory

### ✅ Already Trained (Phase 6)

| Model                | Status     | Records | Performance           | Location                      |
| -------------------- | ---------- | ------- | --------------------- | ----------------------------- |
| **LSTM-DCF (Price)** | ✅ Trained | 111,294 | Val Loss: 0.000092    | `models/lstm_dcf_final.pth`   |
| **RF Ensemble**      | ✅ Trained | Unknown | P/E importance: 98.7% | `models/rf_ensemble.pkl`      |
| **Linear Valuation** | ✅ Trained | Unknown | Basic regression      | `models/linear_valuation.pkl` |
| **Risk Classifier**  | ✅ Trained | Unknown | Beta classification   | `models/risk_classifier.pkl`  |

### 🔄 In Progress (Phase 7)

| Model                      | Status         | Data Available          | Expected Performance       | Location                            |
| -------------------------- | -------------- | ----------------------- | -------------------------- | ----------------------------------- |
| **LSTM Growth Forecaster** | 📋 Not trained | 12 stocks, 930 quarters | Unknown (needs validation) | `models/lstm_growth_forecaster.pth` |

---

## Option 1: Quick Training with 12 Stocks (RECOMMENDED FOR NOW)

### Why Train Now?

1. **FYP Assessment:** Show working ML pipeline
2. **Rapid Iteration:** Test architecture quickly
3. **Proof of Concept:** Validate methodology
4. **Parallel Work:** Train while collecting more data

### Available Data (12 Stocks)

```
✅ Fetched: AAPL, ABBV, ABT, ACN, AMZN, APA, APD, AVGO, AXP, BA, BAC, BDX
📊 Total: 930 quarters (avg 77.5 quarters per stock)
🔢 Sequences: ~57 per stock × 12 = ~684 training sequences
```

### Step-by-Step Process

#### Step 1: Create Training Dataset (5 minutes)

```powershell
# Combine 12 stocks into LSTM growth training dataset
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset
```

**Expected Output:**

- File: `data/processed/lstm_dcf_training/lstm_growth_training_data.csv`
- Records: ~930 rows (quarterly data)
- Columns: `ticker, date, revenue_std, capex_std, da_std, nopat_std`

#### Step 2: Train LSTM Growth Forecaster (30-60 minutes)

```powershell
# Quick training with 12 stocks
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 30 --quick-test
```

**Training Parameters:**

- Epochs: 30 (reduced from 50 for speed)
- Batch size: 32 (reduced from 64 for small dataset)
- Sequence length: 20 quarters
- Expected sequences: ~684 (930 - 20×12)

**Expected Results:**

- Training time: ~30 minutes (GPU) / ~2 hours (CPU)
- Validation loss: Target < 0.01 (MSE)
- R² per component: Target > 0.6
- Model size: ~1.5 MB

#### Step 3: Evaluate Model Performance (30 minutes)

```powershell
# Run comprehensive evaluation
.\venv\Scripts\python.exe scripts\evaluate_lstm_growth.py
```

**Evaluation Metrics:**

- Growth rate forecasting accuracy (MSE, MAE, R²)
- DCF valuation accuracy (compare to traditional DCF)
- Component-wise performance (Revenue, CapEx, D&A, NOPAT)
- Out-of-sample testing on 2 held-out stocks

#### Step 4: Create Assessment Report (1 hour)

```powershell
# Generate comprehensive model report
.\venv\Scripts\python.exe scripts\generate_assessment_report.py
```

**Report Contents:**

- Model architecture details
- Training history (loss curves)
- Performance metrics vs baselines
- Example valuations with visualizations
- Limitations and future improvements

---

## Option 2: Continue Collection for Production (16 Days)

### Why Wait?

1. **Better Performance:** More data = better generalization
2. **Robust Models:** Handle diverse companies
3. **Production Ready:** Meet SRS non-functional requirements
4. **Academic Rigor:** Larger sample size for FYP

### Recommended Timeline

```
Week 1 (Oct 23-29):  Train with 12 stocks, continue daily collection
Week 2 (Oct 30-Nov 5):  Collect 7 days × 8 stocks = 56 stocks (total: 68)
Week 3 (Nov 6-Nov 8):   Collect 3 days × 8 stocks = 24 stocks (total: 92)
                        Retrain with 92 stocks, compare results
Week 4 (Nov 9-Nov 12):  Complete collection to 136 stocks
                        Final production training
```

---

## Comprehensive Model Evaluation for FYP Assessment

### Evaluation Framework

#### 1. Traditional Models (Already Trained)

**Test Script:** `scripts/evaluate_traditional_models.py` (CREATE)

```python
# Evaluate:
# - Linear Valuation Model
# - RF Ensemble Model
# - Risk Classifier
# - Compare against benchmarks
```

**Metrics:**

- Accuracy vs manual calculations
- R² scores per model
- Inference time (must be < 300ms per SRS)
- Feature importance analysis

#### 2. Deep Learning Models

**Test Script:** `scripts/evaluate_deep_learning_models.py` (CREATE)

```python
# Evaluate:
# - LSTM-DCF (Price Prediction) - Current model
# - LSTM Growth Forecaster - New model
# - Hybrid LSTM-DCF - Combined approach
```

**Metrics:**

- Validation loss trends
- Out-of-sample prediction accuracy
- Comparison to traditional DCF
- Computational efficiency

#### 3. Ensemble & Consensus

**Test Script:** `scripts/evaluate_consensus_scoring.py` (CREATE)

```python
# Evaluate:
# - 4-model consensus (LSTM-DCF, RF, Linear, Risk)
# - Weighted voting accuracy
# - Confidence calibration
```

**Metrics:**

- Consensus vs individual model accuracy
- Sharpe ratio improvement
- Hit rate on buy/hold/sell recommendations

#### 4. System Integration

**Test Script:** `scripts/evaluate_full_system.py` (CREATE)

```python
# End-to-end testing:
# - analyze_stock.py with all models
# - Multi-agent system coordination
# - Real-time performance (< 2s per SRS)
```

**Metrics:**

- Total analysis time (target: < 2s)
- ML inference time (target: < 300ms)
- Memory usage
- Error handling robustness

---

## Assessment Deliverables

### For FYP Presentation/Report

1. **Model Performance Report** (PDF)

   - All metrics in tables
   - Training loss curves
   - Comparison charts (models vs benchmarks)
   - Example predictions with visualizations

2. **Live Demo Script**

   - `analyze_stock.py AAPL` with all features
   - Show ML predictions vs traditional
   - Demonstrate consensus scoring
   - Real-time performance

3. **Code Quality Evidence**

   - Test coverage report (pytest)
   - Documentation completeness
   - Architecture diagrams (updated)
   - Git commit history showing progress

4. **SRS Compliance Matrix**
   - FR-ML-1 to FR-ML-5: ✅ / ❌ status
   - NFR-ML-1 to NFR-ML-4: Performance evidence
   - Functional requirements met: X / Y

---

## Recommended Action Plan (IMMEDIATE)

### Today (October 23, 2025)

#### Morning: Train LSTM Growth Forecaster (3 hours)

```powershell
# 1. Create dataset from 12 stocks
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset

# 2. Train model (quick iteration)
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 30 --batch-size 32

# 3. Quick validation
.\venv\Scripts\python.exe -c "from src.models.deep_learning.lstm_growth_forecaster import LSTMGrowthForecaster, DCFValuationWithLSTM; print('✅ Model can be loaded')"
```

#### Afternoon: Create Evaluation Scripts (4 hours)

**Priority 1: Evaluate Traditional Models**
Create `scripts/evaluate_traditional_models.py`:

- Load Linear Valuation, RF Ensemble, Risk Classifier
- Test on 5 stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)
- Generate performance report

**Priority 2: Evaluate LSTM Models**
Create `scripts/evaluate_deep_learning_models.py`:

- Load LSTM-DCF (price model)
- Load LSTM Growth Forecaster (new model)
- Compare predictions
- Generate visualizations

**Priority 3: End-to-End Test**
Create `scripts/evaluate_full_system.py`:

- Run full analysis pipeline
- Measure timing (must meet SRS: < 2s valuation, < 300ms ML)
- Test error handling
- Generate summary report

### Tomorrow (October 24, 2025)

#### Morning: Generate Assessment Report (3 hours)

Create `scripts/generate_assessment_report.py`:

- Compile all metrics
- Create visualizations (matplotlib/seaborn)
- Generate PDF report
- Update SRS compliance matrix

#### Afternoon: Continue Data Collection (5 minutes)

```powershell
# Daily batch fetch (8 more stocks)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --daily-limit 10
```

### Next 2 Weeks

- **Daily:** Run data collection script (5 min/day)
- **Weekly:** Retrain models with new data (1 hour/week)
- **Ongoing:** Refine evaluation scripts, update documentation

---

## Expected Outcomes

### With 12 Stocks (Today)

**Pros:**

- ✅ Working LSTM Growth Forecaster trained
- ✅ Can demonstrate full pipeline for FYP
- ✅ Proof of concept complete
- ✅ Rapid iteration for improvements

**Cons:**

- ⚠️ Limited generalization (only 12 companies)
- ⚠️ May overfit to specific sectors
- ⚠️ Lower confidence in predictions
- ⚠️ Not production-ready

**Performance Estimate:**

- R² per component: 0.5 - 0.7 (moderate)
- MSE: 0.01 - 0.05 (acceptable for PoC)
- DCF accuracy: ±20% vs traditional (experimental)

### With 136 Stocks (3 Weeks)

**Pros:**

- ✅ Production-quality model
- ✅ Better generalization across sectors
- ✅ Higher confidence predictions
- ✅ Meets academic rigor standards

**Cons:**

- ⏰ Longer wait time
- 💰 May need premium API (if free tier insufficient)

**Performance Estimate:**

- R² per component: 0.7 - 0.85 (good)
- MSE: 0.001 - 0.01 (very good)
- DCF accuracy: ±10% vs traditional (production)
- **Expected improvement:** +36% Sharpe ratio (per research article)

---

## Quick Start Commands (Copy-Paste Ready)

### 1. Create Training Dataset

```powershell
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset
```

### 2. Train LSTM Growth Forecaster

```powershell
# Quick training (30 epochs, ~30 min GPU)
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 30 --batch-size 32

# Full training (50 epochs, ~1 hour GPU)
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 50
```

### 3. Test Model Loading

```powershell
.\venv\Scripts\python.exe -c "from src.models.deep_learning.lstm_growth_forecaster import LSTMGrowthForecaster; model = LSTMGrowthForecaster(); model.load_model('models/lstm_growth_forecaster.pth'); print('✅ Model loaded successfully')"
```

### 4. Check Status

```powershell
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --status
```

---

## Questions to Answer for Assessment

### Technical Questions

1. **Why LSTM for growth forecasting?**

   - Captures temporal dependencies in financial data
   - Better than ARIMA for non-stationary series
   - Research-backed methodology (per article)

2. **How do you handle limited data (12 stocks)?**

   - Normalization by assets (scale-invariant)
   - Standardization (mean=0, std=1)
   - Transfer learning from larger stocks to smaller
   - Plan to expand to 136+ stocks

3. **What's the advantage over traditional DCF?**

   - Dynamic growth rates vs static assumptions
   - Learns from historical patterns
   - Adapts to economic cycles
   - Expected +36% Sharpe improvement

4. **How do you validate the model?**
   - Train/val/test split (70/15/15)
   - Out-of-sample testing
   - Compare to benchmark (S&P 500, traditional DCF)
   - Backtest on historical data

### Business Questions

1. **Is this production-ready?**

   - With 12 stocks: Proof of concept (MVP)
   - With 136 stocks: Production beta
   - Continuous improvement planned

2. **How does this fit the SRS?**

   - Meets FR-ML-1 to FR-ML-5 ✅
   - NFR-ML-1: Inference < 300ms (need to verify)
   - NFR-ML-2: Val loss < 0.0001 (target for 136 stocks)
   - NFR-ML-4: Graceful fallback implemented ✅

3. **What's the unique value proposition?**
   - Free institutional-grade analysis
   - ML-powered vs traditional screeners
   - Multi-agent intelligence
   - Transparent methodology

---

## Next Steps Decision Tree

```
START: You have 12 stocks with 930 quarters
│
├─ NEED ASSESSMENT NOW (Within 1 week)?
│  │
│  ├─ YES → Train with 12 stocks TODAY
│  │        Continue collecting in parallel
│  │        Retrain weekly as data grows
│  │
│  └─ NO  → Wait for full 136 stocks (16 days)
│           Train once with complete dataset
│           Higher quality, longer wait
│
└─ RECOMMENDED: Do BOTH!
   1. Train now (12 stocks) for demo/assessment
   2. Continue daily collection (16 days)
   3. Retrain (136 stocks) for production
   4. Compare results in final report
```

---

## Summary & Recommendation

### ✅ YES, Train Now with 12 Stocks

**Reasoning:**

1. You have enough data for proof-of-concept (~684 sequences)
2. Can demonstrate working pipeline for FYP assessment
3. Fast iteration to find and fix issues
4. Parallel data collection continues in background
5. Can retrain with more data later

**Timeline:**

- **Today:** 3-4 hours to train and validate
- **Tomorrow:** 3 hours to create evaluation report
- **Result:** Working LSTM Growth Forecaster for assessment

### 🔄 Continue Collection for Production

- Run daily batch script (5 min/day)
- ~16 more days to complete 136 stocks
- Retrain weekly to compare improvement
- Final production model by November 8

### 📊 Create Evaluation Framework

**Priority Scripts to Create:**

1. `evaluate_traditional_models.py` - Test existing models
2. `evaluate_deep_learning_models.py` - Test LSTM models
3. `evaluate_consensus_scoring.py` - Test ensemble
4. `generate_assessment_report.py` - Compile results

---

**Action:** Shall I create the training and evaluation scripts now so you can start training with your 12 stocks?
