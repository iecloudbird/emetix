# Phase 6: ML-Powered Watchlist Integration

**Objective**: Connect trained ML models (LSTM-DCF + RF Ensemble) to the multi-agent watchlist scoring system.

---

## 🎯 Implementation Plan

### Task 1: Enhance WatchlistManagerAgent with ML Scoring

**Current**: Uses basic weighted formula (growth 30%, sentiment 25%, valuation 20%, risk 15%, macro 10%)

**Target**: Add ML predictions as additional scoring factors

**Changes**:

1. Load LSTM-DCF and RF Ensemble models in `__init__`
2. Create new tool: `calculate_ml_enhanced_score`
3. Integrate ML predictions into composite scoring:
   - **LSTM-DCF prediction**: 25% weight (fair value estimation)
   - **RF expected return**: 20% weight (multi-metric analysis)
   - **Traditional factors**: 55% weight (existing)

**Files to modify**:

- `src/agents/watchlist_manager_agent.py`

---

### Task 2: Update SupervisorAgent to Use EnhancedValuationAgent

**Current**: Only uses traditional agents (Data, Sentiment, Fundamentals)

**Target**: Include ML-powered analysis in comprehensive reports

**Changes**:

1. Import `EnhancedValuationAgent` in supervisor
2. Add ML analysis to `orchestrate_stock_analysis_tool`
3. Include consensus valuation in final output

**Files to modify**:

- `src/agents/supervisor_agent.py`

---

### Task 3: Create ML-Enhanced Watchlist Builder Script

**New script**: `scripts/build_ml_watchlist.py`

**Features**:

- Takes list of tickers
- Runs comprehensive analysis (traditional + ML)
- Generates ranked watchlist with:
  - Composite scores (ML-enhanced)
  - Buy/Hold/Sell signals
  - Contrarian opportunity flags
  - Fair value estimates from LSTM-DCF
  - Risk-adjusted returns from RF

**Output**:

- Console display (formatted table)
- CSV export (`data/processed/watchlist_YYYYMMDD.csv`)
- JSON export for API integration

---

### Task 4: Update analyze_stock.py with ML Integration

**Current**: Uses ValuationAnalyzer (traditional metrics)

**Target**: Add ML-powered analysis section

**Changes**:

1. Import EnhancedValuationAgent
2. Add "ML Analysis" section after traditional valuation
3. Show LSTM-DCF fair value
4. Show RF expected return
5. Show consensus score

**Files to modify**:

- `scripts/analyze_stock.py`

---

## 📊 Architecture After Integration

```
User Request
    │
    ├─→ SupervisorAgent (Orchestrator)
    │       ├─→ DataFetcherAgent (fetch data)
    │       ├─→ SentimentAnalyzerAgent (news, social)
    │       ├─→ FundamentalsAnalyzerAgent (DCF, ratios)
    │       ├─→ EnhancedValuationAgent (NEW: ML models)
    │       │       ├─→ LSTM-DCF prediction
    │       │       ├─→ RF Ensemble prediction
    │       │       └─→ Consensus scoring
    │       └─→ WatchlistManagerAgent (ML-enhanced scoring)
    │               ├─→ Traditional factors (55%)
    │               ├─→ LSTM-DCF fair value (25%)
    │               └─→ RF expected return (20%)
    │
    └─→ Output: ML-Powered Ranked Watchlist
```

---

## 🔧 Implementation Steps

### Step 1: Enhance WatchlistManagerAgent (30 mins)

```python
# Add to __init__
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.ensemble.rf_ensemble import RFEnsembleModel

self.lstm_model = LSTMDCFModel(input_size=12)
self.lstm_model.load_model("models/lstm_dcf_final.pth")
self.rf_model = RFEnsembleModel()
self.rf_model.load("models/rf_ensemble.pkl")
```

### Step 2: Update SupervisorAgent (20 mins)

```python
from src.agents.enhanced_valuation_agent import EnhancedValuationAgent

self.enhanced_valuation = EnhancedValuationAgent(api_key)

# In orchestrate_stock_analysis_tool
ml_result = self.enhanced_valuation.analyze(
    f"What is the consensus valuation for {ticker}?"
)
results['ml_analysis'] = ml_result
```

### Step 3: Create ML Watchlist Builder (40 mins)

- New file: `scripts/build_ml_watchlist.py`
- CLI interface with argparse
- Progress bar for batch processing
- Export to CSV/JSON

### Step 4: Test Integration (20 mins)

- Test with 5 stocks: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
- Verify ML scores match expected values
- Validate ranking logic

---

## 📈 Expected Improvements

### Before (Current):

```
AAPL Watchlist Score: 72/100
  - Growth: 8.5/10
  - Sentiment: 6.0/10
  - Valuation: 7.2/10
  - Risk: 6.8/10
Signal: BUY
```

### After (ML-Enhanced):

```
AAPL Watchlist Score: 78/100 (+6 points)
  - Traditional Factors: 39.3/55 (71%)
  - LSTM-DCF Fair Value: 19.2/25 (undervalued by 12%)
  - RF Expected Return: 16.8/20 (68th percentile)
  - Consensus Confidence: HIGH (3/4 models agree)
Signal: STRONG BUY (ML-Confirmed)
```

---

## 🎯 Success Criteria

1. ✅ LSTM-DCF model loads successfully in WatchlistManager
2. ✅ RF Ensemble predictions integrate into scoring
3. ✅ Consensus scoring visible in watchlist output
4. ✅ ML-enhanced scores differ from traditional scores by 5-15%
5. ✅ Build watchlist script completes in <2 mins for 10 stocks
6. ✅ All 4 agents coordinated by SupervisorAgent

---

## 📝 Files to Create/Modify

### Create:

- [ ] `scripts/build_ml_watchlist.py` (new)
- [ ] `tests/test_ml_watchlist_integration.py` (new)

### Modify:

- [ ] `src/agents/watchlist_manager_agent.py` (add ML models)
- [ ] `src/agents/supervisor_agent.py` (add EnhancedValuationAgent)
- [ ] `scripts/analyze_stock.py` (add ML analysis section)

---

## 🚀 Ready to Implement?

**Time Estimate**: 2 hours total
**Complexity**: Medium
**Risk**: Low (ML models already tested independently)

Shall we start with Task 1: Enhancing WatchlistManagerAgent?
