# LSTM-DCF Implementation Plan: Growth Rate Forecasting Approach

## Overview

Based on the article: [LSTM Networks for estimating growth rates in DCF Models](https://www.revocm.com/articles/lstm-networks-estimating-growth-rates-dcf-models)

**Current Status:** ✅ Architecture designed and implemented  
**Next Steps:** Training data collection + model training  
**Expected Improvement:** Sharpe ratio from 0.55 to 0.75 (article results)

---

## The Proper LSTM-DCF Approach

### ❌ What We Had Before (INCORRECT)

- LSTM trained to predict **stock closing prices**
- Treating price predictions as FCFF (Free Cash Flow to Firm)
- Result: $0.00 valuations due to scale mismatch
- Quick fix: Hybrid weighting based on FCF reliability

### ✅ What We Should Have (CORRECT - Article Approach)

- LSTM trained to predict **growth rates** of DCF components:
  - Revenue growth
  - CapEx growth
  - Depreciation & Amortization growth
  - NOPAT growth (Net Operating Profit After Tax)
- Use forecasted growth rates to project FCFF
- Calculate DCF from projected cash flows

---

## Architecture Comparison

### Article's FCFF Formula

```
FCFF = NOPAT + D&A - CapEx - ΔNWC

Where:
- NOPAT = EBIT × (1 - Tax Rate)
- LSTM forecasts growth rates for: Revenue, CapEx, D&A, NOPAT
- Year 1: LSTM forecast
- Years 2-5: Linear interpolation to industry average
```

### Our Implementation

**Files Created:**

1. **`src/models/deep_learning/lstm_growth_forecaster.py`** (354 lines)

   - `LSTMGrowthForecaster`: Neural network for growth rate prediction
   - `DCFValuationWithLSTM`: Complete valuation pipeline
   - Input: 4 features (Revenue, CapEx, D&A, NOPAT) normalized by assets
   - Output: 4 growth rates for next period

2. **`src/data/fetchers/financial_statements_fetcher.py`** (343 lines)
   - `FinancialStatementsFetcher`: Fetches quarterly financial data
   - Extracts from yfinance:
     - Income Statement: Revenue, EBIT, Tax Rate
     - Cash Flow: CapEx, Depreciation & Amortization
     - Balance Sheet: Total Assets (for normalization)
   - Preprocessing: Asset normalization + standardization (mean=0, std=1)

---

## Training Data Pipeline

### Data Requirements (Per Article)

- **Minimum:** 5 years of quarterly data (20 quarters) per stock
- **Normalization:** Metrics / Total Assets (accounts for company scale)
- **Standardization:** (Value - Mean) / Std (mean=0, std=1)
- **Sequence Length:** 20 quarters (~5 years lookback)
- **Target:** Next quarter's growth rates

### Current Limitation ⚠️

**yfinance only provides 4-5 quarters** of quarterly financial data (API limitation).

### Solutions:

#### Option 1: Alpha Vantage API (FREE)

```python
# Already have API key in .env
from alpha_vantage.fundamentaldata import FundamentalData
fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Get quarterly income statement (5+ years)
income, _ = fd.get_income_statement_quarterly(ticker)

# Get quarterly cash flow
cashflow, _ = fd.get_cash_flow_quarterly(ticker)

# Get quarterly balance sheet
balance, _ = fd.get_balance_sheet_quarterly(ticker)
```

**Pros:**

- ✅ FREE tier: 25 calls/day
- ✅ Full 5-year quarterly data
- ✅ We already have API key

**Cons:**

- ❌ Rate limited (need batch processing)
- ❌ Need 3 API calls per stock

#### Option 2: Financial Modeling Prep API

- **Paid:** $14/month for 250 calls/day
- Provides 10+ years of quarterly financials
- Single endpoint for all statements

#### Option 3: Use Annual Data (Compromise)

- yfinance provides 4 years of annual financials
- Trade frequency for history
- Less accurate but feasible

---

## Recommended Implementation Path

### Phase 1: Data Collection (1-2 days)

```bash
# Create script: scripts/fetch_financial_statements.py

python scripts/fetch_financial_statements.py --tickers SP500 --source alpha_vantage
```

**Tasks:**

1. Implement Alpha Vantage financial statements fetcher
2. Fetch quarterly data for S&P 500 stocks (25/day limit → 20 days for 500 stocks)
3. Store in `data/raw/financial_statements/`
4. Preprocessed data → `data/processed/lstm_dcf_training/`

**Expected Output:**

- CSV with columns: `[ticker, date, revenue_std, capex_std, da_std, nopat_std]`
- ~50,000 records (500 stocks × 20 quarters × 5 metrics)

### Phase 2: Model Training (1 day)

```bash
# Create script: scripts/train_lstm_growth_forecaster.py

python scripts/train_lstm_growth_forecaster.py --epochs 50 --batch_size 64
```

**Training Configuration:**

```yaml
# config/model_config.yaml - Add lstm_growth section

lstm_growth:
  model:
    input_size: 4 # Revenue, CapEx, D&A, NOPAT
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    output_size: 4 # Growth rates for each

  training:
    batch_size: 64
    learning_rate: 0.001
    epochs: 50
    sequence_length: 20 # 5 years of quarterly data
    validation_split: 0.2

  evaluation:
    metrics: ["MSE", "MAE", "R2"] # Per article's table
```

**Expected Performance (from article):**
| Metric | MSE | MAE | R² |
|--------|-----|-----|-----|
| Revenue | 0.012 | 0.085 | 0.45 |
| CapEx | 0.018 | 0.105 | 0.38 |
| D&A | 0.015 | 0.095 | 0.42 |

### Phase 3: Integration into analyze_stock.py (1 day)

**Replace current LSTM section with:**

```python
# In analyze_stock.py, Section 4 (ML-Powered Valuation)

from src.models.deep_learning.lstm_growth_forecaster import (
    LSTMGrowthForecaster,
    DCFValuationWithLSTM
)
from src.data.fetchers.financial_statements_fetcher import FinancialStatementsFetcher

# Initialize
growth_model = LSTMGrowthForecaster(input_size=4, hidden_size=64, num_layers=2)
growth_model.load_model(str(MODELS_DIR / "lstm_growth_forecaster.pth"))

fin_fetcher = FinancialStatementsFetcher()
dcf_valuation = DCFValuationWithLSTM(growth_model)

# Get financial data
fin_data = fin_fetcher.prepare_training_data(ticker, min_quarters=20)
current_metrics = fin_fetcher.fetch_current_metrics(ticker)

if fin_data and current_metrics:
    # Prepare historical sequence for LSTM
    historical_sequence = fin_data['standardized_data'][
        ['revenue_norm_std', 'capex_norm_std', 'da_norm_std', 'nopat_norm_std']
    ].values[-20:]  # Last 20 quarters

    # Full LSTM-DCF valuation
    valuation_result = dcf_valuation.full_valuation(
        historical_data=historical_sequence,
        current_metrics=current_metrics,
        wacc=0.08,
        terminal_growth=0.03,
        shares_outstanding=current_metrics['shares_outstanding'],
        projection_years=5
    )

    # Display results
    print(f"\n🧠 LSTM-DCF Growth Rate Forecasting:")
    print(f"{'─'*80}")
    print(f"\n💰 Valuation Results:")
    print(f"   Current Market Price: ${current_price:,.2f}")
    print(f"   LSTM-DCF Fair Value:  ${valuation_result['fair_value_per_share']:,.2f}")

    gap = ((valuation_result['fair_value_per_share'] - current_price) / current_price) * 100
    print(f"   Valuation Gap:        {gap:+.2f}%")

    print(f"\n📈 5-Year FCFF Projections (LSTM Growth Rates):")
    for year, fcff in enumerate(valuation_result['fcff_projections'], 1):
        print(f"   Year {year}: ${fcff/1e9:,.2f}B")

    print(f"\n📊 Forecasted Growth Rates:")
    for key, trajectory in valuation_result['growth_trajectories'].items():
        print(f"   {key}:")
        print(f"     Year 1 (LSTM): {trajectory[0]*100:+.2f}%")
        print(f"     Year 5 (→Industry): {trajectory[-1]*100:+.2f}%")

    print(f"\n💎 DCF Components:")
    print(f"   Enterprise Value:     ${valuation_result['enterprise_value']/1e9:,.2f}B")
    print(f"   PV of 5Y FCFF:        ${valuation_result['pv_fcff']/1e9:,.2f}B")
    print(f"   Terminal Value (PV):  ${valuation_result['pv_terminal']/1e9:,.2f}B")
```

---

## Expected Improvements

### From Article Results:

**Backtest Performance (2015-2022):**
| Strategy | Sharpe | CAGR | Volatility |
|----------|--------|------|------------|
| S&P 500 | 0.55 | 10.8% | 19.7% |
| **LSTM-DCF** | **0.75** | **13.8%** | **18.5%** |

**Key Insight:**

> "By achieving more precise growth rate estimates for the Free Cash Flow components, we gain a deeper insight into a company's financial trajectory over subsequent years."

### Our Expected Benefits:

1. **More Accurate Valuations:**

   - Growth stocks (TSLA, high-growth tech): LSTM captures growth dynamics better than historical averages
   - Mature stocks (AAPL, KO): LSTM + industry convergence provides realistic long-term projections

2. **Better Risk-Adjusted Returns:**

   - Sharpe ratio improvement: +36% (0.55 → 0.75)
   - Lower volatility despite higher CAGR
   - More informed buy/sell decisions

3. **Proper DCF Fundamentals:**
   - Growth rates → FCFF projections → Enterprise value → Fair value per share
   - No more hybrid price/DCF weighting hacks
   - Methodologically sound approach from finance literature

---

## Timeline Summary

### Quick Path (Using Alpha Vantage - FREE)

- **Day 1-20:** Fetch financial data (25 stocks/day limit)
- **Day 21:** Preprocess and create training sequences
- **Day 22:** Train LSTM growth forecaster
- **Day 23:** Integrate into analyze_stock.py
- **Day 24:** Test and validate

### Fast Path (Using Premium API - $14/month)

- **Day 1:** Fetch all data (no rate limits)
- **Day 2:** Train model
- **Day 3:** Integrate and test

---

## Technical Considerations

### Why This Approach is Better:

1. **Theoretically Sound:**

   - Follows established finance research
   - Published article with backtested results
   - Used by professional quant funds

2. **Captures Complex Patterns:**

   - LSTM learns temporal dependencies in growth
   - Adapts to changing business cycles
   - Outperforms simple historical averages (article's Table 1)

3. **Handles Edge Cases:**

   - Growth stocks: High revenue growth, low current FCF
   - Turnarounds: Improving metrics over time
   - Cyclical: Industry-specific patterns

4. **Interpretable:**
   - Growth rates are understandable metrics
   - Can explain to users: "LSTM forecasts 8% revenue growth"
   - Validates against industry benchmarks

### Current Hybrid Approach (Temporary):

The current smart-weighting system (20%-80% DCF/LSTM based on FCF) is a **pragmatic workaround** but:

- ❌ Still uses LSTM for price prediction (not its strength)
- ❌ Mixes methodologies (price momentum + DCF)
- ❌ Hard to interpret: "Why 70/30 for this stock?"
- ✅ Works as a stopgap solution
- ✅ Better than fixed 70/30 weighting

---

## Decision Point

**Option A: Keep Current Hybrid (Quick)**

- Pros: Already working, no additional data collection
- Cons: Methodologically impure, suboptimal accuracy
- Time: 0 days (done)

**Option B: Implement Proper LSTM-DCF (Best)**

- Pros: Theoretically sound, proven results (+36% Sharpe), professional-grade
- Cons: Requires financial statements data, 3-4 days work
- Time: 3-24 days (depending on API choice)

**Recommendation:** **Option B** - The improvements in accuracy and Sharpe ratio justify the implementation effort. This is what serious quant funds use.

---

## Next Steps

If you want to proceed with proper LSTM-DCF:

1. **Immediate:** Choose data source (Alpha Vantage free vs. paid API)
2. **Week 1:** Implement financial statements fetcher for chosen source
3. **Week 2:** Collect training data (parallel with other work)
4. **Week 3:** Train LSTM growth forecaster
5. **Week 4:** Integrate and validate

Let me know if you want me to:

- [ ] Implement Alpha Vantage financial statements fetcher
- [ ] Create training script for LSTM growth forecaster
- [ ] Keep current hybrid approach for now
- [ ] Research other data sources

---

**Reference:** https://www.revocm.com/articles/lstm-networks-estimating-growth-rates-dcf-models

**Status:** Architecture complete, awaiting data source decision for training.
