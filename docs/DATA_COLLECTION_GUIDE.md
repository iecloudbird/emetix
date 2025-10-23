# LSTM-DCF Training Data Collection Guide

## Quick Start

### 1. Start Daily Collection (Recommended)

```powershell
# Fetch 10 stocks per day (uses ~30 API calls)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --daily-limit 10
```

**Schedule:** Run this command daily for 60 days to collect 500+ stocks.

---

## Overview

This system collects 20 years of quarterly financial statements from Alpha Vantage for training the LSTM Growth Forecaster model.

### What Gets Collected

- **Quarterly Income Statements** (Revenue, EBIT, Tax Rate)
- **Quarterly Cash Flow Statements** (CapEx, Depreciation & Amortization)
- **Quarterly Balance Sheets** (Total Assets for normalization)

### Data Coverage

- **81 quarters** per stock (~20 years: 2005-2025)
- **500+ stocks** from NYSE + S&P 500
- **~30,500 training sequences** after processing

---

## Commands

### Daily Collection

```powershell
# Standard daily fetch (10 stocks)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --daily-limit 10

# Conservative fetch (5 stocks, if rate limited)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --daily-limit 5

# Aggressive fetch (use remaining daily quota)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --daily-limit 8
```

### Specific Stocks

```powershell
# Fetch specific tickers
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --tickers AAPL MSFT GOOGL TSLA

# Re-fetch failed stocks
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --tickers TICKER1 TICKER2
```

### Progress Monitoring

```powershell
# Check collection status
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --status

# Create combined dataset (run after 500+ stocks collected)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset
```

---

## Rate Limits (Alpha Vantage FREE Tier)

| Limit                  | Value   | Impact                                  |
| ---------------------- | ------- | --------------------------------------- |
| **Calls per day**      | 25      | Fetch 8 stocks/day (3 calls each)       |
| **Calls per minute**   | 5       | Auto-throttled, no manual action needed |
| **Total calls needed** | 1,500   | 500 stocks × 3 calls each               |
| **Collection time**    | 60 days | At 8-10 stocks/day                      |

### Smart Features

- ✅ **Smart caching:** Already-fetched stocks are skipped
- ✅ **Progress tracking:** Resume anytime, no data loss
- ✅ **Rate limit handling:** Auto-throttles to avoid API errors
- ✅ **Graceful errors:** Failed stocks tracked separately

---

## Collection Progress

The script maintains progress in:

```
data/processed/lstm_dcf_training/fetch_progress.json
```

### Progress File Structure

```json
{
  "started": "2025-10-23T10:00:00",
  "last_updated": "2025-10-23T15:30:00",
  "fetched_tickers": ["AAPL", "MSFT", "GOOGL", ...],
  "failed_tickers": ["TICKER1", "TICKER2"],
  "total_tickers": 150,
  "total_quarters": 12150,
  "total_api_calls": 450
}
```

### Cached Data Location

```
data/raw/financial_statements/
├── AAPL_income.csv          # 81 quarters income statement
├── AAPL_cashflow.csv        # 81 quarters cash flow
├── AAPL_balance.csv         # 81 quarters balance sheet
├── MSFT_income.csv
├── MSFT_cashflow.csv
└── ...
```

---

## Training Data Preparation

### After Collection Complete (500+ stocks)

1. **Create Combined Dataset**

   ```powershell
   .\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset
   ```

   **Output:** `data/processed/lstm_dcf_training/lstm_growth_training_data.csv`

   **Format:**

   ```csv
   ticker,date,revenue_std,capex_std,da_std,nopat_std
   AAPL,2005-06-30,1.568991,-0.794096,-1.090826,-1.161641
   AAPL,2005-09-30,1.279117,-0.285750,-1.077878,-0.896929
   ...
   ```

2. **Train LSTM Model**

   ```powershell
   # Full training (50 epochs, ~1 hour GPU)
   .\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 50

   # Quick test (5 epochs, ~5 minutes)
   .\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --quick-test
   ```

   **Output:** `models/lstm_growth_forecaster.pth`

---

## Expected Timeline

### Recommended Schedule (10 stocks/day)

```
Week 1 (Day 1-7):     70 stocks   →  143 days remaining
Week 2 (Day 8-14):   140 stocks   →  117 days remaining
Week 3 (Day 15-21):  210 stocks   →   91 days remaining
Week 4 (Day 22-28):  280 stocks   →   65 days remaining
Week 5 (Day 29-35):  350 stocks   →   39 days remaining
Week 6 (Day 36-42):  420 stocks   →   13 days remaining
Week 7 (Day 43-49):  490 stocks   →    0 days remaining
Week 8 (Day 50-56):  500+ stocks  →   COLLECTION COMPLETE ✅
Week 9 (Day 57):     Training    →   Model ready ✅
```

### Conservative Schedule (5 stocks/day)

- **120 days** to collect 500 stocks
- Still fits 30-week FYP timeline (210 days)

---

## Troubleshooting

### "Daily API call limit reached"

**Cause:** Exceeded 25 API calls in 24 hours  
**Solution:** Wait until tomorrow, or use `--daily-limit 5` to be more conservative

### "Rate limit: waiting 60s..."

**Cause:** Exceeded 5 calls/minute  
**Action:** This is normal! Script auto-throttles, no action needed

### Stock shows "Insufficient data"

**Cause:** Stock has <20 quarters of data (new IPO, etc.)  
**Action:** Normal, script skips and marks as failed

### Want to re-fetch a failed stock

```powershell
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --tickers TICKER_NAME
```

### Check what failed

```powershell
# View progress file
cat data\processed\lstm_dcf_training\fetch_progress.json | jq .failed_tickers
```

---

## Stock Universe

### S&P 500 (Top 100)

AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, UNH, XOM, JNJ, JPM, V, PG, MA, HD, CVX, MRK, ABBV, PEP, KO, AVGO, COST, LLY, TMO, WMT, MCD, ACN, CSCO, ABT, DHR, VZ, ADBE, NKE, CRM, TXN, CMCSA, ORCL, INTC, DIS, PM, NEE, BMY, UPS, T, QCOM, RTX, HON, LOW, UNP, INTU, AMD, SPGI, BA, COP, SBUX, AMAT, DE, GS, CAT, AMGN, AXP, BKNG, LMT, PLD, GILD, MDLZ, ADI, TJX, SYK, MMC, ADP, CI, VRTX, BDX, CVS, ZTS, TMUS, CB, SO, PGR, ISRG, DUK, REGN, MO, NOC, SLB, ITW, EOG, MMM, CL, BSX, APD, EQIX, GE, SCHW, PNC, USB, BLK, CME

### NYSE (Top 50)

BAC, WFC, C, GS, MS, AXP, USB, PNC, TFC, COF, BK, STT, FITB, KEY, RF, CFG, HBAN, CMA, ZION, SIVB, F, GM, TSLA, RIVN, LCID, NIO, XPEV, LI, FSR, WKHS, T, VZ, TMUS, S, DISH, CHTR, CMCSA, LUMN, VOD, TEF, XOM, CVX, COP, SLB, HAL, MRO, DVN, APA, EOG, HES

**Total:** 150 unique tickers

---

## Data Quality

### What Makes Alpha Vantage Better?

**yfinance:**

- ❌ Only 5 quarters available
- ❌ Cannot create 20-quarter sequences
- ❌ Insufficient for LSTM training

**Alpha Vantage:**

- ✅ 81 quarters (20 years)
- ✅ Creates 61 sequences per stock
- ✅ 30,500+ training examples from 500 stocks
- ✅ Covers multiple economic cycles
- ✅ Better pattern learning

### Data Preprocessing

**Normalization by Assets:**

```python
revenue_norm = revenue / total_assets
capex_norm = capex / total_assets
da_norm = da / total_assets
nopat_norm = nopat / total_assets
```

**Standardization (mean=0, std=1):**

```python
revenue_std = (revenue_norm - mean) / std
# Ready for neural network input
```

---

## Next Steps After Collection

1. **Day 60:** Collection complete (500+ stocks)
2. **Day 61:** Create combined dataset
3. **Day 62:** Train LSTM Growth Forecaster (~1 hour GPU)
4. **Day 63:** Integrate into `analyze_stock.py`
5. **Day 64+:** Validate and backtest

### Expected Model Performance

Based on research article:
| Metric | S&P 500 | LSTM-DCF | Improvement |
|--------|---------|----------|-------------|
| Sharpe Ratio | 0.55 | 0.75 | +36% |
| CAGR | 10.8% | 13.8% | +28% |
| Volatility | 19.7% | 18.5% | -6% |

---

## Integration with analyze_stock.py

Once trained, the model will be used in Section 4 (ML-Powered Valuation):

```python
# Load LSTM Growth Forecaster
from src.models.deep_learning.lstm_growth_forecaster import (
    LSTMGrowthForecaster,
    DCFValuationWithLSTM
)

# Forecast growth rates
growth_rates = model.forecast_growth_rates(historical_data)
# {'revenue_growth': 0.08, 'capex_growth': 0.05, 'da_growth': 0.03, 'nopat_growth': 0.07}

# Project FCFF and calculate DCF
valuation = dcf.full_valuation(
    historical_data,
    current_metrics,
    projection_years=5
)
# Fair value per share: $175.50
```

---

## Questions?

See also:

- `docs/LSTM_DCF_IMPLEMENTATION_PLAN.md` - Detailed methodology
- `docs/ALPHA_VANTAGE_VS_YFINANCE.md` - Data source comparison
- `docs/PROGRESS_SUMMARY.md` - Overall project status

**Research Article:** https://www.revocm.com/articles/lstm-networks-estimating-growth-rates-dcf-models
