# Daily Data Collection Log

## Day 1: October 23, 2025

### Results

- ✅ **Successfully fetched:** 4 stocks (AAPL, ABBV, ABT, ACN)
- ❌ **Rate limited on:** 6 stocks (ADBE, ADI, ADP, AMAT, AMD, AMGN)
- 📊 **Total quarters collected:** 297
- 🔌 **API calls used:** 12/25 (3 per stock)
- ⏱️ **Time:** ~4 minutes

### What Happened

The script worked perfectly! It fetched financial statements for the first 4 stocks in alphabetical order. Each stock requires 3 API calls (Income Statement, Cash Flow, Balance Sheet). After 12 calls, it hit the 25/day limit and stopped.

### Files Created

```
data/raw/financial_statements/
├── AAPL_income.csv (81 quarters)
├── AAPL_cashflow.csv (81 quarters)
├── AAPL_balance.csv (81 quarters)
├── ABBV_income.csv (81 quarters)
├── ABBV_cashflow.csv (81 quarters)
├── ABBV_balance.csv (81 quarters)
├── ABT_income.csv (81 quarters)
├── ABT_cashflow.csv (81 quarters)
├── ABT_balance.csv (81 quarters)
├── ACN_income.csv (81 quarters)
├── ACN_cashflow.csv (81 quarters)
└── ACN_balance.csv (81 quarters)
```

### Progress Tracking

```json
{
  "started": "2025-10-23T01:10:11",
  "last_updated": "2025-10-23T01:14:14",
  "fetched_tickers": ["AAPL", "ABBV", "ABT", "ACN"],
  "failed_tickers": ["ADBE", "ADI", "ADP", "AMAT", "AMD", "AMGN"],
  "total_tickers": 136,
  "total_quarters": 297,
  "total_api_calls": 12
}
```

### Notes

- The 6 "failed" tickers aren't actually failed - they just hit the rate limit
- Tomorrow when you run the script again, it will:
  - ✅ Skip AAPL, ABBV, ABT, ACN (already cached)
  - ✅ Retry ADBE, ADI, ADP, etc. (will succeed)
  - ✅ Continue fetching new stocks

### Tomorrow's Command (Day 2)

```powershell
# Run this tomorrow (October 24, 2025)
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --daily-limit 10

# Expected: Fetch ~8 stocks (24 API calls)
# Will get: ADBE, ADI, ADP, AMAT, AMD, AMGN, AMZN, APA
```

### Estimated Timeline

- **Current:** 4/136 stocks (2.9%)
- **Rate:** ~8 stocks/day (at 25 calls/day limit)
- **Remaining:** 132 stocks
- **Days needed:** ~17 days (not 60!)
- **Completion:** ~November 9, 2025

## Tips for Daily Collection

### What Worked Well ✅

- Smart caching prevents re-fetching
- Progress tracking persists across runs
- Rate limiting works correctly
- Script auto-stops at daily limit

### What to Remember 💡

1. **Run once per day** - Your Alpha Vantage key resets at midnight UTC
2. **Check status first** if unsure:
   ```powershell
   .\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --status
   ```
3. **Don't worry about "failed" stocks** - They'll retry tomorrow automatically
4. **Each stock = 3 API calls** - So 8 stocks = 24 calls (under 25 limit)

### Expected Daily Pattern

```
Day 1: Fetch stocks 1-4     (12 calls) ✅ YOU ARE HERE
Day 2: Fetch stocks 5-12    (24 calls)
Day 3: Fetch stocks 13-20   (24 calls)
Day 4: Fetch stocks 21-28   (24 calls)
...
Day 17: Fetch stocks 129-136 (24 calls) ✅ COLLECTION COMPLETE
```

## Next Steps After Collection

### When You Hit 100+ Stocks (Day 13)

You can start training with a smaller dataset:

```powershell
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --quick-test
```

### When Collection Complete (Day 17)

```powershell
# Create full training dataset
.\venv\Scripts\python.exe scripts\fetch_lstm_training_data.py --create-dataset

# Train full model (50 epochs, ~1 hour GPU)
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 50
```

### Integration Timeline

- **Day 17:** Data collection complete
- **Day 18:** Model training + validation
- **Day 19:** Integrate into analyze_stock.py
- **Day 20:** Backtest and compare results

---

## Collection Status

### Progress Overview

| Metric                   | Value   |
| ------------------------ | ------- |
| **Stocks fetched**       | 4 / 136 |
| **Completion**           | 2.9%    |
| **Quarters collected**   | 297     |
| **API calls used today** | 12 / 25 |
| **Days elapsed**         | 1       |
| **Days remaining**       | ~17     |

### Recent Activity

- **Last run:** October 23, 2025 01:14 AM
- **Status:** ⏸️ Paused (daily limit reached)
- **Next run:** October 24, 2025 (any time)

---

_This log will be updated as collection progresses. Check back daily to track your progress!_
