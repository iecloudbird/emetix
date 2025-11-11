# Enhanced LSTM Training Data Collection Guide

## Multi-Source Data Pipeline (Alpha Vantage + Finnhub)

---

## 🎯 Overview

The enhanced data collection system combines **Alpha Vantage** and **Finnhub APIs** to overcome data limitations and expand your training dataset.

### Key Improvements

| Feature             | Original System     | Enhanced System         |
| ------------------- | ------------------- | ----------------------- |
| **APIs**            | Alpha Vantage only  | Alpha Vantage + Finnhub |
| **Daily Limit**     | 25 stocks/day       | 100+ stocks/day         |
| **Rate Limit**      | 5 calls/min         | 60 calls/min (Finnhub)  |
| **Ticker Universe** | 136 stocks          | 200+ stocks             |
| **Redundancy**      | Single source       | Multi-source fallback   |
| **Speed**           | 17 days to complete | 2-3 days to complete    |

---

## 📦 New Components

### 1. FinnhubFinancialsFetcher (`src/data/fetchers/finnhub_financials.py`)

- Fetches quarterly financial statements from Finnhub
- Rate limit: 60 calls/minute, no daily limit
- Same data format as Alpha Vantage fetcher
- Automatic caching to `data/raw/finnhub_financials/`

### 2. UnifiedFinancialsFetcher (`src/data/fetchers/unified_financials_fetcher.py`)

- Intelligent multi-source fetching strategy
- Automatic fallback: Alpha Vantage → Finnhub
- Smart API usage: respects both rate limits
- Source tracking for data provenance

### 3. Enhanced Collection Script (`scripts/fetch_enhanced_training_data.py`)

- Replaces `fetch_lstm_training_data.py` with multi-source support
- Expanded ticker universe (200+ stocks)
- Flexible batch sizes (50-100+ stocks per run)
- Progress tracking with source statistics

---

## 🚀 Quick Start

### Step 1: Verify API Keys

Check your `.env` file has both API keys:

```bash
# Alpha Vantage API (25 calls/day)
ALPHA_VANTAGE_API_KEY=WLC4IX6W15DTMC1S

# Finnhub API (60 calls/min, unlimited daily)
FINNHUB_API_KEY=d3j9i71r01qkv9jumut0d3j9i71r01qkv9jumutg
```

### Step 2: Test Both APIs

```powershell
# Test Finnhub
.\venv\Scripts\python.exe src\data\fetchers\finnhub_financials.py

# Test Unified Fetcher
.\venv\Scripts\python.exe src\data\fetchers\unified_financials_fetcher.py
```

### Step 3: Fetch Enhanced Dataset

```powershell
# Option A: Use Finnhub primarily (RECOMMENDED for speed)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 50 --prefer-finnhub

# Option B: Use Alpha Vantage primarily (better quality, slower)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 25 --prefer-alpha-vantage

# Option C: Balanced approach (mixed sources)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 40
```

### Step 4: Create Combined Dataset

```powershell
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --create-dataset
```

This creates: `data/processed/lstm_dcf_training/lstm_growth_training_data_enhanced.csv`

---

## 📖 Usage Examples

### Example 1: Quick Collection (100+ stocks in one run)

```powershell
# Fetch 100 stocks using Finnhub (no daily limit)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py `
    --batch-size 100 `
    --prefer-finnhub

# Expected: ~100 stocks fetched in 10-15 minutes
```

### Example 2: High-Quality Collection (Alpha Vantage priority)

```powershell
# Fetch 25 stocks using Alpha Vantage (better quality)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py `
    --batch-size 25 `
    --prefer-alpha-vantage `
    --alpha-vantage-limit 25

# Expected: 25 stocks from Alpha Vantage
# Run daily to collect more
```

### Example 3: Mixed Strategy (Best of Both)

```powershell
# Use Alpha Vantage for first 25, then Finnhub for rest
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py `
    --batch-size 50 `
    --alpha-vantage-limit 25

# Expected: 25 from Alpha Vantage, 25 from Finnhub
```

### Example 4: Compare Original vs Enhanced

```powershell
# Original data (86 stocks)
Get-Content data\processed\lstm_dcf_training\lstm_growth_training_data.csv | Measure-Object -Line

# Enhanced data (will be 100-200+ stocks)
Get-Content data\processed\lstm_dcf_training\lstm_growth_training_data_enhanced.csv | Measure-Object -Line
```

---

## 🔄 Data Collection Strategy

### Recommended Workflow

```
Day 1: Initial Bulk Collection
├── Run: --batch-size 100 --prefer-finnhub
├── Expected: 80-100 stocks from Finnhub
└── Time: 15-20 minutes

Day 2: Fill Gaps with Alpha Vantage
├── Run: --batch-size 25 --prefer-alpha-vantage
├── Expected: 25 high-quality stocks
└── Time: 5 minutes

Day 3: Final Sweep
├── Run: --batch-size 50 --prefer-finnhub
├── Expected: Remaining stocks
└── Time: 10 minutes

Result: 150-175 stocks in 3 days vs 17 days with original
```

### Intelligent Source Selection

The unified fetcher automatically:

1. **Checks cache first** (instant, no API calls)
2. **Tries preferred source** (Alpha Vantage or Finnhub)
3. **Falls back to alternate** if first fails
4. **Tracks source usage** for transparency

```python
# Example: Automatic fallback logic
ticker = "AAPL"
1. Try Alpha Vantage → ✅ Success (81 quarters)
   Return: data from Alpha Vantage

ticker = "NEWCO"
1. Try Alpha Vantage → ❌ Failed (no data)
2. Try Finnhub → ✅ Success (45 quarters)
   Return: data from Finnhub

ticker = "BADCO"
1. Try Alpha Vantage → ❌ Failed
2. Try Finnhub → ❌ Failed
   Return: None (marked as failed)
```

---

## 📊 Expected Outcomes

### Original System (Alpha Vantage only)

- **Stocks fetched:** 86/136 (63%)
- **Quarters:** 6,501
- **Records:** 6,635
- **Time:** 13+ days
- **Failed:** 56 stocks

### Enhanced System (Alpha Vantage + Finnhub)

- **Stocks fetched:** 150-200/200 (75-100%)
- **Quarters:** 12,000-15,000+
- **Records:** 12,000-15,000+
- **Time:** 2-3 days
- **Failed:** 10-20 stocks

### Data Quality Comparison

| Metric    | Original      | Enhanced     | Improvement |
| --------- | ------------- | ------------ | ----------- |
| Stocks    | 86            | 150-200      | +74-133%    |
| Quarters  | 6,501         | 12,000+      | +85%        |
| Records   | 6,635         | 12,000+      | +81%        |
| Coverage  | 63%           | 90%+         | +43%        |
| Diversity | Single source | Multi-source | Better      |

---

## 🔍 Progress Tracking

### Check Collection Status

```powershell
# View enhanced progress
Get-Content data\processed\lstm_dcf_training\enhanced_fetch_progress.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Progress File Structure

```json
{
  "started": "2025-11-07T...",
  "last_updated": "2025-11-07T...",
  "fetched_tickers": ["AAPL", "MSFT", ...],
  "failed_tickers": ["BRK.B", ...],
  "source_stats": {
    "Alpha Vantage": 25,
    "Finnhub": 125
  },
  "total_tickers": 200,
  "total_quarters": 12500
}
```

---

## ⚙️ Command Reference

### Basic Commands

```powershell
# Fetch with default settings (50 stocks, balanced)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py

# Fetch large batch with Finnhub (fast)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 100 --prefer-finnhub

# Fetch small batch with Alpha Vantage (quality)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 25 --prefer-alpha-vantage

# Create combined dataset
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --create-dataset
```

### Advanced Options

```powershell
# Custom Alpha Vantage limit
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py `
    --batch-size 50 `
    --alpha-vantage-limit 20

# Force Finnhub only (skip Alpha Vantage)
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py `
    --batch-size 100 `
    --prefer-finnhub `
    --alpha-vantage-limit 0
```

---

## 🎓 Training with Enhanced Dataset

### Step 1: Create Enhanced Dataset

```powershell
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --create-dataset
```

### Step 2: Update Training Script

The dataset has an extra `source` column. Update your training to handle it:

```python
# Load enhanced dataset
df = pd.read_csv('data/processed/lstm_dcf_training/lstm_growth_training_data_enhanced.csv')

# Drop source column before training (or keep for analysis)
df_train = df.drop(columns=['source'])

# Or analyze by source
print(df.groupby('source')['ticker'].nunique())
```

### Step 3: Train LSTM with More Data

```powershell
# Train with enhanced dataset (12K+ records vs 6.6K)
.\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 50

# Expected improvements:
# - Better generalization (more diverse data)
# - Lower overfitting risk (larger dataset)
# - Improved R² score
# - More robust predictions
```

---

## 🐛 Troubleshooting

### Issue 1: Finnhub API Key Error

```
ValueError: FINNHUB_API_KEY not found
```

**Solution:**

```powershell
# Check .env file
Get-Content .env | Select-String "FINNHUB"

# Should show:
# FINNHUB_API_KEY=d3j9i71r01qkv9jumut0d3j9i71r01qkv9jumutg
```

### Issue 2: Rate Limit Hit

```
Rate limit: waiting 60.0s...
```

**Solution:** This is normal! The system automatically waits and retries.

### Issue 3: Some Tickers Fail

```
❌ All sources failed for BRK.B
```

**Solution:** Some tickers (like BRK.B) have special formats that APIs don't support. This is expected. The system marks them as failed and continues.

### Issue 4: No Improvement Over Original

**Solution:** Delete old cache and re-fetch:

```powershell
# Clear caches
Remove-Item data\raw\finnhub_financials\* -Recurse -Force
Remove-Item data\processed\lstm_dcf_training\enhanced_fetch_progress.json

# Re-fetch
.\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 100 --prefer-finnhub
```

---

## 📈 Performance Benchmarks

### API Speed Comparison

| API           | Calls/Min | Calls/Day | Stock/Day | Time for 100 stocks |
| ------------- | --------- | --------- | --------- | ------------------- |
| Alpha Vantage | 5         | 25        | ~8        | 12.5 days           |
| Finnhub       | 60        | Unlimited | ~200      | 0.5 days            |
| **Combined**  | **60**    | **~225**  | **~75**   | **1.3 days**        |

### Data Quality Metrics

- **Alpha Vantage:** 81 quarters average, high quality, standardized
- **Finnhub:** 40-50 quarters average, varies by company, real-time
- **Combined:** Best of both, fills gaps, higher coverage

---

## ✅ Success Checklist

After enhanced collection, you should have:

- [ ] 150+ stocks fetched (vs 86 originally)
- [ ] 12,000+ training records (vs 6,635)
- [ ] Multi-source data (Alpha Vantage + Finnhub)
- [ ] Enhanced dataset created: `lstm_growth_training_data_enhanced.csv`
- [ ] Progress file shows source distribution
- [ ] Failed tickers < 20 (vs 56 originally)
- [ ] Ready to train improved LSTM model

---

## 🎯 Next Steps

1. **Collect Enhanced Data:**

   ```powershell
   .\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --batch-size 100 --prefer-finnhub
   ```

2. **Create Dataset:**

   ```powershell
   .\venv\Scripts\python.exe scripts\fetch_enhanced_training_data.py --create-dataset
   ```

3. **Compare Datasets:**

   ```powershell
   # Original
   .\venv\Scripts\python.exe scripts\inspect_dataset.py

   # Enhanced (update script to use enhanced dataset)
   ```

4. **Train with Enhanced Data:**

   ```powershell
   # Update train_lstm_growth_forecaster.py to use enhanced dataset
   # Then train:
   .\venv\Scripts\python.exe scripts\train_lstm_growth_forecaster.py --epochs 50
   ```

5. **Evaluate Improvements:**
   - Compare validation loss
   - Check test R² score
   - Analyze per-component accuracy
   - Test on unseen stocks

---

## 📚 Additional Resources

- **Finnhub API Docs:** https://finnhub.io/docs/api
- **Alpha Vantage Docs:** https://www.alphavantage.co/documentation/
- **Original Guide:** `docs/DAILY_COLLECTION_LOG.md`
- **ML Pipelines:** `docs/ML_PIPELINES_ARCHITECTURE.md`

---

**Ready to expand your training dataset?** Run the enhanced fetcher now! 🚀
