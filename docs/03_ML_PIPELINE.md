# 3. Machine Learning Pipeline

> **LSTM-DCF + Transparent GARP Scoring**

---

## 🎯 ML Strategy Overview

Emetix uses a **simplified, transparent scoring approach** for stock valuation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONSENSUS SCORING SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │   LSTM-DCF     │  │  GARP Score    │  │  Risk Score    │     │
│  │   Fair Value   │  │ (Forward P/E   │  │  (Beta + Vol)  │     │
│  │                │  │  + PEG Ratio)  │  │                │     │
│  │   Weight:      │  │   Weight:      │  │   Weight:      │     │
│  │     50%        │  │     25%        │  │     25%        │     │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘     │
│          │                   │                   │               │
│          └───────────────────┼───────────────────┘               │
│                              ▼                                   │
│                 ┌─────────────────────────┐                      │
│                 │   Weighted Consensus    │                      │
│                 │   + MoS Penalty         │                      │
│                 └─────────────────────────┘                      │
│                              │                                   │
│                              ▼                                   │
│                 ┌─────────────────────────┐                      │
│                 │  Valuation Score 0-100  │                      │
│                 │  + Recommendation       │                      │
│                 └─────────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

> **Architecture Shift (Jan 2025)**: RF Ensemble was found to use P/E ratio
> at 99.93% importance, making it a redundant black-box. Replaced with
> transparent GARP scoring (Forward P/E + PEG) for explainability.

---

## 📊 Model 1: LSTM-DCF (Deep Learning)

### Purpose

Predict **fair value** by forecasting FCF growth rates using time-series patterns.

### Architecture (V2 - Jan 2026)

The current production model is **LSTM-DCF v2** with improved architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM-DCF MODEL V2                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Layer                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Sequence: 8 quarters (2 years) × 16 features            │    │
│  │ Features: revenue, fcf, ebitda, margins, growth rates   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Input Normalization (NEW in v2)                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ BatchNorm1d(16) - Stabilizes input distribution         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  LSTM Layers (2 layers)                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 1: LSTM(input=16, hidden=128, dropout=0.3)        │    │
│  │ Layer 2: LSTM(hidden=128, dropout=0.3)                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Fully Connected                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FC1: Linear(128 → 64) + ReLU + Dropout(0.3)             │    │
│  │ FC2: Linear(64 → 2) → [Revenue Growth, FCF Growth]      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Output: Predicted growth rates → DCF Fair Value                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### V2 Architecture Changes (Jan 2026)

| Parameter           | V1                | V2              | Rationale                            |
| ------------------- | ----------------- | --------------- | ------------------------------------ |
| Sequence Length     | 60 quarters       | 8 quarters      | Recent trends, more training samples |
| LSTM Layers         | 3                 | 2               | Reduced overfitting                  |
| Dropout             | 0.2               | 0.3             | Better regularization                |
| Input Normalization | None              | BatchNorm1d     | Stabilizes training                  |
| Loss Function       | MSE               | Huber (δ=1.0)   | Robust to outliers                   |
| Optimizer           | Adam              | AdamW (wd=0.01) | Better generalization                |
| Output Clipping     | ±30-50% hardcoded | No clipping     | Full expression range                |
| Feature Scaler      | StandardScaler    | RobustScaler    | Outlier handling                     |
| Target Scaler       | StandardScaler    | StandardScaler  | Mean-centered outputs                |

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 LSTM-DCF TRAINING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] DATA COLLECTION                                             │
│      ├── Alpha Vantage API (quarterly financials)                │
│      │   ├── Income Statement                                    │
│      │   ├── Cash Flow Statement                                 │
│      │   └── Balance Sheet                                       │
│      │                                                           │
│      └── Output: data/raw/financial_statements/                  │
│          ├── {ticker}_income.csv                                 │
│          ├── {ticker}_cashflow.csv                               │
│          └── {ticker}_balance.csv                                │
│                              │                                   │
│                              ▼                                   │
│  [2] FEATURE ENGINEERING                                         │
│      ├── Extract 16 features per quarter:                        │
│      │   revenue, capex, d&a, fcf, ebitda, margins...            │
│      ├── Normalize by total assets                               │
│      └── Standardize (mean=0, std=1)                             │
│                              │                                   │
│      Output: data/processed/lstm_dcf_training/                   │
│                              │                                   │
│                              ▼                                   │
│  [3] SEQUENCE CREATION                                           │
│      ├── Group by ticker                                         │
│      ├── Create overlapping windows (60 steps)                   │
│      └── Split: 70% train, 15% val, 15% test                     │
│                              │                                   │
│                              ▼                                   │
│  [4] TRAINING                                                    │
│      ├── PyTorch Lightning (GPU acceleration)                    │
│      ├── Early stopping (patience=10)                            │
│      ├── Learning rate: 0.001 with scheduler                     │
│      └── Batch size: 64                                          │
│                              │                                   │
│      Output: models/lstm_dcf_enhanced.pth (1.29 MB)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Commands

```powershell
# GPU training (recommended)
python scripts/lstm/train_lstm_dcf_enhanced.py

# Training time: ~6 minutes on RTX 3050
# CPU fallback: ~30-60 minutes
```

### Model Files

| File                           | Version | Size    | Description                         |
| ------------------------------ | ------- | ------- | ----------------------------------- |
| `models/lstm_dcf_enhanced.pth` | **V2**  | ~650 KB | Current production model (Jan 2026) |
| `models/lstm_dcf_final.pth`    | V1      | ~1.3 MB | Legacy model (deprecated)           |

> **Note**: V2 checkpoint includes `feature_scaler` (RobustScaler), `target_scaler` (StandardScaler), and `model_version: 'v2'` marker.

---

## 📊 Model 2: GARP Scoring (Growth at Reasonable Price)

### Purpose

Score stocks by **value + growth balance** using transparent metrics.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GARP SCORING MODEL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Metrics                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • Forward P/E      • Trailing P/E    • PEG Ratio        │    │
│  │ • Revenue Growth   • EPS Growth      • FCF Growth       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Scoring Logic (Transparent)                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • PEG < 1.0: Strong value signal (growth > P/E)         │    │
│  │ • PEG 1.0-2.0: Fair value (balanced)                    │    │
│  │ • PEG > 2.0: Overvalued relative to growth              │    │
│  │ • Growth adjustment: reward double-digit growth         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Output (0-100 Score)                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • GARP Score: 0-100 (higher = better value/growth)      │    │
│  │ • Classification: Undervalued / Fair / Overvalued       │    │
│  │ • Transparent components: easily explained to users     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Scoring Formula

```python
# PEG = P/E ÷ Growth Rate (lower = better value)
peg_score = pe_ratio / (growth_rate * 100)

# GARP Score (0-100, higher = better)
if peg < 1.0:
    garp_score = 80 + (1 - peg) * 20  # 80-100 for PEG < 1
elif peg < 2.0:
    garp_score = 60 + (2 - peg) * 20  # 60-80 for PEG 1-2
else:
    garp_score = max(0, 60 - (peg - 2) * 20)  # < 60 for PEG > 2
```

> **Architecture Note (Jan 2026)**: RF Ensemble was deprecated after analysis
> showed 99.93% P/E importance. GARP scoring provides the same insights
> with full transparency and explainability.

### Model Files

| File                           | Size    | Description                 |
| ------------------------------ | ------- | --------------------------- |
| `models/lstm_dcf_final.pth`    | ~3.7 MB | LSTM-DCF trained model      |
| `models/lstm_dcf_enhanced.pth` | ~4.0 MB | Enhanced LSTM (16 features) |

---

## 🔄 Inference Pipeline

### Fair Value Sanity Checks (Updated Jan 2026)

To prevent extreme predictions from misleading users:

| Check                | Limit            | Rationale                       |
| -------------------- | ---------------- | ------------------------------- |
| **Fair Value Cap**   | 2× current price | Max 100% upside is realistic    |
| **Fair Value Floor** | 0.4× current     | Max 60% downside                |
| **Margin of Safety** | -100% to +100%   | Prevents display of extreme MoS |

> **Why cap at 100%?** A stock showing 200%+ MoS suggests model uncertainty,
> not a guaranteed opportunity. Retail investors should see realistic ranges.

### Real-Time Scoring

```python
# How the screener uses ML models:

class StockScreener:
    def __init__(self):
        # Load models at startup
        self.lstm_model = LSTMDCFModel()
        self.lstm_model.load_model("models/lstm_dcf_enhanced.pth")

    def _calculate_lstm_fair_value(self, ticker, data):
        """
        1. Fetch historical data
        2. Create feature sequence (60 steps)
        3. Normalize features
        4. Run LSTM inference
        5. Convert growth rate → fair value
        6. Apply sanity caps (0.4x - 2x current price)
        """
        features = self._extract_features(data)
        growth_rate = self.lstm_model.predict(features)
        fair_value = self._dcf_calculation(growth_rate, data)
        # Cap to realistic range
        fair_value = np.clip(fair_value, price * 0.4, price * 2.0)
        return fair_value

    def _calculate_valuation_score(self, data):
        """
        Weighted scoring (Updated Jan 2026):
        - Forward P/E:      15%  # Future earnings outlook
        - PEG Ratio:        15%  # Growth at reasonable price
        - Earnings Growth:  10%  # Forward-looking growth
        - P/B vs sector:    10%
        - Margin of Safety: 15%  # LSTM fair value vs price (capped ±100%)
        - FCF Yield:        12%
        - Profitability:    13%  # ROE + margins
        - Financial Health: 10%  # Current/quick ratios

        Key improvements:
        - Forward P/E < 15: Excellent, > 30: Poor
        - PEG < 1: Excellent, 1-2: Good (GARP), > 3: Poor
        - Negative MoS penalized (multiplier 0.60-0.75)
        """
        # Returns 0-100 score
```

### Stock Universe

| Universe     | Count  | Description                              |
| ------------ | ------ | ---------------------------------------- |
| S&P 500 Core | ~150   | Large-cap blue chips                     |
| Extended     | ~64    | Growth tech, mid-cap opportunities       |
| **Curated**  | ~214   | Default screening universe               |
| **Full US**  | ~5,700 | All US-traded common stocks (NASDAQ FTP) |

#### Full Universe Mode (NEW - Jan 2025)

The screener can now scan the **entire US stock market** instead of the curated 214-ticker list:

```python
# API endpoint
GET /api/screener/watchlist?full_universe=true

# Python usage
screener = StockScreener(use_full_universe=True)
stocks = screener.get_top_undervalued(n=20)

# Limit for faster results (recommended for testing)
screener = StockScreener(use_full_universe=True, max_universe_tickers=500)
```

**Performance Notes**:

- Full scan takes 10-30 minutes depending on market data availability
- Uses `src/data/fetchers/ticker_universe.py` to fetch from NASDAQ FTP
- Tickers cached for 24 hours to avoid repeated downloads
- Filters: Common stocks only (excludes ETFs, ADRs, warrants, preferred)

> **Tip**: For production, consider running full scans as background jobs and
> caching results rather than real-time scanning.

### Categorized Watchlist

The screener now provides three separate lists based on investment strategy:

| Category        | Selection Criteria                       | Use Case                     |
| --------------- | ---------------------------------------- | ---------------------------- |
| **Undervalued** | Positive Margin of Safety (fair > price) | Deep value investing         |
| **Quality**     | Highest valuation score (0-100)          | Blue-chip, may be fair value |
| **Growth**      | PEG < 2.0 + earnings growth > 10% (GARP) | Growth at reasonable price   |

```python
# Example usage
screener = StockScreener()
categorized = screener.get_categorized_watchlist(n=10)
# Returns: { 'undervalued': [...], 'quality': [...], 'growth': [...] }
```

### Valuation Status Classification

| Status                    | Margin of Safety | Badge Color |
| ------------------------- | ---------------- | ----------- |
| SIGNIFICANTLY_UNDERVALUED | ≥ 30%            | 🟢 Green    |
| MODERATELY_UNDERVALUED    | 10% - 30%        | 🟢 Green    |
| SLIGHTLY_UNDERVALUED      | 0% - 10%         | 🟡 Yellow   |
| FAIRLY_VALUED             | -5% - 0%         | ⚪ Gray     |
| SLIGHTLY_OVERVALUED       | -15% - -5%       | 🟡 Yellow   |
| MODERATELY_OVERVALUED     | -30% - -15%      | 🟠 Orange   |
| SIGNIFICANTLY_OVERVALUED  | < -30%           | 🔴 Red      |

### Inference Performance

| Metric                 | Target  | Actual  |
| ---------------------- | ------- | ------- |
| Single stock           | < 300ms | ~200ms  |
| Full scan (150 stocks) | < 2 min | ~90 sec |
| Model load time        | < 1 sec | ~500ms  |

---

## 🎯 Pillar Scoring System (Phase 3)

### Overview

Pillar scoring provides a **transparent, modular** evaluation framework that complements LSTM-DCF fair value calculation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PILLAR SCORING SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Each pillar = 0-100 score, 25% weight in composite              │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐│
│  │    VALUE     │ │   QUALITY    │ │    GROWTH    │ │  SAFETY  ││
│  │     25%      │ │     25%      │ │     25%      │ │   25%    ││
│  ├──────────────┤ ├──────────────┤ ├──────────────┤ ├──────────┤│
│  │ • MoS (LSTM) │ │ • FCF ROIC   │ │ • Revenue Gr │ │ • Beta   ││
│  │ • P/E vs     │ │ • ROE vs     │ │ • Earnings Gr│ │ • Vol    ││
│  │   sector     │ │   sector     │ │ • LSTM Pred  │ │ • D/E    ││
│  │ • Forward PE │ │ • Profit Mgn │ │ • PEG Ratio  │ │ • Curr   ││
│  │ • EV/EBITDA  │ │ • D/E ratio  │ │              │ │   Ratio  ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘│
│                                                                  │
│  COMPOSITE = (VALUE + QUALITY + GROWTH + SAFETY) / 4             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pillar Component Weights

#### VALUE Pillar (25% of Composite)

| Component     | Weight | Scoring Logic                                    |
| ------------- | ------ | ------------------------------------------------ |
| LSTM MoS      | 40%    | >30%=100, 20-30%=80, 10-20%=60, 0-10%=40, <0%=20 |
| P/E vs Sector | 25%    | >30% below=100, 15-30%=80, avg=50, >30% above=20 |
| Forward P/E   | 20%    | Improving trend bonus, vs sector comparison      |
| EV/EBITDA     | 15%    | <10=100, 10-15=70, 15-20=50, >20=30              |

#### QUALITY Pillar (25% of Composite)

| Component     | Weight | Scoring Logic                                    |
| ------------- | ------ | ------------------------------------------------ |
| FCF ROIC      | 35%    | >20%=100, 15-20%=85, 10-15%=70, 5-10%=50, <5%=30 |
| ROE vs Sector | 25%    | >1.5x sector=100, 1.2x=80, 1x=60, <0.8x=40       |
| Profit Margin | 25%    | >25%=100, 15-25%=80, 8-15%=60, <8%=40            |
| Debt/Equity   | 15%    | <0.3=100, 0.3-0.6=80, 0.6-1=60, 1-2=40, >2=20    |

#### GROWTH Pillar (25% of Composite)

| Component       | Weight | Scoring Logic                                    |
| --------------- | ------ | ------------------------------------------------ |
| LSTM Predicted  | 30%    | Normalized to 0-100 based on prediction range    |
| Revenue Growth  | 30%    | >20%=100, 15-20%=80, 10-15%=60, 5-10%=40, <5%=20 |
| Earnings Growth | 25%    | Similar scale to revenue                         |
| PEG Ratio       | 15%    | <1=100, 1-1.5=80, 1.5-2=60, 2-3=40, >3=20        |

#### SAFETY Pillar (25% of Composite)

| Component     | Weight | Scoring Logic                                         |
| ------------- | ------ | ----------------------------------------------------- |
| Beta          | 35%    | <0.8=100, 0.8-1.0=80, 1.0-1.3=60, 1.3-1.5=40, >1.5=20 |
| Volatility    | 25%    | <15%=100, 15-25%=80, 25-35%=60, 35-50%=40, >50%=20    |
| Current Ratio | 20%    | >2=100, 1.5-2=80, 1-1.5=60, <1=30                     |
| MoS Buffer    | 20%    | How much cushion above profile's min MoS threshold    |

### Management Quality Bonus (Finnhub Data)

Applied as ±10 points to QUALITY pillar:

| Signal          | Condition           | Points |
| --------------- | ------------------- | ------ |
| Insider Buying  | 3-month ratio > 1.5 | +5     |
| Active Buybacks | >2% of shares TTM   | +3     |
| Dividend Growth | 5+ year streak      | +2     |
| Insider Selling | 3-month ratio < 0.5 | -5     |
| No Buybacks     | And FCF > 0         | -2     |

### Classification Thresholds

| Classification | MoS Requirement | Score Requirement     |
| -------------- | --------------- | --------------------- |
| **BUY**        | ≥ 20%           | ≥ 70                  |
| **HOLD**       | -10% to +20%    | ≥ 70                  |
| **WATCH**      | Any             | ≥ 60 (qualified list) |

---

## 📈 FCF Growth Prediction Evaluation (Jan 2026)

### Evaluation Methodology

The model is evaluated on its **core task**: predicting FCF growth rates. This is more meaningful than comparing stock returns to SPY, as it directly measures what the model predicts.

- **Data**: 144 tickers, 10,480 quarterly samples (2003-2025)
- **Metric**: Compare predicted FCF growth to actual FCF CAGR over various horizons
- **Key insight**: Monotonic quintile ranking is the most important metric for portfolio construction

### Results by Horizon

| Horizon     | Samples | Correlation | Direction Accuracy | MAE   | Quintile Monotonic? |
| ----------- | ------- | ----------- | ------------------ | ----- | ------------------- |
| **1 Year**  | 6,046   | 0.200       | **56.1%**          | 56.7% | ✅ Yes              |
| **2 Year**  | 5,936   | **0.286**   | **56.1%**          | 52.4% | ✅ Yes              |
| **3 Year**  | 5,577   | 0.285       | 53.8%              | 51.2% | ✅ Yes              |
| **5 Year**  | 4,785   | 0.280       | 52.9%              | 51.4% | ✅ Yes              |
| **10 Year** | 2,828   | 0.275       | 49.4%              | 52.1% | ✅ Yes              |

### Quintile Analysis (2-Year Horizon - Best Correlation)

Higher predictions → Higher actual FCF growth (monotonic relationship):

| Quintile      | Predicted Growth | Actual FCF CAGR | Spread vs Q1 |
| ------------- | ---------------- | --------------- | ------------ |
| Q1 (Low)      | -65.0%           | 0.7%            | —            |
| Q2            | -27.9%           | 7.2%            | +6.5%        |
| Q3            | -6.6%            | 8.6%            | +7.9%        |
| Q4            | +18.5%           | 13.4%           | +12.7%       |
| **Q5 (High)** | +127.0%          | **35.7%**       | **+35.0%**   |

### Interpretation

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL RELIABILITY ASSESSMENT                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ✅ STRENGTHS (Use the model for these purposes)                │
│  ────────────────────────────────────────────────────────────── │
│  • Relative ranking: Top quintile consistently outperforms      │
│  • 1-2 year horizon: Best correlation (r=0.29)                  │
│  • Direction accuracy > 55% at short horizons                   │
│  • Monotonic quintile spread: 35% CAGR difference Q5 vs Q1      │
│                                                                  │
│  ⚠️ LIMITATIONS (Don't rely on model for these)                 │
│  ────────────────────────────────────────────────────────────── │
│  • 10-year predictions: Direction accuracy = 49% (random)       │
│  • Absolute values: MAE ~50% means point estimates unreliable   │
│  • Extreme predictions: Model predicts ±100% but actuals ±35%   │
│                                                                  │
│  📊 RECOMMENDED USE                                              │
│  ────────────────────────────────────────────────────────────── │
│  • Use for RANKING stocks, not absolute valuation               │
│  • Best for 1-3 year investment horizons                        │
│  • Combine with GARP (25%) + Risk (25%) for consensus           │
│  • Focus on quintile selection: Buy Q5, avoid Q1                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Statistical Significance

All correlations are statistically significant (p < 0.0001), confirming the model captures real signal despite modest correlation values.

---

## 📈 Legacy Backtesting Results

### Methodology (Historical Reference)

- **Universe**: 78 stocks across 7 sectors
- **Period**: 2010-2020 (avoid COVID anomaly)
- **Predictions**: 1,431 stock-year combinations
- **Benchmark**: S&P 500 (SPY)

### Key Metrics

| Metric               | Result | Interpretation                   |
| -------------------- | ------ | -------------------------------- |
| **Spearman ρ**       | -0.007 | Ranking needs improvement        |
| **Q5-Q1 Spread**     | -16.4% | High upside picks underperformed |
| **SPY Win Rate**     | 90%    | 9/10 cohorts beat SPY            |
| **Deployment Ready** | 1/4    | SPY comparison passes            |

### Interpretation

The model shows promise in **broad market outperformance** but needs refinement in **individual stock ranking**. The 90% SPY win rate suggests value in the overall screening methodology.

---

## ⚠️ Computational Blindspots

### Hard to Automate (Qualitative Factors)

| Factor              | Why Hard               | Partial Proxy              |
| ------------------- | ---------------------- | -------------------------- |
| Management Quality  | Requires reading calls | Insider buying, buybacks   |
| Competitive Moat    | Subjective durability  | Sustained ROIC > 15%       |
| Capital Allocation  | M&A judgment           | ROIC trend, FCF conversion |
| Culture & Execution | Intangible             | (Future: Glassdoor API)    |
| Regulatory Risk     | Domain-specific        | Sector classification only |

### Data Limitations

| Factor              | Issue        | Mitigation                    |
| ------------------- | ------------ | ----------------------------- |
| Forward Estimates   | Analyst bias | Fall back to LSTM prediction  |
| Small Cap Coverage  | Less data    | Weight historical data higher |
| Finnhub Rate Limits | 60/min free  | Batch + cache, Stage 2 only   |

> **Thesis Note**: "The system automates quantitative screening but investors should supplement with qualitative due diligence on management, competitive positioning, and industry dynamics."

---

## 🛠️ Model Maintenance

### Retraining Schedule

| Task             | Frequency        | Script                                       |
| ---------------- | ---------------- | -------------------------------------------- |
| Data collection  | Weekly           | `scripts/quick_start_data_collection.py`     |
| LSTM retraining  | Quarterly        | `scripts/lstm/train_lstm_dcf_enhanced.py`    |
| Attention scan   | Weekly           | `scripts/pipeline/weekly_attention_scan.py`  |
| Qualified update | Daily            | `scripts/pipeline/daily_qualified_update.py` |
| Backtesting      | After retraining | `scripts/backtest_validator.py`              |

### Model Versioning

Models are stored with timestamps in `models/lstm_checkpoints/`:

- `lstm_dcf_epoch_10_val_loss_0.0001.pth`
- `lstm_dcf_epoch_20_val_loss_0.00009.pth`

---

_Next: [4. Multi-Agent System](./04_MULTIAGENT_SYSTEM.md)_
