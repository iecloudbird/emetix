# 3. ML Pipeline & Scoring Methodology

---

## Overview

Emetix uses a multi-layered approach to stock valuation and scoring:

1. **LSTM-DCF V2** — Deep learning fair value estimation using quarterly fundamentals
2. **5-Pillar Composite Scoring** — Systematic stock grading (v3.1)
3. **Quality Growth Gate** — 4-path qualification filter
4. **Consensus Scoring** — Weighted combination of LSTM-DCF, GARP, and Risk scores
5. **3-Stage Pipeline** — Filtering ~5,800 stocks down to ~100 curated picks

---

## 1. LSTM-DCF V2 Model

### Architecture

| Parameter       | Value                                                              |
| --------------- | ------------------------------------------------------------------ |
| Model type      | LSTM (PyTorch Lightning)                                           |
| Hidden size     | 128                                                                |
| Num layers      | 2                                                                  |
| Dropout         | 0.3                                                                |
| Loss function   | Huber Loss                                                         |
| Sequence length | 8 quarters                                                         |
| Input features  | Quarterly fundamentals (revenue, FCF, margins, growth rates, etc.) |
| Output          | 10-year FCFF forecast for DCF valuation                            |
| Accelerator     | `auto` (GPU via CUDA 11.8 when available)                          |

### Model Files

| File                         | Location  | Purpose                       |
| ---------------------------- | --------- | ----------------------------- |
| `lstm_dcf_enhanced.pth`      | `models/` | Primary production model (V2) |
| `lstm_dcf_final.pth`         | `models/` | Original V1 model             |
| `lstm_growth_forecaster.pth` | `models/` | Growth forecasting variant    |

### Training Pipeline

```
Quarterly Fundamentals (yfinance / Alpha Vantage)
        │
        ▼
┌─────────────────────────┐
│  lstm_v2_processor.py   │  Feature engineering
│  8-quarter sequences    │  Normalisation (MinMaxScaler)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  train_lstm_dcf_v2.py   │  PyTorch Lightning training
│  OR train_lstm_dcf_     │  Huber loss, Adam optimiser
│  enhanced.py            │  Early stopping, LR scheduler
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  models/lstm_dcf_       │  Saved checkpoint (.pth)
│  enhanced.pth           │
└─────────────────────────┘
```

**Training scripts**: `scripts/lstm/train_lstm_dcf_v2.py`, `scripts/lstm/train_lstm_dcf_enhanced.py`

**GPU acceleration**: Automatically uses CUDA when available. RTX 3050: ~6 min vs ~30–60 min on CPU.

### Inference

The model is loaded by `EnhancedValuationAgent` and used via the `MLPoweredValuation` tool:

```python
from src.models.deep_learning.lstm_dcf import LSTMDCFModel

model = LSTMDCFModel(input_size=12, hidden_size=128, num_layers=2)
model.load_model(str(MODELS_DIR / "lstm_dcf_enhanced.pth"))
model.eval()  # Inference mode
```

---

## 2. Five-Pillar Composite Scoring (v3.1)

**Source**: `src/analysis/pillar_scorer.py` (828 lines)

### Pillar Weights (v3.1 — Value-Focused)

| Pillar       | Weight  | Key Components                                               |
| ------------ | ------- | ------------------------------------------------------------ |
| **Value**    | **25%** | Margin of Safety (40%), P/E vs Sector (30%), FCF Yield (30%) |
| **Quality**  | **25%** | FCF ROIC, Profit Margin, ROE, Debt-to-Equity                 |
| **Growth**   | **20%** | Revenue Growth, Earnings Growth, LSTM Forecast bonus         |
| **Safety**   | **15%** | Beta, Volatility, Max Drawdown Risk                          |
| **Momentum** | **15%** | RSI, MA Crossovers, Market-share trend                       |

Each pillar produces a score on a **0–100 scale**. The weighted composite determines stock classification.

### Classification Thresholds (v3.1)

| Threshold             | Value | Description                                          |
| --------------------- | ----- | ---------------------------------------------------- |
| `BUY_THRESHOLD`       | 70    | Composite score for Buy classification               |
| `HOLD_THRESHOLD`      | 60    | Composite score for Hold classification              |
| `MIN_QUALIFIED_SCORE` | 60    | Minimum to enter qualified pool                      |
| `PILLAR_FLOOR`        | 40    | Minimum for any core pillar (Value, Quality, Safety) |

### Classification Rules

| Classification | Criteria                                                       |
| -------------- | -------------------------------------------------------------- |
| **Buy**        | MoS ≥ 25% AND composite ≥ 70 AND no pillar floor failures      |
| **Hold**       | MoS −5% to 25% AND composite ≥ 60 AND no pillar floor failures |
| **Watch**      | Everything else                                                |

### Watch Sub-Categories

| Sub-Category             | Criteria                                     |
| ------------------------ | -------------------------------------------- |
| `high_quality_expensive` | Quality or Growth > 70, but MoS < 0%         |
| `cheap_junk`             | Value > 70, but Quality or Safety < 50%      |
| `needs_research`         | Default Watch (does not meet other criteria) |

---

## 3. Quality Growth Gate

**Source**: `src/analysis/quality_growth_gate.py`

A 4-path filter that ensures only genuine growth companies pass through to the qualified pool:

| Path | Name               | Min Revenue Growth | Min FCF ROIC |
| ---- | ------------------ | ------------------ | ------------ |
| 1    | Quality Compounder | ≥ 10%              | ≥ 15%        |
| 2    | Balanced Growth    | ≥ 15%              | ≥ 10%        |
| 3    | Growth Focused     | ≥ 20%              | ≥ 5%         |
| 4    | Hypergrowth        | ≥ 25%              | FCF > 0 only |

Stocks must qualify through **at least one path** to pass the gate.

---

## 4. Consensus Scoring

**Source**: `config/model_config.yaml`, `src/models/ensemble/consensus_scorer.py`

### Agent-Level Consensus Weights

| Model          | Weight  | Rationale                            |
| -------------- | ------- | ------------------------------------ |
| **LSTM-DCF**   | **50%** | Highest-confidence ML fair value     |
| **GARP Score** | **25%** | Growth-at-Reasonable-Price screening |
| **Risk Score** | **25%** | Beta-based risk classification       |

```python
from src.models.ensemble.consensus_scorer import ConsensusScorer

scorer = ConsensusScorer()
consensus = scorer.calculate_consensus({
    'lstm_dcf': 0.70,   # 50% weight
    'garp': 0.65,       # 25% weight
    'risk': 0.60        # 25% weight
})
```

### WatchlistManagerAgent Weights (ML-Enhanced)

When LSTM models are available:

| Component          | Weight |
| ------------------ | ------ |
| LSTM-DCF Valuation | 50%    |
| Growth Score       | 15%    |
| Sentiment Score    | 12%    |
| Valuation Score    | 13%    |
| Risk Score         | 10%    |

**Fallback** (no ML models):

| Component | Weight |
| --------- | ------ |
| Growth    | 30%    |
| Sentiment | 25%    |
| Valuation | 20%    |
| Risk      | 15%    |
| Macro     | 10%    |

---

## 5. Three-Stage Quality Growth Pipeline

### Stage 1 — Attention Scan (Weekly)

**Input**: ~5,800 US stocks  
**Output**: ~200–400 attention-worthy stocks

**5 Attention Triggers** (any one qualifies):

| Trigger                 | Condition                                           |
| ----------------------- | --------------------------------------------------- |
| A — Undervaluation      | P/E < 15 AND P/B < 2.0                              |
| B — Income + Safety     | Dividend Yield > 3% AND Payout < 75% AND Beta < 1.2 |
| C — Momentum Shift      | RSI crossed above 30 (oversold recovery)            |
| D — Growth at Value     | PEG < 1.5 AND Revenue Growth > 10%                  |
| E — Quality Compounding | ROE > 15% AND Debt/Equity < 0.5 AND Margin > 20%    |

**Veto**: Beneish M-Score > −1.78 → flagged as potential earnings manipulation.

**Script**: `scripts/pipeline/weekly_attention_scan.py`

### Stage 2 — Qualification (Daily)

**Input**: Attention stocks  
**Output**: ~100–200 qualified stocks

Applies the **5-Pillar Composite Scoring** to each stock. Qualification criteria:

- Composite score ≥ 60, **OR**
- At least 2 pillars scoring ≥ 65

**Script**: `scripts/pipeline/daily_qualified_update_v3.py`

### Stage 3 — Classification & Curation (On-Demand)

**Input**: Qualified stocks  
**Output**: ~100 curated watchlist entries

Classifies stocks as Buy / Hold / Watch and curates the final list:

| Tier         | Count | Criteria                                     |
| ------------ | ----- | -------------------------------------------- |
| Strong Buy   | ~15   | Composite ≥ 70, MoS ≥ 25%, no floor failures |
| Moderate Buy | ~15   | Composite ≥ 70, MoS 15–25%                   |
| Hold         | ~30   | Composite ≥ 60, MoS −5% to 15%               |
| Watch        | ~40   | Below Hold criteria, sub-categorised         |

**Script**: `scripts/pipeline/stage3_curate_watchlist.py`

---

## Backtest Methodology

### Approach

- **Year-by-year cohort analysis** — Group entry signals by year, measure 5-year outcomes
- **Matched SPY baseline** — Compare identical holding periods against S&P 500
- **Win rate** — Percentage of cohorts outperforming the benchmark

### Honest Metrics

| Metric             | Value   | Interpretation                             |
| ------------------ | ------- | ------------------------------------------ |
| Signal correlation | 0.2–0.4 | Moderate                                   |
| Direction accuracy | ~31%    | Weak individual prediction                 |
| Cohort win rate    | ~80%    | 80% of yearly cohorts beat SPY (2010–2020) |

> **Caveat**: The 2010–2020 period was a strong bull market for US tech. Results should be interpreted with this context.

**Backtest data**: `data/processed/backtesting/`, `data/processed/performance_analysis/`  
**Evaluation scripts**: `scripts/evaluation/backtest_lstm_dcf_10year.py`, `scripts/evaluation/quick_model_test.py`

---

## Model Evaluation

Evaluation metrics are stored in `data/evaluation/`:

| File                           | Contents                       |
| ------------------------------ | ------------------------------ |
| `deep_learning_eval.json`      | LSTM-DCF training/test metrics |
| `traditional_models_eval.json` | Baseline model comparisons     |
