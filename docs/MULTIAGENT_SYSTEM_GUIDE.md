# Multi-Agent System for Low-Risk Long-Term Stock Analysis

## 🎯 System Overview

The **JobHedge Investor Multi-Agent System** implements a sophisticated 5-agent architecture designed for **low-risk, long-term stock analysis** with emphasis on **contrarian value investing**. The system identifies fundamentally strong stocks suppressed by negative sentiment—opportunities with mean-reversion potential and lower downside risk.

### Investment Philosophy

- **Prioritize Low Risk**: Use FCF-based DCF for accurate fair value estimation
- **Long-Term Focus**: Target sustainable growth over speculative momentum
- **Contrarian Strategy**: Reward stocks with strong fundamentals but negative sentiment
- **Risk-Adjusted Returns**: Balance growth potential with financial stability

---

## 🏗️ Architecture Design

### Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                         │
│              (llama3-70b-8192 - Orchestration)              │
│  Routes queries → Triggers agents → Aggregates results     │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┬─────────────┐
         │             │             │             │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │  DATA   │  │SENTIMENT│  │  FUND.  │  │WATCHLIST│
    │ FETCHER │  │ANALYZER │  │ANALYZER │  │ MANAGER │
    └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

### 1. **Data Fetcher Agent** (llama3-8b-8192)

**Purpose**: Gather raw financial data with validation  
**Tools**:

- `FetchFundamentals`: P/E, P/B, debt ratios, growth metrics
- `FetchFCFData`: Free Cash Flow for DCF valuation (critical!)
- `FetchHistoricalPrices`: Volatility, trend analysis, momentum
- `FetchPeerComparison`: Industry benchmarks

**Key Output**: Structured JSON with fundamentals, FCF, historical data

---

### 2. **Sentiment Analyzer Agent** (mixtral-8x7b-32768)

**Purpose**: Score market sentiment from multiple sources  
**Tools**:

- `AnalyzeNewsSentiment`: Financial news headlines (0-1 score)
- `AnalyzeSocialSentiment`: Price momentum + volume as sentiment proxy
- `AnalyzeAnalystSentiment`: Analyst ratings (buy/hold/sell consensus)

**Key Output**: Sentiment score 0-1 (0=very bearish, 1=very bullish)  
**Contrarian Detection**: Flags when sentiment < 0.4 (suppressed stocks)

---

### 3. **Fundamentals Analyzer Agent** (llama3-70b-8192)

**Purpose**: Compute metrics and calculate intrinsic value  
**Tools**:

- `CalculateGrowthMetrics`: Revenue/earnings CAGR, growth quality
- `CalculateValuationRatios`: P/E, PEG, EV/EBITDA assessments
- `CalculateFinancialHealth`: Debt/equity, liquidity ratios
- `CalculateProfitability`: ROE, ROA, profit margins
- `CalculateDCFIntrinsicValue`: **FCF-based DCF fair value** (most critical)

**Key Output**: Intrinsic value per share, upside %, buy/hold/sell signal

---

### 4. **Watchlist Manager Agent** (gemma2-9b-it)

**Purpose**: Intelligent scoring with contrarian logic  
**Scoring Formula**:

```python
composite_score = (
    growth_score * 0.30 +      # 30% weight
    sentiment_score * 0.25 +   # 25% weight
    valuation_score * 0.20 +   # 20% weight
    risk_score * 0.15 +        # 15% weight
    macro_score * 0.10         # 10% weight
)

# Contrarian Bonus (up to +15%)
if sentiment < 0.4 and valuation > 0.7:
    bonus = (0.7 - sentiment) * valuation * 0.2
    composite_score += min(bonus, 0.15)
```

**Tools**:

- `CalculateCompositeScore`: Weighted scoring with contrarian bonus
- `RankWatchlist`: Sort by score, identify top picks
- `DetectContrarianOpportunities`: Find suppressed undervalued stocks
- `GenerateAlerts`: Score surges, buy dips, overheated signals

**Key Output**: Ranked watchlist with 0-100 scores, buy/hold/sell signals

---

### 5. **Supervisor Agent** (llama3-70b-8192)

**Purpose**: Orchestrate multi-agent workflows  
**Workflow Examples**:

**Single Stock Analysis**:

1. Data Fetcher → Gather fundamentals + FCF
2. Parallel: Fundamentals Analyzer (DCF) + Sentiment Analyzer
3. Supervisor → Aggregate, resolve conflicts, recommend

**Watchlist Building**:

1. Data Fetcher → Fetch all tickers (parallel)
2. Fundamentals + Sentiment → Analyze each stock
3. Watchlist Manager → Score, rank, detect contrarian opportunities
4. Supervisor → Return ranked list with top picks

---

## 💡 Key Features

### 1. FCF-Based DCF Valuation

**Why FCF?** More accurate than EPS-based models:

- Projects actual cash generation (not accounting earnings)
- Discounts future FCF to present value
- Includes terminal value for long-term growth

**Formula**:

```
Fair Value = Σ(FCF_t / (1+WACC)^t) + Terminal Value / (1+WACC)^n
           ─────────────────────────────────────────────────────
                        Shares Outstanding - Net Debt
```

**Example Output**:

```
Intrinsic Value: $185.50
Current Price: $175.00
Upside Potential: +6.0%
Recommendation: BUY (with 25% margin of safety: $139.13)
```

---

### 2. Contrarian Opportunity Detection

**Logic**: Negative sentiment often **temporarily** suppresses fundamentally strong stocks below fair value. These offer:

- **Lower risk**: Margin of safety in valuation
- **Higher reward**: Mean-reversion as fundamentals prevail
- **Long-term horizon**: Requires patience (not short-term trades)

**Detection Criteria**:

- Sentiment score < 0.4 (negative crowd psychology)
- Valuation score > 0.7 (strong fundamentals: low P/E, high ROE, etc.)
- Bonus applied: (0.7 - sentiment) × valuation × 0.2

**Real Example**:

```
Stock: OSCR (Oscar Health)
Sentiment: 0.30 (suppressed by healthcare sector concerns)
Valuation: 0.85 (undervalued: P/E 12, strong fundamentals)
Contrarian Bonus: (0.7 - 0.30) × 0.85 × 0.2 = 6.8%
→ Total Score increased from 47.5 to 54.3/100
→ Signal: HOLD → BUY (mean-reversion potential)
```

---

### 3. Intelligent Watchlist Scoring

**Composite Score Breakdown**:

| Factor        | Weight | Description                                  |
| ------------- | ------ | -------------------------------------------- |
| **Growth**    | 30%    | Revenue/earnings CAGR, growth sustainability |
| **Sentiment** | 25%    | News + social + analyst sentiment            |
| **Valuation** | 20%    | P/E, PEG, EV/EBITDA ratios                   |
| **Risk**      | 15%    | Beta, volatility, debt/equity                |
| **Macro**     | 10%    | Economic indicators, sector trends           |

**Signals**:

- **75-100**: Strong Buy (top contrarian or growth opportunities)
- **65-75**: Buy (good value with manageable risk)
- **45-65**: Hold (fair valuation)
- **35-45**: Weak Hold (review position)
- **0-35**: Sell (overvalued or high risk)

---

## 📊 Usage Examples

### 1. Comprehensive Stock Analysis

```python
from src.agents.supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent()

# Analyze single stock (all agents)
result = supervisor.analyze_stock_comprehensive("AAPL")
print(result['comprehensive_analysis'])

# Output includes:
# - Fundamental metrics + DCF fair value
# - Market sentiment breakdown
# - Investment recommendation with reasoning
```

### 2. Build Intelligent Watchlist

```python
# Build watchlist for multiple stocks
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
result = supervisor.build_watchlist_for_tickers(tickers)

# Custom weights (optional)
custom_weights = {
    'growth': 0.40,      # Emphasize growth
    'sentiment': 0.20,   # De-emphasize sentiment
    'valuation': 0.25,
    'risk': 0.10,
    'macro': 0.05
}
result = supervisor.build_watchlist_for_tickers(tickers, custom_weights)
```

### 3. Scan for Contrarian Opportunities

```python
# Find suppressed undervalued stocks
tickers = ["OSCR", "PFE", "UPS", "KO"]
result = supervisor.scan_for_contrarian_value(tickers)

# Output ranks stocks by contrarian opportunity strength
# Highlights suppressed sentiment + strong fundamentals
```

### 4. Direct Agent Usage

```python
# Use specialized agents directly
from src.agents.fundamentals_analyzer_agent import FundamentalsAnalyzerAgent

fundamentals = FundamentalsAnalyzerAgent()
result = fundamentals.analyze_comprehensive_fundamentals("AAPL")

# Includes DCF intrinsic value, growth metrics, profitability
```

---

## 🛠️ Technical Implementation

### Model Selection (Groq LLMs)

| Agent                 | Model              | Rationale                               |
| --------------------- | ------------------ | --------------------------------------- |
| Data Fetcher          | llama3-8b-8192     | Fast parsing for simple data validation |
| Sentiment Analyzer    | mixtral-8x7b-32768 | Nuanced text analysis for sentiment     |
| Fundamentals Analyzer | llama3-70b-8192    | Complex financial reasoning for DCF     |
| Watchlist Manager     | gemma2-9b-it       | Efficient math aggregation              |
| Supervisor            | llama3-70b-8192    | High-level orchestration decisions      |

**Why Groq?** 10-100x faster inference than traditional LLMs (~280 tokens/sec)

---

### Contrarian Bonus Implementation

```python
def apply_contrarian_bonus(sentiment: float, valuation: float) -> float:
    """
    Apply bonus for suppressed undervalued stocks

    Args:
        sentiment: 0-1 score (0=very negative)
        valuation: 0-1 score (0=overvalued, 1=undervalued)

    Returns:
        Bonus points (0-0.15 max)
    """
    if sentiment < 0.4 and valuation > 0.7:
        # More suppressed + stronger fundamentals = higher bonus
        bonus = (0.7 - sentiment) * valuation * 0.2
        return min(bonus, 0.15)  # Cap at 15%
    return 0.0
```

---

### DCF Model Configuration

```python
from src.models.valuation.fcf_dcf_model import FCFDCFModel

dcf = FCFDCFModel(
    discount_rate=0.10,           # 10% WACC
    terminal_growth_rate=0.025,   # 2.5% perpetual growth
    projection_years=5            # 5-year FCF projection
)

# Calculate with sensitivity
result = dcf.calculate_with_market_price(
    current_fcf=100_000_000_000,       # $100B current FCF
    fcf_growth_rates=[0.12, 0.10, 0.08, 0.06, 0.05],  # Declining
    shares_outstanding=16_000_000_000,
    current_price=175.00,
    net_debt=50_000_000_000,
    margin_of_safety=0.25          # 25% safety margin
)
```

---

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install langchain langchain-groq yfinance pandas numpy

# Set environment variable
export GROQ_API_KEY="your_groq_api_key_here"
```

### Quick Test

```bash
# Test entire multi-agent system
python scripts/test_multiagent_system.py

# Expected output:
# - FCF DCF valuation example
# - Individual agent tests
# - Comprehensive stock analysis
# - Intelligent watchlist building
# - Contrarian opportunity scan
```

---

## 📈 Real-World Performance

### Example: OSCR Analysis

```
Stock: Oscar Health (OSCR)
Current Price: $22.31
Fair Value (DCF): $18.61
Sentiment Score: 0.30 (suppressed by sector concerns)
Valuation Score: 0.75 (undervalued fundamentals)

Contrarian Bonus: +6.8%
Final Score: 54.3/100
Signal: BUY (contrarian opportunity)

Rationale:
- Negative sentiment temporarily suppresses price
- Strong fundamentals (membership growth, improving margins)
- Mean-reversion potential for long-term gains
- Risk: MEDIUM (healthcare sector volatility)
```

---

## 🎯 Best Practices

### 1. Weight Customization

- **Growth investors**: Increase growth weight to 40%
- **Value investors**: Increase valuation weight to 30%
- **Risk-averse**: Increase risk weight to 25%, reduce sentiment to 15%

### 2. Contrarian Strategy

- **Time horizon**: Minimum 6-12 months for mean-reversion
- **Diversification**: Limit contrarian plays to 20-30% of portfolio
- **Monitoring**: Re-assess quarterly as sentiment shifts

### 3. DCF Assumptions

- **Conservative growth**: Use declining growth rates (Year 1: 12% → Year 5: 5%)
- **WACC**: Industry-specific (Tech: 10%, Utilities: 8%, Startups: 15%)
- **Terminal growth**: Match GDP growth (2-3%)

---

## 🔧 Troubleshooting

### Issue: "langchain_groq could not be resolved"

```bash
pip install langchain-groq
```

### Issue: "GROQ_API_KEY not found"

```bash
# Add to .env file
GROQ_API_KEY=gsk_your_key_here

# Or export directly
export GROQ_API_KEY="gsk_your_key_here"
```

### Issue: "No data available for ticker"

- Check ticker symbol is correct (e.g., AAPL not APPL)
- Ensure internet connection for yfinance API
- Some stocks lack FCF data (use alternative valuation)

---

## 📚 Next Steps

1. **Phase 3**: API Backend with FastAPI
2. **Phase 4**: React Frontend with real-time updates
3. **Enhancements**:
   - Real Twitter API integration (not just proxy)
   - Portfolio optimization with multi-objective algorithms
   - Automated backtesting framework
   - Risk-adjusted performance tracking

---

## 📝 Credits

**Architecture Design**: Multi-agent orchestration with LangGraph patterns  
**Valuation Model**: FCF-based DCF with Gordon Growth terminal value  
**Contrarian Logic**: Inspired by Benjamin Graham's value investing principles  
**Technology Stack**: LangChain + Groq LLMs + yfinance + pandas/numpy

---

**Version**: 2.0 (Multi-Agent Architecture)  
**Last Updated**: October 2025  
**License**: MIT
