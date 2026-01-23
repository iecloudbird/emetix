# 4. Multi-Agent System

> **LangChain/LangGraph Agents with Gemini LLM Orchestration**

---

## 🎯 System Overview

The Emetix multi-agent system uses **LangGraph** (langgraph.prebuilt) to orchestrate specialized AI agents, each with domain expertise. A **Supervisor Agent** coordinates the workflow and aggregates insights.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────────────────────┐                       │
│                    │  SUPERVISOR AGENT   │                       │
│                    │   (Orchestration)   │                       │
│                    │  gemini-2.5-flash   │                       │
│                    └──────────┬──────────┘                       │
│                               │                                  │
│         ┌─────────────────────┼─────────────────────┐            │
│         │                     │                     │            │
│         ▼                     ▼                     ▼            │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐       │
│  │    RISK     │      │ VALUATION   │      │ SENTIMENT   │       │
│  │   AGENT     │      │   AGENT     │      │   AGENT     │       │
│  │ (Risk Score)│      │(Fair Value) │      │(News Score) │       │
│  └─────────────┘      └─────────────┘      └─────────────┘       │
│         │                     │                     │            │
│         │                     │                     │            │
│         ▼                     ▼                     ▼            │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐       │
│  │ WATCHLIST   │      │ FUNDAMENTAL │      │  ENHANCED   │       │
│  │  MANAGER    │      │  ANALYZER   │      │ VALUATION   │       │
│  │  AGENT      │      │   AGENT     │      │   AGENT     │       │
│  └─────────────┘      └─────────────┘      └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Specifications

### 1. Supervisor Agent

**Purpose**: Orchestrate multi-agent workflows and aggregate results

**Location**: `src/agents/supervisor_agent.py`

**Responsibilities**:

- Route user queries to appropriate agents
- Coordinate sequential/parallel agent calls
- Aggregate and synthesize results
- Generate final recommendations

**LLM**: Google Gemini `gemini-2.5-flash-lite` (default) or `gemini-2.5-flash` (large)

```python
# Example usage
from src.agents.supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent()
result = supervisor.analyze("Analyze AAPL for long-term investment")
# Returns: comprehensive analysis from multiple agents
```

---

### 2. Risk Agent

**Purpose**: Classify stocks by risk level using Beta and fundamentals

**Location**: `src/agents/risk_agent.py`

**Tools**:
| Tool | Function |
|------|----------|
| `CalculateBetaRisk` | Beta-based risk classification |
| `AssessVolatility` | Historical price volatility |
| `EvaluateDebtRisk` | Debt/equity analysis |

**Risk Classification**:

```
Beta < 0.8  → LOW Risk
0.8 ≤ Beta ≤ 1.2 → MEDIUM Risk
Beta > 1.2  → HIGH Risk
```

**Output**:

```json
{
  "ticker": "AAPL",
  "risk_level": "MEDIUM",
  "beta": 1.15,
  "confidence": 85,
  "justification": "Beta of 1.15 indicates market-average volatility..."
}
```

---

### 3. Valuation Agent

**Purpose**: Calculate fair value using traditional DCF

**Location**: `src/agents/valuation_agent.py`

**Tools**:
| Tool | Function |
|------|----------|
| `CalculateDCF` | Discounted Cash Flow model |
| `CalculatePERatio` | P/E relative valuation |
| `CalculatePEGRatio` | Growth-adjusted P/E |

**DCF Formula**:

```
Fair Value = FCF × (1 + growth) / (WACC - terminal_growth)
```

---

### 4. Enhanced Valuation Agent

**Purpose**: ML-powered valuation with specialized tools

**Location**: `src/agents/enhanced_valuation_agent.py`

**Tools**:
| Tool | Function |
|------|----------|
| `ComprehensiveValuation` | Traditional P/E, P/B, PEG ratios |
| `LSTM_DCF_Valuation` | Deep learning fair value prediction |
| `ConsensusValuation` | Multi-model weighted average |
| `StockComparison` | Compare multiple stocks side-by-side |

**Architecture (Jan 2025)**:

- LSTM-DCF: 50% weight (Deep learning fair value)
- GARP Score: 25% weight (Forward P/E + PEG)
- Risk Score: 25% weight (Beta + volatility)

**This is the primary valuation agent used by the screener.**

---

### 5. Sentiment Analyzer Agent

**Purpose**: Aggregate news sentiment from multiple sources

**Location**: `src/agents/sentiment_analyzer_agent.py`

**Tools**:
| Tool | Function |
|------|----------|
| `FetchNewsHeadlines` | Multi-source news aggregation |
| `AnalyzeSentiment` | NLP sentiment scoring |
| `DetectContrarianSignal` | Low sentiment + strong fundamentals |

**Sentiment Score**:

```
0.0 - 0.3 → Very Bearish (Contrarian opportunity if fundamentals strong)
0.3 - 0.5 → Bearish
0.5 - 0.7 → Neutral
0.7 - 0.9 → Bullish
0.9 - 1.0 → Very Bullish (Possible overvaluation)
```

---

### 6. Watchlist Manager Agent

**Purpose**: Score and rank stocks for watchlist

**Location**: `src/agents/watchlist_manager_agent.py`

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
if sentiment < 0.4 and valuation_score > 70:
    bonus = (0.7 - sentiment) * valuation_score / 100 * 0.2
    composite_score += min(bonus, 15)
```

---

### 7. Fundamentals Analyzer Agent

**Purpose**: Deep financial health analysis

**Location**: `src/agents/fundamentals_analyzer_agent.py`

**Tools**:
| Tool | Function |
|------|----------|
| `AnalyzeProfitability` | ROE, ROA, margins |
| `AnalyzeLiquidity` | Current ratio, quick ratio |
| `AnalyzeSolvency` | Debt coverage, interest coverage |
| `AnalyzeEfficiency` | Asset turnover, inventory days |

---

## 🔗 LLM Integration

### Google Gemini Configuration

```python
# config/settings.py
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"        # Default (10 RPM, 250K TPM)
GEMINI_MODEL_LARGE = "gemini-2.5-flash"       # Large model (5 RPM)
```

### Rate Limiting (Free Tier)

| Model                 | Requests/min | Tokens/min |
| --------------------- | ------------ | ---------- |
| gemini-2.5-flash-lite | 10           | 250,000    |
| gemini-2.5-flash      | 5            | N/A        |

### Error Handling

```python
def call_agent_with_fallback(query):
    try:
        return agent.invoke(query)
    except ResourceExhausted:
        time.sleep(60)  # Wait for rate limit reset
        return agent.invoke(query)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"error": str(e)}
```

---

## 🔄 Agent Workflow

### Typical Query Flow

```
User Query: "Should I invest in MSFT?"
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ SUPERVISOR AGENT                                                │
│ 1. Parse query → Investment decision                            │
│ 2. Determine required agents:                                   │
│    - Risk Agent (risk assessment)                               │
│    - Valuation Agent (fair value)                               │
│    - Sentiment Agent (news analysis)                            │
└─────────────────────────────────────────────────────────────────┘
       │
       ├─────────────────┬─────────────────┐
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RISK AGENT  │  │  VALUATION   │  │  SENTIMENT   │
│  Beta: 0.95  │  │  FV: $420    │  │  Score: 0.72 │
│  Risk: LOW   │  │  MoS: 15%    │  │  Bullish     │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ SUPERVISOR AGGREGATION                                          │
│                                                                  │
│ "MSFT shows LOW risk (Beta 0.95), with fair value of $420       │
│  suggesting 15% upside. Bullish sentiment from recent cloud     │
│  revenue growth. RECOMMENDATION: BUY for long-term investors."  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Enabling AI Insights

```python
# In screener initialization
from src.analysis.stock_screener import StockScreener

screener = StockScreener(
    enable_ai_insights=True,  # Enable LangChain agents
    enable_lstm=True          # Enable ML models
)
```

### Environment Variables

```env
# .env file
GEMINI_API_KEY=your_google_ai_api_key_here
```

---

## 📊 Agent Performance

### Response Times

| Agent             | Avg Time | Token Usage  |
| ----------------- | -------- | ------------ |
| Risk Agent        | 1.5s     | ~500 tokens  |
| Valuation Agent   | 2.0s     | ~800 tokens  |
| Sentiment Agent   | 2.5s     | ~1000 tokens |
| Supervisor (full) | 6-10s    | ~3000 tokens |

### Cost Estimation (Gemini)

- Free tier: ~10 analyses/minute (rate limited by RPM)
- Generous token quota: 250K TPM per model

---

_Next: [5. API Reference](./05_API_REFERENCE.md)_
