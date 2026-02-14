# 4. Multi-Agent System

---

## Overview

Emetix uses a **LangGraph-based multi-agent system** to orchestrate comprehensive stock analysis. A supervisor agent coordinates 5 specialised sub-agents, each with dedicated tools, using Google Gemini as the primary LLM.

---

## Architecture

```
                     User Query
                         │
                         ▼
              ┌─────────────────────┐
              │  Supervisor Agent   │
              │  (LangGraph ReAct)  │
              │  gemini-2.5-flash   │
              │  5 orchestration    │
              │  tools              │
              └─────────┬───────────┘
                        │
        ┌───────┬───────┼───────┬───────────┐
        │       │       │       │           │
        ▼       ▼       ▼       ▼           ▼
   ┌────────┐ ┌─────┐ ┌─────┐ ┌──────┐ ┌────────┐
   │ Data   │ │Senti│ │Fund │ │Watch │ │Enhanced│
   │Fetcher │ │ment │ │amen │ │list  │ │Valua-  │
   │ Agent  │ │Analy│ │tals │ │Mngr  │ │tion    │
   └────────┘ └─────┘ └─────┘ └──────┘ └────────┘
```

---

## LLM Configuration

### Provider Hierarchy (`src/utils/llm_provider.py`)

| Tier         | Model                   | Use Case                                    |
| ------------ | ----------------------- | ------------------------------------------- |
| **Large**    | `gemini-2.5-flash`      | Supervisor orchestration, complex reasoning |
| **Fast**     | `gemini-2.5-flash-lite` | Sub-agent tools, quick lookups              |
| **Fallback** | `llama-3.3-70b` (Groq)  | When Gemini quota exhausted                 |

The `FallbackLLM` wrapper automatically retries with the fallback provider on failure. Configured via:

- `GOOGLE_GEMINI_API_KEY` (primary)
- `GROQ_API_KEY` (fallback)
- `LLM_PROVIDER` = `gemini` | `groq` | `auto`

---

## Agent Details

### Supervisor Agent (`src/agents/supervisor_agent.py`)

The orchestrator that routes user queries to appropriate sub-agents.

**LLM**: `get_llm(model_tier="large", temperature=0.2)`  
**Pattern**: LangGraph `create_react_agent` (ReAct reasoning loop)

**Tools** (5):

| Tool                          | Description                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `OrchestrateStockAnalysis`    | 3-phase pipeline: Data fetch → Parallel analysis (Fundamentals + Sentiment + ML) → Aggregation |
| `BuildIntelligentWatchlist`   | Multi-ticker scoring across growth/sentiment/valuation/risk/macro                              |
| `FindContrarianOpportunities` | Finds stocks with negative sentiment but strong fundamentals                                   |
| `ComparePeerStocks`           | Cross-fundamental peer comparison                                                              |
| `MLPoweredValuation`          | LSTM-DCF fair value + consensus scoring                                                        |

**Public methods**: `orchestrate_stock_analysis()`, `build_intelligent_watchlist()`, `find_contrarian_opportunities()`

---

### Sub-Agents

| #   | Agent                     | Class                       | Module                                      | Purpose                                                |
| --- | ------------------------- | --------------------------- | ------------------------------------------- | ------------------------------------------------------ |
| 1   | **Data Fetcher**          | `DataFetcherAgent`          | `src/agents/data_fetcher_agent.py`          | Retrieves market data from yfinance and other sources  |
| 2   | **Sentiment Analyzer**    | `SentimentAnalyzerAgent`    | `src/agents/sentiment_analyzer_agent.py`    | News and social sentiment analysis                     |
| 3   | **Fundamentals Analyzer** | `FundamentalsAnalyzerAgent` | `src/agents/fundamentals_analyzer_agent.py` | Financial metrics evaluation (P/E, P/B, margins, etc.) |
| 4   | **Watchlist Manager**     | `WatchlistManagerAgent`     | `src/agents/watchlist_manager_agent.py`     | Scoring, ranking, and watchlist construction           |
| 5   | **Enhanced Valuation**    | `EnhancedValuationAgent`    | `src/agents/enhanced_valuation_agent.py`    | LSTM-DCF fair value + GARP + consensus scoring         |

### Additional Standalone Agents

| #   | Agent                  | Class               | Module                              | Purpose                                          |
| --- | ---------------------- | ------------------- | ----------------------------------- | ------------------------------------------------ |
| 6   | **Risk Agent**         | `RiskAgent`         | `src/agents/risk_agent.py`          | Beta-based risk classification (Low/Medium/High) |
| 7   | **Valuation Agent**    | `ValuationAgent`    | `src/agents/valuation_agent.py`     | Traditional DCF valuation                        |
| 8   | **Risk Profile Agent** | `PersonalRiskAgent` | `src/agents/personal_risk_agent.py` | Personal risk capacity assessment                |

---

## Agent Tool Pattern

All LangChain tools **must return strings** and **never raise exceptions**:

```python
from langchain.tools import Tool

def tool_function(ticker: str) -> str:
    try:
        fetcher = YFinanceFetcher()
        data = fetcher.fetch_stock_data(ticker)

        # CRITICAL: Check DataFrame with .empty, NOT truthiness
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return "Error: No data available"

        return f"Analysis result: {data['pe_ratio']}"
    except Exception as e:
        return f"Error: {str(e)}"
```

---

## Orchestration Flow

When a user requests a stock analysis (e.g., "Analyze AAPL"):

```
1. Supervisor receives query
2. Selects OrchestrateStockAnalysis tool
3. Phase 1 — Data Fetch:
   └─ DataFetcherAgent retrieves market data from yfinance
4. Phase 2 — Parallel Analysis:
   ├─ FundamentalsAnalyzerAgent → P/E, margins, growth metrics
   ├─ SentimentAnalyzerAgent → News sentiment score
   └─ EnhancedValuationAgent → LSTM-DCF fair value + consensus
5. Phase 3 — Aggregation:
   └─ Supervisor synthesises all results into final recommendation
6. Returns structured analysis to API
```

---

## Consensus Integration

The multi-agent system feeds into the consensus scoring mechanism:

| Agent Output                     | Maps To          | Consensus Weight |
| -------------------------------- | ---------------- | ---------------- |
| Enhanced Valuation → LSTM-DCF    | `lstm_dcf` score | **50%**          |
| Growth Screener → GARP           | `garp` score     | **25%**          |
| Risk Agent → Beta classification | `risk` score     | **25%**          |

Final consensus score determines the confidence level in the valuation estimate, displayed alongside individual agent outputs in the frontend.

---

## Risk Classification (Risk Agent)

| Beta Range    | Classification  | Description                   |
| ------------- | --------------- | ----------------------------- |
| β < 0.8       | **Low Risk**    | Less volatile than the market |
| 0.8 ≤ β ≤ 1.2 | **Medium Risk** | Market-like volatility        |
| β > 1.2       | **High Risk**   | More volatile than the market |

---

## API Integration

The multi-agent system is exposed through the **Multi-Agent Router** (`/api/multiagent/`):

| Endpoint                                      | Method | Description                    |
| --------------------------------------------- | ------ | ------------------------------ |
| `/api/multiagent/stock/{ticker}`              | GET    | Full multi-agent analysis      |
| `/api/multiagent/stock/{ticker}/sentiment`    | GET    | Sentiment-only analysis        |
| `/api/multiagent/stock/{ticker}/fundamentals` | GET    | Fundamentals-only analysis     |
| `/api/multiagent/stock/{ticker}/ml-valuation` | GET    | LSTM-DCF + consensus only      |
| `/api/multiagent/watchlist/analyze`           | POST   | Multi-stock watchlist analysis |

See [05 — API Reference](05_API_REFERENCE.md) for full endpoint documentation.
