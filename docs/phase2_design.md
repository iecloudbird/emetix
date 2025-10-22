# Phase 2: Literature Review & System Design

## Overview

Weeks 3-10 focus on researching existing solutions, identifying gaps, and designing the JobHedge Investor system architecture.

## Objectives

1. Conduct comprehensive literature review on AI agents for finance
2. Identify gaps in existing retail investor tools
3. Design data pipeline and ETL processes
4. Select and prototype ML models
5. Design AI agent architecture
6. Create system UML diagrams

## Literature Review (Weeks 3-6)

### Key Areas

- AI agents in financial analysis (LangChain, AutoGen)
- Stock valuation models (DCF, multiples, ML approaches)
- Risk assessment methodologies
- Portfolio optimization techniques
- Retail investor behavior and needs

### Resources

- LangChain Documentation: https://langchain.com/docs
- Papers on AI in finance (arXiv, IEEE)
- Medium articles on stock analysis agents
- FinTech research papers
- Bloomberg/Reuters API documentation

### Gaps Identified

1. **Limited retail-focused tools**: Most AI solutions target institutional investors
2. **Complexity barrier**: Existing tools require financial expertise
3. **No integrated agents**: Lack of autonomous multi-agent systems for retail
4. **Risk prioritization**: Few tools emphasize low-risk filtering
5. **Cost barriers**: Premium tools too expensive for retail users

## System Design (Weeks 7-10)

### Data Pipeline

#### Data Sources

1. **Yahoo Finance (yfinance)**: Free, comprehensive stock data
2. **Alpha Vantage**: Fundamental data, financial statements
3. **SEC EDGAR**: Official company filings
4. **Market indices**: S&P 500, NASDAQ for benchmarking

#### ETL Process

```
Extract → Transform → Load
  ↓         ↓         ↓
APIs    Pandas     Cache/DB
```

#### Features Extracted

- **Fundamental**: P/E, P/B, debt/equity, current ratio, ROE
- **Price**: Historical OHLCV, volatility, beta
- **Growth**: Revenue growth, EPS growth
- **Sentiment**: News sentiment scores (optional)

### Machine Learning Models

#### 1. Valuation Model

- **Type**: Linear Regression
- **Features**: P/E, debt/equity, revenue growth, beta, market cap
- **Target**: Fair value (current price)
- **Evaluation**: MAE, R², cross-validation

#### 2. Risk Classifier

- **Type**: Random Forest
- **Features**: Beta, volatility, debt levels, liquidity ratios
- **Labels**: Low (0), Medium (1), High (2) risk
- **Evaluation**: Accuracy, precision, recall, F1-score

#### 3. Portfolio Optimizer (Future)

- **Type**: Clustering (K-Means) + optimization
- **Goal**: Diversification across sectors
- **Constraints**: Max/min weights per stock

### AI Agent Architecture

#### Agent Types

1. **Risk Agent**: Assesses individual stock risk
2. **Valuation Agent**: Calculates fair value
3. **Portfolio Agent**: Manages portfolio allocation
4. **Watchlist Agent**: Scans for opportunities

#### Agent Framework

- **Framework**: LangChain
- **LLM**: Groq (llama3-8b-8192) - Free tier
- **Tools**: Data fetchers, ML models, calculators
- **Memory**: Conversation buffer for context

#### Agent Workflow

```
User Query → Agent Orchestrator → Specialized Agent → Tools → Response
                    ↓
              (Risk/Valuation/Portfolio)
```

### System Architecture

```
┌─────────────┐
│   Frontend  │ (React)
│  Dashboard  │
└──────┬──────┘
       │ API
┌──────▼──────┐
│  Flask API  │
│  Endpoints  │
└──────┬──────┘
       │
┌──────▼──────────────┐
│  Agent Orchestrator │
└─┬────────┬─────────┬┘
  │        │         │
┌─▼──┐  ┌─▼──┐   ┌──▼──┐
│Risk│  │Val │   │Port │ AI Agents
│Agnt│  │Agnt│   │Agnt │
└─┬──┘  └─┬──┘   └──┬──┘
  │        │         │
┌─▼────────▼─────────▼─┐
│   Data Layer         │
│ (Fetchers + ML)      │
└──────────────────────┘
```

## Deliverables

### Code

- ✅ Data fetcher modules (yfinance, Alpha Vantage)
- ✅ ETL pipeline
- ✅ Linear valuation model
- ✅ DCF calculator
- ✅ Risk classifier
- ✅ Basic Risk Agent with LangChain

### Documentation

- ✅ System architecture diagram
- ✅ Data flow diagrams
- Literature review summary (15+ references)
- UML sequence diagrams
- API endpoint specifications

### Models

- ✅ Trained valuation model (.pkl)
- ✅ Trained risk classifier (.pkl)
- Model performance metrics

## Timeline

| Week | Task                         | Status         |
| ---- | ---------------------------- | -------------- |
| 3-4  | Literature review            | 🚧 In Progress |
| 5-6  | Gap analysis & references    | 📅 Planned     |
| 7    | Data pipeline implementation | ✅ Complete    |
| 8    | ML model development         | ✅ Complete    |
| 9    | AI agent prototyping         | ✅ Complete    |
| 10   | System design documentation  | 🚧 In Progress |

## Next Steps (Phase 3)

1. Expand agent capabilities (Valuation Agent, Portfolio Agent)
2. Build Flask/FastAPI backend
3. Create React frontend
4. Implement watchlist bot
5. Integration testing

## References

1. LangChain Documentation - https://langchain.com
2. Groq AI Platform - https://groq.com
3. Yahoo Finance API - yfinance library
4. Scikit-learn ML Library
5. "AI Agents for Financial Analysis" - Medium articles
6. "Value Investing with ML" - arXiv papers
7. Alpha Vantage API Documentation
8. Stock valuation methodologies (Graham, Buffett)
9. Modern Portfolio Theory (Markowitz)
10. Risk metrics and beta calculation
11. Retail investor behavior studies
12. FinTech SaaS market research
13. AI in algorithmic trading literature
14. Sentiment analysis for stocks
15. Backtesting methodologies

---

_Document created: October 2025_  
_Status: Phase 2 (Weeks 7-10) - Design & Implementation_
