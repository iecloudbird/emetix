# Emetix Documentation

> **AI-Powered Multi-Agent Low-Risk Stock Watchlist & Risk Management**

---

## Document Index

| #   | Document                                         | Description                                                |
| --- | ------------------------------------------------ | ---------------------------------------------------------- |
| 1   | [Executive Summary](01_EXECUTIVE_SUMMARY.md)     | Project overview, mission, technology stack, key metrics   |
| 2   | [System Architecture](02_SYSTEM_ARCHITECTURE.md) | Full-stack architecture, data flow, project structure      |
| 3   | [ML Pipeline](03_ML_PIPELINE.md)                 | LSTM-DCF V2, 5-pillar scoring, consensus, 3-stage pipeline |
| 4   | [Multi-Agent System](04_MULTIAGENT_SYSTEM.md)    | 8 agents, LangGraph orchestration, tools, LLM config       |
| 5   | [API Reference](05_API_REFERENCE.md)             | All 40+ endpoints across 6 routers                         |
| 6   | [Frontend Guide](06_FRONTEND_GUIDE.md)           | Next.js 16 app, components, API client, pages              |
| 7   | [Deployment](07_DEPLOYMENT.md)                   | Vercel + Render.com, environment variables, local setup    |

---

## Quick Reference

### Technology Stack

| Layer      | Technologies                                                       |
| ---------- | ------------------------------------------------------------------ |
| Frontend   | Next.js 16.1.1, React 19.2.3, TypeScript 5, Tailwind v4, shadcn/ui |
| Backend    | FastAPI, Python 3.10, Uvicorn                                      |
| AI / LLM   | LangGraph, Google Gemini (2.5-flash), Groq fallback                |
| ML         | PyTorch Lightning, LSTM-DCF V2                                     |
| Data       | yfinance, Finnhub, Alpha Vantage, NewsAPI                          |
| Database   | MongoDB Atlas                                                      |
| Deployment | Vercel (frontend), Render.com (backend)                            |

### Core Pipeline

```
~5,800 US Stocks → Attention Scan → Qualification → Curation → ~100 Picks
                    (5 triggers)     (5-pillar ≥60)  (Buy/Hold/Watch)
```

### Scoring System

- **5-Pillar Composite** (v3.1): Value 25%, Quality 25%, Growth 20%, Safety 15%, Momentum 15%
- **Consensus**: LSTM-DCF 50% + GARP 25% + Risk 25%
- **Classification**: Buy (≥70), Hold (≥60), Watch (below 60)
