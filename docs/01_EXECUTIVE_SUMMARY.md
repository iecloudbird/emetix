# 1. Executive Summary

> **Emetix — AI-Powered Multi-Agent Low-Risk Stock Watchlist & Risk Management**

---

## Project Overview

**Emetix** (אמת "truth" + Matrix) is an AI-powered stock valuation platform built as a 30-week Final Year Project (FYP). It combines a LangGraph multi-agent architecture, an LSTM-DCF V2 deep learning model, and a 3-stage Quality Growth Pipeline to help retail investors identify undervalued, low-risk stocks from a universe of **~5,800 US equities**.

### Mission Statement

_Democratize professional stock analysis by making institutional-grade valuation tools accessible to retail investors through AI-powered automation._

---

## Value Proposition

| Challenge                 | Impact                                                      |
| ------------------------- | ----------------------------------------------------------- |
| **Information Asymmetry** | Retail investors lack access to professional-grade analysis |
| **Time Constraints**      | Manual DCF calculations take 2–4 hours per stock            |
| **Complexity Barrier**    | Financial modelling requires specialised knowledge          |
| **Emotional Bias**        | Human decisions prone to FOMO, loss aversion                |

### Solution

| Feature                         | Benefit                                   |
| ------------------------------- | ----------------------------------------- |
| Multi-agent AI orchestration    | Automated, parallel stock analysis        |
| LSTM-DCF V2 fair value model    | ML-powered intrinsic value estimation     |
| 5-Pillar composite scoring      | Systematic, quantitative stock ranking    |
| 3-Stage Quality Growth Pipeline | From ~5,800 stocks to a curated ~100 list |
| Personal risk profiling         | Tailored position sizing & suitability    |

---

## Core Components

| Component              | Technology                            | Purpose                              |
| ---------------------- | ------------------------------------- | ------------------------------------ |
| **Frontend**           | Next.js 16, React 19                  | Interactive dashboard & screener     |
| **Backend API**        | FastAPI (Python)                      | RESTful endpoints for all features   |
| **Multi-Agent System** | LangGraph + Google Gemini             | 8 specialised AI agents              |
| **ML Pipeline**        | PyTorch Lightning LSTM-DCF            | Fair value prediction (V2)           |
| **Scoring Engine**     | 5-Pillar Composite v3.1               | Quantitative stock scoring           |
| **Data Sources**       | Yahoo Finance, Finnhub, Alpha Vantage | Real-time & historical data          |
| **Storage**            | MongoDB Atlas                         | Persistent pipeline & watchlist data |

---

## System Workflow

```
~5,800 US Stocks
     │
     ▼
┌────────────────────────────┐
│  Stage 1 — Attention Scan  │  5 triggers (A–E) + Beneish M-Score veto
│  (Weekly)                  │  → ~200–400 stocks
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Stage 2 — Qualification   │  5-Pillar scoring (composite ≥ 60)
│  (Daily)                   │  → ~100–200 stocks
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Stage 3 — Curation        │  Buy / Hold / Watch classification
│  (On-demand)               │  → ~100 curated stocks
└────────────────────────────┘
```

---

## Key Technical Achievements

| Metric                    | Value                          |
| ------------------------- | ------------------------------ |
| Stock universe coverage   | ~5,800 US equities             |
| LSTM-DCF V2 architecture  | 2-layer, hidden_size=128       |
| Consensus scoring weights | LSTM 50% + GARP 25% + Risk 25% |
| 5-Pillar scoring version  | v3.1 (value-focused)           |
| Agent orchestration       | 8 agents via LangGraph ReAct   |
| Frontend pages            | 6 interactive pages            |
| API endpoints             | 40+ RESTful endpoints          |
| GPU training support      | CUDA 11.8 (RTX 3050 tested)    |

---

## Technology Stack

| Layer      | Technologies                                                                     |
| ---------- | -------------------------------------------------------------------------------- |
| Frontend   | Next.js 16.1.1, React 19.2.3, TypeScript 5, Tailwind v4, shadcn/ui, Recharts 3.6 |
| Backend    | FastAPI, Python 3.10, Uvicorn                                                    |
| AI / LLM   | LangGraph, LangChain, Google Gemini (2.5-flash), Groq fallback                   |
| ML         | PyTorch Lightning, LSTM-DCF V2                                                   |
| Data       | yfinance, Finnhub, Alpha Vantage, NewsAPI                                        |
| Database   | MongoDB Atlas                                                                    |
| Deployment | Vercel (frontend), Render.com (backend)                                          |

---

## Documentation Index

| Document                                              | Contents                                             |
| ----------------------------------------------------- | ---------------------------------------------------- |
| [02 — System Architecture](02_SYSTEM_ARCHITECTURE.md) | Full-stack architecture, data flow, component design |
| [03 — ML Pipeline](03_ML_PIPELINE.md)                 | LSTM-DCF V2, 5-pillar scoring, consensus methodology |
| [04 — Multi-Agent System](04_MULTIAGENT_SYSTEM.md)    | 8 agents, LangGraph orchestration, tool descriptions |
| [05 — API Reference](05_API_REFERENCE.md)             | All endpoints across 6 routers                       |
| [06 — Frontend Guide](06_FRONTEND_GUIDE.md)           | Next.js 16 app, pages, components, API integration   |
| [07 — Deployment](07_DEPLOYMENT.md)                   | Vercel + Render.com setup, environment variables     |
