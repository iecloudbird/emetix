# 1. Executive Summary

> **Emetix — AI-Powered Multi-Agent Low-Risk Stock Watchlist & Risk Management Platform**

---

## Project Overview

**Emetix** (אמת "truth" + Matrix) is a full-stack, AI-powered stock valuation platform built as a 30-week Final Year Project (FYP). It combines a **LangGraph multi-agent architecture** with 8 specialised AI agents, an **LSTM-DCF V2 deep learning model** for fair value estimation, and a **3-stage Quality Growth Pipeline** that filters ~5,800 US equities down to a curated watchlist of ~100 actionable picks — all accessible through a polished, interactive web application.

### Mission Statement

_Democratize professional stock analysis by making institutional-grade valuation tools accessible to retail investors through AI-powered automation._

---

## Value Proposition

| Challenge for Retail Investors | How Emetix Solves It                                           |
| ------------------------------ | -------------------------------------------------------------- |
| **Information Asymmetry**      | Multi-agent AI aggregates data from 4 sources simultaneously   |
| **Time Constraints**           | Full analysis in seconds vs 2–4 hours manual DCF per stock     |
| **Complexity Barrier**         | LSTM-DCF model automates intrinsic value estimation            |
| **Emotional Bias**             | Systematic 5-Pillar scoring removes subjective decision-making |
| **Risk Management**            | Personal risk profiling tailors recommendations to investor    |

### Key Features

| Feature                           | Description                                                                 |
| --------------------------------- | --------------------------------------------------------------------------- |
| 3-Stage Quality Growth Pipeline   | Filters ~5,800 stocks → ~100 curated picks via attention, scoring, curation |
| Multi-Agent AI Analysis           | 8 LangGraph agents for comprehensive stock evaluation                       |
| LSTM-DCF V2 Fair Value Model      | Deep learning model forecasts 10-year FCFF for intrinsic value              |
| 5-Pillar Composite Scoring (v3.1) | Value, Quality, Growth, Safety, Momentum — 0–100 scale                      |
| Personal Risk Profiling           | Questionnaire-based risk assessment with position sizing guidance           |
| Smart Screener with Risk Filters  | Risk profile integration filters stocks matching investor's appetite        |
| AI Stock Preview Panel            | Instant AI-generated headlines and company descriptions in screener         |
| LLM Analysis Caching              | Server-side MongoDB cache (8h TTL) + client localStorage (1–2 hr TTL)       |
| Data-First Analysis Architecture  | 75% LLM reduction: data-driven narratives + single LLM synthesis call       |
| Enriched Data-Driven Narratives   | Benchmark annotations, top headlines, turning-point flags — no LLM needed   |
| Cold-Start Awareness Banner       | Graceful UX when backend wakes from Render free-tier sleep                  |
| Command Palette (Cmd+K)           | Quick-search navigation across all stocks                                   |

---

## Core Components

| Component              | Technology                            | Purpose                                     |
| ---------------------- | ------------------------------------- | ------------------------------------------- |
| **Frontend**           | Next.js 16, React 19, Tailwind v4     | Interactive dashboard & stock screener      |
| **Backend API**        | FastAPI, Python 3.10                  | 50+ RESTful endpoints across 6 routers      |
| **Multi-Agent System** | LangGraph + Google Gemini 2.5         | 8 specialised AI agents                     |
| **ML Pipeline**        | PyTorch Lightning LSTM-DCF V2         | Fair value prediction & consensus           |
| **Scoring Engine**     | 5-Pillar Composite v3.1               | Quantitative stock scoring & classification |
| **Data Sources**       | Yahoo Finance, Finnhub, Alpha Vantage | Real-time & historical financial data       |
| **Database**           | MongoDB Atlas                         | Pipeline data, watchlists, strategies       |

---

## System Workflow (3-Stage Pipeline)

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

| Metric                   | Value                            |
| ------------------------ | -------------------------------- |
| Stock universe coverage  | ~5,800 US equities               |
| Pipeline output          | ~100 curated low-risk picks      |
| LSTM-DCF V2 architecture | 2-layer LSTM, hidden_size=128    |
| Consensus scoring        | LSTM 50% + GARP 25% + Risk 25%   |
| 5-Pillar scoring version | v3.1 (value-focused weights)     |
| Agent orchestration      | 8 agents via LangGraph ReAct     |
| Frontend pages           | 6 interactive routes             |
| API endpoints            | 40+ RESTful (6 routers)          |
| GPU training             | CUDA 11.8 (RTX 3050: ~6 min)     |
| LLM caching              | localStorage, 1–2 hr TTL         |
| Risk profile integration | Screener filters by user profile |
| Production deployment    | Vercel (FE) + Render.com (BE)    |

---

## Technology Stack

| Layer      | Technologies                                                                     |
| ---------- | -------------------------------------------------------------------------------- |
| Frontend   | Next.js 16.1.1, React 19.2.3, TypeScript 5, Tailwind v4, shadcn/ui, Recharts 3.6 |
| Backend    | FastAPI 3.5, Python 3.10, Uvicorn                                                |
| AI / LLM   | LangGraph, LangChain, Google Gemini 2.5-flash, Groq fallback                     |
| ML         | PyTorch Lightning, LSTM-DCF V2, Huber Loss                                       |
| Data       | yfinance, Finnhub, Alpha Vantage, NewsAPI                                        |
| Database   | MongoDB Atlas (7 collections)                                                    |
| Deployment | Vercel (frontend), Render.com (backend), MongoDB Atlas (database)                |

---

## Documentation Index

| Document                                              | Contents                                             |
| ----------------------------------------------------- | ---------------------------------------------------- |
| [02 — System Architecture](02_SYSTEM_ARCHITECTURE.md) | Full-stack architecture, data flow, component design |
| [03 — ML Pipeline](03_ML_PIPELINE.md)                 | LSTM-DCF V2, 5-pillar scoring, consensus methodology |
| [04 — Multi-Agent System](04_MULTIAGENT_SYSTEM.md)    | 8 agents, LangGraph orchestration, tool descriptions |
| [05 — API Reference](05_API_REFERENCE.md)             | All endpoints across 6 routers                       |
| [06 — Frontend Guide](06_FRONTEND_GUIDE.md)           | Pages, components, UX features, caching strategy     |
| [07 — Deployment](07_DEPLOYMENT.md)                   | Vercel + Render.com setup, cold-start handling       |
