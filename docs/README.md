# Emetix Documentation

> **AI-Powered Low-Risk Stock Watchlist & Valuation Platform**

This documentation provides a comprehensive overview of the Emetix platform for stakeholders, developers, and academic reviewers.

---

## 📚 Documentation Index

| Document                                              | Description                                                          | Audience                         |
| ----------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------- |
| [1. Executive Summary](./01_EXECUTIVE_SUMMARY.md)     | High-level project overview, value proposition, and key achievements | Stakeholders, Investors          |
| [2. System Architecture](./02_SYSTEM_ARCHITECTURE.md) | Complete technical architecture with diagrams                        | Developers, Technical Reviewers  |
| [3. ML Pipeline](./03_ML_PIPELINE.md)                 | Machine learning models, training, and inference                     | Data Scientists, Developers      |
| [4. Multi-Agent System](./04_MULTIAGENT_SYSTEM.md)    | LangChain agents, orchestration, and AI insights                     | AI Engineers, Developers         |
| [5. API Reference](./05_API_REFERENCE.md)             | FastAPI endpoints, request/response schemas                          | Frontend Developers, Integrators |
| [6. Frontend Integration](./06_FRONTEND_GUIDE.md)     | React/Next.js integration guide with recommended libraries           | Frontend Developers              |
| [7. Deployment Guide](./07_DEPLOYMENT.md)             | Production deployment, environment setup                             | DevOps, Developers               |

> **Note**: Implementation plans (PHASE3_IMPLEMENTATION.md, PIPELINE_SPEC.md) have been archived to `docs/archive/` after completion. Their content is integrated into the main documentation above.

---

## 🎯 Quick Start

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Start API server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 3. Access documentation
# Swagger UI: http://localhost:8000/docs
# API Root:   http://localhost:8000/
```

---

## 🏗️ Project Structure

```
emetix/
├── config/           # Settings, logging, model configuration
├── data/
│   ├── raw/          # Fetched financial data (cached)
│   └── processed/    # Training data, backtest results
├── docs/             # This documentation
├── models/           # Trained ML models (.pth, .pkl)
├── scripts/          # CLI tools and utilities
├── src/
│   ├── agents/       # LangChain multi-agent system
│   ├── analysis/     # Valuation, screening logic
│   ├── api/          # FastAPI backend
│   ├── data/         # Data fetchers
│   └── models/       # ML model definitions
└── tests/            # Unit and integration tests
```

---

## 📊 Technology Stack

| Layer              | Technologies                                       |
| ------------------ | -------------------------------------------------- |
| **Backend**        | Python 3.11, FastAPI, Pydantic                     |
| **ML/AI**          | PyTorch, LangChain, Google Gemini (2.5-flash-lite) |
| **Data**           | Yahoo Finance, Alpha Vantage, Finnhub              |
| **Frontend**       | React 18, Next.js 15, TailwindCSS, Recharts        |
| **Database**       | MongoDB Atlas                                      |
| **Infrastructure** | Docker, CUDA 11.8 (GPU training)                   |

---

## 📈 Key Metrics

- **Stock Universe**: ~2,000 tradeable US stocks (filtered from 5,700)
- **ML Models**: LSTM-DCF Enhanced (16 features, 2 outputs)
- **Screening Pipeline**: 3-stage (Attention → Qualified → Classified)
- **4-Pillar Scoring**: Value / Quality / Growth / Safety (25% each)
- **API Response**: < 300ms per stock
- **Training Time**: 6 minutes (GPU) / 30 minutes (CPU)

---

## 🔄 Current Phase

**Phase 3: Quality Growth Pipeline** (In Progress)

| Component              | Status | Description                             |
| ---------------------- | ------ | --------------------------------------- |
| Core Metrics           | ✅     | FCF ROIC, MAs, Next-Year Revenue Growth |
| Quality Growth Gate    | ✅     | 4-path qualification (ROIC + Growth)    |
| 4-Pillar Scorer        | ✅     | Value, Quality, Growth, Safety          |
| Attention Triggers     | ✅     | 52W Drop, Quality Growth, Deep Value    |
| MongoDB Pipeline       | ✅     | attention_stocks, qualified_stocks      |
| Pipeline API Routes    | ✅     | /api/pipeline/\*                        |
| Weekly Attention Scan  | ✅     | CLI script for Stage 1                  |
| Daily Qualified Update | ✅     | CLI script for Stage 2                  |
| Frontend Integration   | 📋     | Buy/Hold/Watch tabs, PillarRadarChart   |

---

_Last Updated: January 2026_
