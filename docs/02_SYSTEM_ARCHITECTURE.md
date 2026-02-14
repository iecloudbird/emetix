# 2. System Architecture

---

## High-Level Architecture

```
┌──────────────────────┐     HTTPS      ┌──────────────────────────┐
│   Frontend (Vercel)  │ ◄────────────► │   Backend API (Render)   │
│   Next.js 16.1.1     │                │   FastAPI + Uvicorn      │
│   React 19.2.3       │                │   Python 3.10            │
└──────────────────────┘                └──────────┬───────────────┘
                                                   │
                        ┌──────────────────────────┬┴──────────────────────┐
                        │                          │                       │
                        ▼                          ▼                       ▼
               ┌────────────────┐       ┌──────────────────┐    ┌──────────────────┐
               │  Multi-Agent   │       │   ML Pipeline    │    │   Data Layer     │
               │  System        │       │   (LSTM-DCF V2)  │    │                  │
               │  (LangGraph)   │       │   5-Pillar Score  │    │  Yahoo Finance   │
               │  8 agents      │       │   Quality Gate   │    │  Finnhub         │
               │  Google Gemini │       │   Consensus      │    │  Alpha Vantage   │
               └────────────────┘       └──────────────────┘    │  MongoDB Atlas   │
                                                                └──────────────────┘
```

---

## Project Structure

```
emetix/
├── config/                 # Settings, logging, model_config.yaml
│   ├── settings.py         # Environment variables & path constants
│   ├── logging_config.py   # Centralised logging
│   └── model_config.yaml   # ML hyperparameters & weights
├── data/
│   ├── raw/                # Fetched source data
│   ├── processed/          # Training data, backtest results
│   ├── evaluation/         # Model evaluation metrics (JSON)
│   └── cache/              # API response cache
├── docs/                   # This documentation (01–07)
├── frontend/               # Next.js 16 application
├── models/                 # Trained model files (.pth)
├── notebooks/              # Jupyter analysis notebooks
├── scripts/                # CLI tools & pipeline runners
│   ├── data_collection/    # Data fetching scripts
│   ├── evaluation/         # Model evaluation & backtest
│   ├── lstm/               # LSTM training scripts
│   ├── pipeline/           # Pipeline execution scripts
│   └── consensus/          # Consensus scoring tests
├── src/
│   ├── agents/             # LangGraph multi-agent system
│   ├── analysis/           # Scoring, screening, valuation
│   ├── api/                # FastAPI application & routes
│   ├── data/               # Fetchers & processors
│   ├── models/             # ML model definitions (PyTorch)
│   └── utils/              # LLM provider, helpers
└── tests/                  # Unit & integration tests
```

---

## Backend Architecture

### FastAPI Application (`src/api/app.py`)

The backend is a single FastAPI application (v3.5.0) with 6 routers:

| Router           | Module                       | Prefix   | Responsibility                                   |
| ---------------- | ---------------------------- | -------- | ------------------------------------------------ |
| **Screener**     | `routes.screener`            | `/api`   | Stock lookup, charts, sectors, agent tools       |
| **Pipeline**     | `routes.pipeline`            | `/api`   | 3-stage pipeline endpoints (attention → curated) |
| **Analysis**     | `routes.analysis`            | `/api`   | AI-powered stock analysis                        |
| **Multi-Agent**  | `routes.multiagent_analysis` | `/api`   | Full multi-agent orchestration                   |
| **Risk Profile** | `routes.risk_profile`        | _(none)_ | Personal risk assessment & position sizing       |
| **Storage**      | `routes.storage`             | `/api`   | MongoDB watchlist & strategy CRUD                |

**CORS**: `allow_origins=["*"]` in development; production locks to `https://emetix-woad.vercel.app`.

**Entry points**: `GET /` (API info), `GET /health` (health check).

---

### Data Layer (`src/data/`)

#### Fetchers (`src/data/fetchers/`)

| Fetcher                           | Source                   | API Key Required              |
| --------------------------------- | ------------------------ | ----------------------------- |
| `yfinance_fetcher.py`             | Yahoo Finance            | No                            |
| `finnhub_financials.py`           | Finnhub                  | Yes (`FINNHUB_API_KEY`)       |
| `alpha_vantage_financials.py`     | Alpha Vantage            | Yes (`ALPHA_VANTAGE_API_KEY`) |
| `news_sentiment_fetcher.py`       | Yahoo Finance + NewsAPI  | Optional (`NEWS_API_KEY`)     |
| `technical_sentiment_fetcher.py`  | Computed from price data | No                            |
| `ticker_universe.py`              | Static US universe       | No                            |
| `unified_financials_fetcher.py`   | Multi-source aggregator  | Depends                       |
| `financial_statements_fetcher.py` | Financial statements     | Depends                       |

All fetchers return `None` on failure — they never raise exceptions.

#### Processors (`src/data/processors/`)

| Processor                  | Purpose                                             |
| -------------------------- | --------------------------------------------------- |
| `lstm_v2_processor.py`     | Prepares quarterly fundamental features for LSTM V2 |
| `time_series_processor.py` | Time series normalisation and sequencing            |

#### MongoDB (`src/data/mongo_client.py`, `src/data/pipeline_db.py`)

- **Connection**: MongoDB Atlas via `MONGODB_URI` environment variable
- **Database**: `emetix_pipeline` (configurable)
- **Collections**: `universe_stocks`, `attention_stocks`, `qualified_stocks`, `classified_stocks`, `curated_watchlist`, `watchlists`, `strategies`

---

## Analysis Layer (`src/analysis/`)

| Module                      | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| `pillar_scorer.py`          | 5-Pillar composite scoring engine (v3.1) |
| `quality_growth_gate.py`    | 4-path qualification gate                |
| `stock_screener.py`         | Core screening engine                    |
| `valuation_analyzer.py`     | Traditional valuation metrics            |
| `growth_screener.py`        | GARP growth screening                    |
| `red_flag_detector.py`      | Beneish M-Score / accounting red flags   |
| `moat_detector.py`          | Economic moat detection                  |
| `personal_risk_capacity.py` | Risk capacity scoring & position sizing  |

---

## Configuration (`config/`)

### Environment Variables (`config/settings.py`)

| Variable                | Default             | Purpose                                    |
| ----------------------- | ------------------- | ------------------------------------------ |
| `GOOGLE_GEMINI_API_KEY` | `''`                | Primary LLM (Google Gemini)                |
| `GROQ_API_KEY`          | `''`                | Fallback LLM (Groq / Llama)                |
| `FINNHUB_API_KEY`       | `''`                | Finnhub financial data                     |
| `ALPHA_VANTAGE_API_KEY` | `''`                | Alpha Vantage financial data               |
| `NEWS_API_KEY`          | `''`                | NewsAPI for sentiment                      |
| `MONGODB_URI`           | `''`                | MongoDB Atlas connection string            |
| `MONGODB_DATABASE`      | `'emetix_pipeline'` | MongoDB database name                      |
| `LLM_PROVIDER`          | `'gemini'`          | LLM selection (`gemini` / `groq` / `auto`) |
| `PORT`                  | `5000`              | API server port                            |
| `LOG_LEVEL`             | `'INFO'`            | Logging level                              |

### Path Constants

| Constant             | Resolves To       |
| -------------------- | ----------------- |
| `BASE_DIR`           | Project root      |
| `DATA_DIR`           | `data/`           |
| `RAW_DATA_DIR`       | `data/raw/`       |
| `PROCESSED_DATA_DIR` | `data/processed/` |
| `CACHE_DIR`          | `data/cache/`     |
| `MODELS_DIR`         | `models/`         |

### ML Configuration (`config/model_config.yaml`)

Houses hyperparameters for LSTM-DCF model, consensus scoring weights, and pipeline thresholds. Referenced by training scripts and the consensus scorer.

---

## Frontend Architecture

See [06 — Frontend Guide](06_FRONTEND_GUIDE.md) for full details.

| Aspect     | Details                                                                   |
| ---------- | ------------------------------------------------------------------------- |
| Framework  | Next.js 16.1.1 (App Router)                                               |
| UI         | React 19.2.3, Tailwind v4, shadcn/ui (Radix primitives)                   |
| State      | @tanstack/react-query 5.90                                                |
| Charts     | Recharts 3.6                                                              |
| Forms      | react-hook-form 7.70 + zod 4.3 validation                                 |
| API Client | `src/lib/api.ts` — 25 exported functions                                  |
| Pages      | 6 routes (Home, Screener, Stock Detail, Pipeline, Risk Assessment, About) |
| Deployment | Vercel (auto-deploys from `frontend/` directory)                          |

---

## Data Flow Diagram

```
                User Request (Frontend)
                        │
                        ▼
              ┌─────────────────────┐
              │  FastAPI Backend     │
              │  (6 routers)        │
              └─────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
  ┌───────────┐  ┌────────────┐  ┌───────────────┐
  │ Screener  │  │ Pipeline   │  │ Multi-Agent   │
  │ Router    │  │ Router     │  │ Router        │
  └─────┬─────┘  └─────┬──────┘  └───────┬───────┘
        │               │                │
        ▼               ▼                ▼
  ┌───────────┐  ┌────────────┐  ┌───────────────┐
  │ yfinance  │  │ Pillar     │  │ Supervisor    │
  │ Fetcher   │  │ Scorer     │  │ Agent         │
  │ Analysis  │  │ Quality    │  │ (LangGraph)   │
  │ Modules   │  │ Gate       │  │ 5 sub-agents  │
  └───────────┘  │ MongoDB    │  │ 5 tools       │
                 └────────────┘  └───────┬───────┘
                                         │
                              ┌──────────┼──────────┐
                              │          │          │
                              ▼          ▼          ▼
                        Sentiment   Fundamentals  LSTM-DCF
                        Agent       Agent         Agent
```

---

## Testing

```
tests/
├── conftest.py
├── integration/              # (planned)
└── unit/
    ├── test_analysis/
    │   ├── test_growth_screener.py
    │   └── test_valuation_analyzer.py
    └── test_data/
        └── test_yfinance_fetcher.py
```

Run tests:

```bash
pytest -m unit                           # Unit tests only
pytest --cov=src --cov-report=html       # Coverage report
```
