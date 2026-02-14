# Emetix

**AI-Powered Multi-Agent Low-Risk Stock Watchlist & Risk Management Platform**

A 30-week FYP that helps retail investors identify undervalued, low-risk stocks through LangGraph multi-agent AI orchestration, LSTM-DCF deep learning, and a 3-stage Quality Growth Pipeline — screening ~5,800 US equities down to ~100 curated picks.

---

## Key Features

- **Multi-Agent AI Analysis** — 8 specialised agents (LangGraph + Google Gemini) for sentiment, fundamentals, and valuation
- **LSTM-DCF V2 Fair Value** — Deep learning model forecasts 10-year FCFF for intrinsic value estimation
- **5-Pillar Scoring (v3.1)** — Value (25%), Quality (25%), Growth (20%), Safety (15%), Momentum (15%)
- **3-Stage Pipeline** — Attention Scan → Qualification → Curation (Buy / Hold / Watch)
- **Consensus Scoring** — LSTM-DCF 50% + GARP 25% + Risk 25%
- **Personal Risk Profiling** — Match stock risk to your investor profile with position sizing

---

## Tech Stack

| Layer      | Technologies                                                       |
| ---------- | ------------------------------------------------------------------ |
| Frontend   | Next.js 16.1.1, React 19.2.3, TypeScript 5, Tailwind v4, shadcn/ui |
| Backend    | FastAPI, Python 3.10, Uvicorn                                      |
| AI / LLM   | LangGraph, Google Gemini 2.5-flash, Groq fallback                  |
| ML         | PyTorch Lightning, LSTM-DCF V2                                     |
| Data       | yfinance, Finnhub, Alpha Vantage                                   |
| Database   | MongoDB Atlas                                                      |
| Deployment | Vercel (frontend), Render.com (backend)                            |

---

## Quick Start

### Prerequisites

- Python 3.10+, Node.js 18+, Git
- API keys: Google Gemini (required), Finnhub, Alpha Vantage (optional)
- MongoDB Atlas connection string

### Backend

```bash
git clone <repo-url> && cd emetix

# Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Create .env with your API keys
# GOOGLE_GEMINI_API_KEY=...
# MONGODB_URI=...

# Start server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local
npm run dev
```

### Verify

- Backend health: `http://localhost:8000/health`
- Frontend: `http://localhost:3000`

---

## Environment Variables

| Variable                | Required | Description                          |
| ----------------------- | -------- | ------------------------------------ |
| `GOOGLE_GEMINI_API_KEY` | **Yes**  | Google AI Studio — primary LLM       |
| `MONGODB_URI`           | **Yes**  | MongoDB Atlas connection string      |
| `GROQ_API_KEY`          | No       | Fallback LLM (Groq / Llama 3.3)      |
| `FINNHUB_API_KEY`       | No       | Finnhub financial data               |
| `ALPHA_VANTAGE_API_KEY` | No       | Alpha Vantage financial data         |
| `NEWS_API_KEY`          | No       | NewsAPI for sentiment                |
| `LLM_PROVIDER`          | No       | `gemini` (default) / `groq` / `auto` |

---

## Project Structure

```
emetix/
├── config/          # Settings, logging, model_config.yaml
├── data/            # Raw data, processed data, evaluation metrics
├── docs/            # Full documentation (01–07)
├── frontend/        # Next.js 16 application
├── models/          # Trained LSTM-DCF models (.pth)
├── scripts/         # CLI tools, pipeline runners, training
├── src/
│   ├── agents/      # 8 LangGraph AI agents
│   ├── analysis/    # 5-pillar scorer, screeners, valuation
│   ├── api/         # FastAPI app + 6 routers (40+ endpoints)
│   ├── data/        # Fetchers (yfinance, Finnhub) + processors
│   ├── models/      # ML model definitions (PyTorch)
│   └── utils/       # LLM provider, helpers
└── tests/           # Unit tests (pytest)
```

---

## CLI Usage

```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Stock analysis
python scripts/analyze_stock.py AAPL              # Full AI analysis
python scripts/analyze_stock.py AAPL --basic       # No AI (faster)
python scripts/analyze_stock.py AAPL MSFT --compare

# Pipeline
python scripts/pipeline/weekly_attention_scan.py   # Stage 1
python scripts/pipeline/daily_qualified_update_v3.py # Stage 2
python scripts/pipeline/stage3_curate_watchlist.py # Stage 3

# LSTM training (GPU-accelerated)
python scripts/lstm/train_lstm_dcf_v2.py

# Evaluation
python scripts/evaluation/quick_model_test.py
```

---

## Documentation

Full documentation is in [`docs/`](docs/README.md):

| Doc                                                        | Contents                                        |
| ---------------------------------------------------------- | ----------------------------------------------- |
| [01 — Executive Summary](docs/01_EXECUTIVE_SUMMARY.md)     | Project overview, mission, key metrics          |
| [02 — System Architecture](docs/02_SYSTEM_ARCHITECTURE.md) | Full-stack architecture, data flow              |
| [03 — ML Pipeline](docs/03_ML_PIPELINE.md)                 | LSTM-DCF V2, 5-pillar scoring, 3-stage pipeline |
| [04 — Multi-Agent System](docs/04_MULTIAGENT_SYSTEM.md)    | 8 agents, LangGraph, tools                      |
| [05 — API Reference](docs/05_API_REFERENCE.md)             | All 40+ endpoints across 6 routers              |
| [06 — Frontend Guide](docs/06_FRONTEND_GUIDE.md)           | Next.js 16 app, components, pages               |
| [07 — Deployment](docs/07_DEPLOYMENT.md)                   | Vercel + Render.com setup                       |

---

## Testing

```bash
pytest -m unit                           # Unit tests
pytest --cov=src --cov-report=html       # Coverage report
```

---

## License

FYP / Academic project with [Apache-2.0 license](LICENSE).
