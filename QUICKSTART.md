# Emetix Quick Start Guide

> Get up and running with the Quality Growth Pipeline in 5 minutes.

---

## Prerequisites

- Python 3.11+
- MongoDB Atlas account (free tier works)
- Groq API key (free at console.groq.com)

---

## Initial Setup

### 1. Virtual Environment

```powershell
# Create virtual environment (first time only)
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` file in project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Required for Pipeline (MongoDB Atlas)
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DATABASE=emetix

# Optional (for additional data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
```

### 3. Verify Installation

```powershell
# Test data fetcher
.\venv\Scripts\python.exe -c "from src.data.fetchers import YFinanceFetcher; f=YFinanceFetcher(); print(f.fetch_stock_data('AAPL').keys())"

# Test pipeline imports
.\venv\Scripts\python.exe -c "from src.analysis.quality_growth_gate import QualityGrowthGate, AttentionTriggers; from src.analysis.pillar_scorer import PillarScorer; print('All Phase 3 components imported!')"
```

---

## Running the API

### Start API Server

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn src.api.app:app --reload --port 8000
```

Access at:

- **Swagger UI**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/

### Key API Endpoints (Phase 3)

| Endpoint                       | Description                         |
| ------------------------------ | ----------------------------------- |
| `GET /api/pipeline/attention`  | Stocks that triggered entry         |
| `GET /api/pipeline/qualified`  | Quality-filtered with pillar scores |
| `GET /api/pipeline/classified` | Buy/Hold/Watch lists                |
| `GET /api/pipeline/stock/AAPL` | Single stock analysis               |
| `GET /api/screener/stock/AAPL` | Detailed stock fundamentals         |

---

## Running the Pipeline

### Weekly Attention Scan (Stage 1)

Scans universe for stocks that trigger attention signals:

```powershell
.\venv\Scripts\python.exe scripts/pipeline/weekly_attention_scan.py

# Test with specific tickers
.\venv\Scripts\python.exe scripts/pipeline/weekly_attention_scan.py --tickers AAPL,NVDA,MSFT,META --no-save
```

### Daily Qualified Update (Stage 2)

Scores attention stocks with 4-pillar methodology:

```powershell
.\venv\Scripts\python.exe scripts/pipeline/daily_qualified_update.py

# Test with specific tickers
.\venv\Scripts\python.exe scripts/pipeline/daily_qualified_update.py --tickers NVDA,META --no-save
```

---

## Quick Stock Analysis

### Command Line Analysis

```powershell
# Full analysis (with AI insights if GROQ_API_KEY set)
.\venv\Scripts\python.exe scripts/analyze_stock.py AAPL

# Basic analysis (no AI, faster)
.\venv\Scripts\python.exe scripts/analyze_stock.py AAPL --basic

# Compare multiple stocks
.\venv\Scripts\python.exe scripts/analyze_stock.py AAPL MSFT GOOGL --compare
```

### Python API

```python
# Quality Growth Gate (4-path qualification)
from src.analysis.quality_growth_gate import QualityGrowthGate
gate = QualityGrowthGate()
result = gate.check_qualification("NVDA")
print(f"Qualified: {result['qualified']}, Path: {result['path']}")

# 4-Pillar Scoring
from src.analysis.pillar_scorer import PillarScorer
scorer = PillarScorer()
data = {
    'pe_ratio': 25, 'pb_ratio': 8, 'peg_ratio': 1.2,
    'roe': 35, 'profit_margin': 25, 'current_ratio': 2.0,
    'beta': 1.1, 'revenue_growth': 15, 'eps_growth': 20,
    'debt_equity': 0.3, 'fcf_roic': 18
}
scores = scorer.calculate_pillar_scores(data)
print(f"Composite: {scores['composite_score']:.1f}")
print(f"Value: {scores['value']:.1f}, Quality: {scores['quality']:.1f}")
```

---

## Running Frontend

```powershell
cd frontend
npm install
npm run dev
```

Access at: http://localhost:3000

---

## Testing

```powershell
# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_stock_screener.py

# With coverage
pytest --cov=src --cov-report=html
```

---

## Troubleshooting

### MongoDB Connection

```powershell
# Test MongoDB connection
.\venv\Scripts\python.exe -c "from src.data.mongo_client import is_mongo_available; print('MongoDB:', 'Connected' if is_mongo_available() else 'Not available')"
```

### Common Issues

| Issue                    | Solution                                      |
| ------------------------ | --------------------------------------------- |
| `GROQ_API_KEY missing`   | AI features disabled, core analysis works     |
| `MongoDB not configured` | Pipeline storage disabled, use --no-save flag |
| `Import errors`          | Ensure using `.\venv\Scripts\python.exe`      |
| `DataFrame ValueError`   | Always use `.empty` check, never truthiness   |

---

## Documentation

| Document                                                       | Description             |
| -------------------------------------------------------------- | ----------------------- |
| [docs/README.md](docs/README.md)                               | Documentation index     |
| [docs/05_API_REFERENCE.md](docs/05_API_REFERENCE.md)           | Full API documentation  |
| [docs/PHASE3_IMPLEMENTATION.md](docs/PHASE3_IMPLEMENTATION.md) | Current pipeline design |

---

**Happy Investing! 📈**

_Emetix - AI-Powered Low-Risk Stock Watchlist Platform_
