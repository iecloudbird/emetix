# Emetix

**AI-Powered Multi-Agent Low-Risk Stock Watchlist & Risk Management Platform**

A 30-week FYP thesis project designed for retail investors to quickly calculate insights and valuations for stocks using LangChain multi-agent architecture + advanced ML models (LSTM-DCF, GARP Scoring).

## ✨ Key Features

### Phase 3: Quality Growth Pipeline (NEW)

- **3-Stage Automated Screening**: Attention → Qualified → Classified
- **4-Pillar Scoring**: Value, Quality, Growth, Safety (25% each, 0-100)
- **Quality Growth Gate**: 4-path qualification (ROIC + Revenue Growth)
- **Buy/Hold/Watch Classification**: Actionable lists based on MoS thresholds
- **Momentum Check**: 50MA/200MA accumulation zones

### Core ML Features

- **LSTM-DCF Fair Value**: ML-powered fair value estimation using deep learning
- **Personal Risk Capacity** (Thesis Innovation): Match stock risk to YOUR investor profile
- **Full US Stock Universe**: Screen ~2,000 filtered from 5,700 US stocks
- **Position Sizing**: Kelly-inspired recommendations with emotional buffer

## 🚀 Quick Start

### 1. Setup Virtual Environment

```powershell
# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file in project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for training data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Optional (for watchlist storage)
MONGODB_URI=your_mongoDB_connection_url_here
MONGODB_DATABASE=emetix
```

### 3. Run API Server

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Start FastAPI server
python -m uvicorn src.api.app:app --reload --port 8000
```

Access Swagger UI at: http://localhost:8000/docs

### 4. Run Frontend

```powershell
cd frontend
npm install
npm run dev
```

Access at: http://localhost:3000

## 📁 Project Structure

```
emetix/
├── config/          # Configuration files
├── data/            # Data storage (raw, processed, cache)
├── docs/            # Documentation (7 numbered sections)
├── frontend/        # Next.js frontend
├── models/          # Trained ML models (.pth, .pkl)
├── scripts/         # Executable scripts
├── src/
│   ├── agents/      # LangChain AI agents
│   ├── analysis/    # StockScreener, PersonalRiskCapacity
│   ├── api/         # FastAPI routes
│   ├── data/        # Fetchers, MongoDB client
│   └── models/      # ML model definitions
└── tests/           # Unit and integration tests
```

## 🧪 Development

```powershell
# Run tests
pytest

# Run specific test
pytest tests/unit/test_agents/

# Check code coverage
pytest --cov=src
```

## 📚 Documentation

| Document                                                    | Description                       |
| ----------------------------------------------------------- | --------------------------------- |
| [01_EXECUTIVE_SUMMARY.md](docs/01_EXECUTIVE_SUMMARY.md)     | Project overview, goals, status   |
| [02_SYSTEM_ARCHITECTURE.md](docs/02_SYSTEM_ARCHITECTURE.md) | Backend, agents, data flow        |
| [03_ML_PIPELINE.md](docs/03_ML_PIPELINE.md)                 | LSTM-DCF, 4-Pillar Scoring        |
| [04_MULTIAGENT_SYSTEM.md](docs/04_MULTIAGENT_SYSTEM.md)     | LangChain agents, tools           |
| [05_API_REFERENCE.md](docs/05_API_REFERENCE.md)             | FastAPI endpoints, schemas        |
| [06_FRONTEND_GUIDE.md](docs/06_FRONTEND_GUIDE.md)           | Next.js, React, UI components     |
| [07_DEPLOYMENT.md](docs/07_DEPLOYMENT.md)                   | Hosting, Docker, CI/CD            |
| [PHASE3_IMPLEMENTATION.md](docs/PHASE3_IMPLEMENTATION.md)   | Quality Growth Pipeline (current) |

---

_FYP Thesis Project - Personal Risk Capacity Framework for Retail Investors_
