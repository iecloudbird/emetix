# Emetix

## Quick Start

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
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### 3. Run Scripts

```powershell
# Always activate venv first
.\venv\Scripts\Activate.ps1

# Run data pipeline
.\venv\Scripts\python.exe scripts/fetch_historical_data.py

# Train ML models
.\venv\Scripts\python.exe scripts/train_lstm_dcf.py
.\venv\Scripts\python.exe scripts/train_rf_ensemble.py

# Test models
.\venv\Scripts\python.exe scripts/quick_model_test.py

# Run stock analysis
.\venv\Scripts\python.exe scripts/analyze_stock.py AAPL
```

### 4. Deactivate

```powershell
deactivate
```

## Project Structure

```
emetix/
├── config/          # Configuration files
├── data/            # Data storage (raw, processed, cache)
├── models/          # Trained ML models
├── scripts/         # Executable scripts
├── src/             # Source code (agents, models, data fetchers)
└── tests/           # Unit and integration tests
```

## Development

```powershell
# Run tests
pytest

# Run specific test
pytest tests/unit/test_agents/

# Check code coverage
pytest --cov=src
```

---

_For detailed documentation, see `/docs` directory._
