# Quick Start Guide

Welcome to JobHedge Investor! This guide will help you get started quickly.

## Initial Setup (5 minutes)

### 1. Environment Setup

After installing dependencies, set up your environment variables:

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add these keys:
```

Required API keys (all FREE):

#### Groq API Key (for AI Agents)

1. Go to https://console.groq.com
2. Sign up for a free account
3. Navigate to API Keys
4. Create a new API key
5. Copy it to your `.env` file: `GROQ_API_KEY=your_key_here`

#### Alpha Vantage API Key (optional, for additional data)

1. Go to https://www.alphavantage.co/support/#api-key
2. Enter your email
3. Copy the key to `.env`: `ALPHA_VANTAGE_API_KEY=your_key_here`

### 2. Verify Installation

Test that everything is working:

```bash
# Test data fetcher
python src/data/fetchers/yfinance_fetcher.py

# You should see output like:
# INFO - Successfully fetched data for AAPL
```

## Running Your First Analysis

### Option 1: Quick Valuation Test (30 seconds)

Test our comprehensive valuation system on any stock:

```bash
# Analyze a single stock (e.g., Apple)
python scripts/quick_ticker_test.py

# Or analyze a specific ticker
python -c "from src.analysis.valuation_analyzer import ValuationAnalyzer; analyzer = ValuationAnalyzer(); result = analyzer.analyze_stock('AAPL'); print(f'Score: {result[\"overall_score\"]}/100, Rating: {result[\"recommendation\"]}')"
```

### Option 2: Comprehensive Stock Analysis

Run a full analysis on any stock ticker:

```bash
# Example: Analyze Oscar Health (OSCR)
python scripts/analyze_oscr_core.py

# This will show:
# - Current valuation metrics (P/E, P/B, PEG ratios)
# - Fair value estimation with DCF model
# - Growth opportunity assessment
# - Risk factor analysis
# - Peer comparison in industry
```

### Option 3: Growth Stock Screening

Find undervalued growth opportunities:

```bash
python -c "
from src.analysis.growth_screener import GrowthScreener
screener = GrowthScreener()
opportunities = screener.screen_growth_opportunities(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])
for ticker, score in opportunities.items():
    print(f'{ticker}: Growth Score {score}/100')
"
```

### Option 4: AI Agent Analysis (requires GROQ_API_KEY)

Train models and test the AI agent:

```bash
# Step 1: Fetch training data
python scripts/fetch_historical_data.py

# Step 2: Train ML models
python scripts/train_models.py

# Step 3: Test the AI agent (requires GROQ_API_KEY)
python scripts/test_agent.py
```

## Advanced Features

### 1. Comprehensive Stock Valuation

Our valuation system provides 12+ financial metrics with a 0-100 scoring system:

```python
from src.analysis.valuation_analyzer import ValuationAnalyzer

analyzer = ValuationAnalyzer()
result = analyzer.analyze_stock('AAPL')

print(f"Overall Score: {result['overall_score']}/100")
print(f"Recommendation: {result['recommendation']}")
print(f"Fair Value: ${result['fair_value_estimate']:.2f}")
print(f"Current Price: ${result['current_price']:.2f}")
```

### 2. Growth Stock Screening (GARP Strategy)

Find undervalued growth opportunities using Growth at a Reasonable Price strategy:

```python
from src.analysis.growth_screener import GrowthScreener

screener = GrowthScreener()
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
opportunities = screener.screen_growth_opportunities(tickers)

for ticker, analysis in opportunities.items():
    print(f"{ticker}: Growth Score {analysis['growth_score']}/100")
```

### 3. Peer Comparison Analysis

Compare stocks within the same industry:

```python
from src.analysis.valuation_analyzer import ValuationAnalyzer

analyzer = ValuationAnalyzer()

# Compare healthcare insurance companies
healthcare_stocks = ['UNH', 'CNC', 'HUM', 'OSCR']
comparison = analyzer.compare_stocks(healthcare_stocks)

print("Healthcare Insurance Comparison:")
for ticker, data in comparison.items():
    print(f"{ticker}: Score {data['overall_score']}/100")
```

### 4. AI-Powered Analysis Reports

Generate comprehensive analysis reports with our AI agents:

```python
# Note: Requires langchain_groq package installation
from src.agents.valuation_agent import ValuationAgent

agent = ValuationAgent()
report = agent.analyze_stock_comprehensive('AAPL')
print(report)
```

## Common Use Cases

### 1. Analyze a Single Stock

```python
from src.data.fetchers import YFinanceFetcher

# Fetch stock data
fetcher = YFinanceFetcher()
data = fetcher.fetch_stock_data('AAPL')
print(data)
```

### 2. Calculate Fair Value (DCF)

```python
from src.models.valuation import DCFModel

# Create DCF model
dcf = DCFModel(discount_rate=0.10)

# Calculate fair value
result = dcf.calculate_intrinsic_value_with_margin(
    eps=5.50,              # Earnings per share
    growth_rate=0.15,      # 15% growth
    current_price=150.00   # Current market price
)

print(f"Fair Value: ${result['fair_value']:.2f}")
print(f"Recommendation: {result['recommendation']}")
```

### 3. Assess Risk with ML

```python
from src.models.risk import RiskClassifier
import pandas as pd

# Load trained model
risk_model = RiskClassifier()
risk_model.load('models/risk_model.pkl')

# Prepare stock data
stock_features = pd.DataFrame({
    'beta': [0.7],
    'volatility': [25],
    'debt_equity': [0.5],
    'current_ratio': [2.0]
})

# Get risk assessment
result = risk_model.assess_risk(stock_features)
print(result)
```

### 4. Use AI Agent (Requires Groq API Key)

```python
from src.agents import RiskAgent

# Initialize agent
agent = RiskAgent()

# Get comprehensive analysis
result = agent.assess_risk("AAPL")
print(result['analysis'])
```

## Project Workflow

### For Phase 2 (Current):

1. **Data Collection**

   ```bash
   python scripts/fetch_historical_data.py
   ```

2. **Model Training**

   ```bash
   python scripts/train_models.py
   ```

3. **Experimentation**

   - Open Jupyter: `jupyter notebook`
   - Navigate to `notebooks/`
   - Create new notebooks for analysis

4. **Testing**
   ```bash
   pytest tests/
   ```

### For Phase 3 (Next):

1. Build API backend
2. Create React frontend
3. Implement watchlist bot
4. Integration testing

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd jobhedge-investor

# Activate virtual environment
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### API Key Issues

```bash
# Check if .env file exists
dir .env  # Windows
ls .env   # Mac/Linux

# Verify keys are loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"
```

### Data Fetching Errors

- **Rate limits**: Wait a few seconds between requests
- **Invalid ticker**: Verify ticker symbol is correct
- **Network issues**: Check internet connection

## Next Steps

1. ✅ Complete Phase 2 implementation
2. 📖 Read `docs/phase2_design.md`
3. 🧪 Experiment in notebooks
4. 📊 Test models with different stocks
5. 🤖 Build additional AI agents

## Resources

- **Documentation**: `docs/`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **API Docs**: `docs/api/endpoints.md`
- **Phase 2 Details**: `docs/phase2_design.md`

## Getting Help

- Check existing notebooks for examples
- Review test files in `tests/`
- Read module docstrings
- Open an issue on GitHub

---

**Happy Coding! 🚀**

_Built for FYP - JobHedge Investor_
