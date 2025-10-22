# JobHedge Investor - Project Structure

## Overview

AI-Powered Stock Risk Assessment Platform with AI Agents for Portfolio Management

## Directory Structure

```
jobhedge-investor/
│
├── config/                          # Configuration files
│   ├── __init__.py
│   ├── settings.py                  # Application settings
│   ├── logging_config.py            # Logging configuration
│   ├── api_keys.example.py          # Example API keys template
│   └── model_config.yaml            # ML model hyperparameters
│
├── data/                            # Data storage
│   ├── raw/                         # Raw data from APIs
│   │   ├── stocks/                  # Stock price data
│   │   ├── fundamentals/            # Financial fundamentals
│   │   └── market/                  # Market indices data
│   ├── processed/                   # Cleaned and processed data
│   │   ├── features/                # Feature engineered datasets
│   │   └── training/                # Training datasets
│   └── cache/                       # Cached API responses
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── agents/                      # AI Agent modules
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Base agent class
│   │   ├── risk_agent.py            # Risk assessment agent
│   │   ├── valuation_agent.py       # Stock valuation agent
│   │   ├── portfolio_agent.py       # Portfolio management agent
│   │   ├── watchlist_agent.py       # Watchlist scanner agent
│   │   └── orchestrator.py          # Agent orchestration layer
│   │
│   ├── data/                        # Data pipeline
│   │   ├── __init__.py
│   │   ├── fetchers/                # Data fetching modules
│   │   │   ├── __init__.py
│   │   │   ├── yfinance_fetcher.py  # Yahoo Finance API
│   │   │   ├── alpha_vantage.py     # Alpha Vantage API
│   │   │   └── sec_fetcher.py       # SEC filings
│   │   ├── processors/              # Data processing
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py           # Data cleaning
│   │   │   ├── feature_engineer.py  # Feature extraction
│   │   │   └── validator.py         # Data validation
│   │   └── etl_pipeline.py          # Main ETL orchestration
│   │
│   ├── models/                      # ML Models
│   │   ├── __init__.py
│   │   ├── valuation/               # Valuation models
│   │   │   ├── __init__.py
│   │   │   ├── linear_valuation.py  # Linear regression
│   │   │   ├── dcf_model.py         # DCF calculator
│   │   │   └── ensemble_valuation.py
│   │   ├── risk/                    # Risk models
│   │   │   ├── __init__.py
│   │   │   ├── risk_classifier.py   # Random Forest classifier
│   │   │   ├── volatility_model.py  # Volatility analysis
│   │   │   └── beta_calculator.py
│   │   ├── portfolio/               # Portfolio optimization
│   │   │   ├── __init__.py
│   │   │   ├── diversification.py   # Clustering-based
│   │   │   └── rebalancer.py        # Portfolio rebalancing
│   │   ├── model_trainer.py         # Training pipeline
│   │   └── model_evaluator.py       # Backtesting & metrics
│   │
│   ├── analysis/                    # Analysis modules
│   │   ├── __init__.py
│   │   ├── fundamental.py           # Fundamental analysis (P/E, D/E)
│   │   ├── technical.py             # Technical indicators
│   │   ├── sentiment.py             # Sentiment analysis
│   │   └── scoring.py               # Weighted scoring system
│   │
│   ├── api/                         # Backend API
│   │   ├── __init__.py
│   │   ├── app.py                   # Flask/FastAPI main app
│   │   ├── routes/                  # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── stocks.py            # Stock endpoints
│   │   │   ├── portfolio.py         # Portfolio endpoints
│   │   │   ├── watchlist.py         # Watchlist endpoints
│   │   │   └── health.py            # Health check
│   │   ├── middleware/              # Middleware
│   │   │   ├── __init__.py
│   │   │   ├── auth.py              # Authentication
│   │   │   ├── rate_limiter.py      # Rate limiting
│   │   │   └── error_handler.py     # Error handling
│   │   └── schemas/                 # Pydantic/Marshmallow schemas
│   │       ├── __init__.py
│   │       ├── stock_schema.py
│   │       └── portfolio_schema.py
│   │
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py                # Logging utilities
│   │   ├── cache.py                 # Caching utilities
│   │   ├── validators.py            # Input validators
│   │   ├── formatters.py            # Data formatters
│   │   └── helpers.py               # General helpers
│   │
│   └── bot/                         # Watchlist Bot
│       ├── __init__.py
│       ├── scanner.py               # Stock scanner
│       ├── alerter.py               # Alert system
│       └── scheduler.py             # Scheduled tasks
│
├── frontend/                        # React frontend
│   ├── public/                      # Public assets
│   ├── src/
│   │   ├── components/              # React components
│   │   │   ├── Dashboard/
│   │   │   ├── StockAnalysis/
│   │   │   ├── Portfolio/
│   │   │   ├── Watchlist/
│   │   │   └── Visualizations/
│   │   ├── pages/                   # Page components
│   │   ├── services/                # API services
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── utils/                   # Frontend utilities
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── README.md
│
├── models/                          # Saved ML models
│   ├── valuation_model.pkl
│   ├── risk_model.pkl
│   ├── portfolio_model.pkl
│   └── model_metadata.json
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_agent_testing.ipynb
│
├── scripts/                         # Utility scripts
│   ├── setup_db.py                  # Database initialization
│   ├── train_models.py              # Train all models
│   ├── fetch_historical_data.py     # Bulk data download
│   ├── backtest.py                  # Run backtesting
│   └── deploy.sh                    # Deployment script
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── unit/                        # Unit tests
│   │   ├── test_agents/
│   │   ├── test_data/
│   │   ├── test_models/
│   │   └── test_api/
│   ├── integration/                 # Integration tests
│   │   ├── test_etl_pipeline.py
│   │   └── test_agent_workflow.py
│   ├── conftest.py                  # Pytest fixtures
│   └── test_config.py
│
├── docs/                            # Documentation
│   ├── api/                         # API documentation
│   │   └── endpoints.md
│   ├── architecture/                # System design
│   │   ├── data_pipeline.md
│   │   ├── agent_design.md
│   │   └── uml_diagrams/
│   ├── literature_review.md
│   ├── methodology.md
│   ├── user_guide.md
│   └── deployment.md
│
├── .github/                         # GitHub workflows
│   └── workflows/
│       ├── ci.yml                   # CI/CD pipeline
│       └── deploy.yml
│
├── .gitignore                       # Git ignore file
├── .env.example                     # Environment variables template
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup
├── Procfile                         # Heroku deployment
├── docker-compose.yml               # Docker setup (optional)
├── Dockerfile                       # Docker image
├── pytest.ini                       # Pytest configuration
├── README.md                        # Project overview
└── LICENSE                          # License file
```

## Key Design Principles

### 1. Separation of Concerns

- **agents/**: AI agent logic isolated for reusability
- **data/**: ETL pipeline separate from business logic
- **models/**: ML models with clear interfaces
- **api/**: Backend API with RESTful design
- **frontend/**: Decoupled UI layer

### 2. Scalability

- Modular agents can be deployed independently
- Caching layer for API responses
- Async processing for heavy computations
- Database-ready structure (add `database/` if needed)

### 3. Maintainability

- Clear naming conventions
- Each module has single responsibility
- Comprehensive testing structure
- Documentation alongside code

### 4. Production-Ready Features

- Environment configuration
- Logging and monitoring
- Error handling middleware
- Rate limiting
- API authentication
- CI/CD workflows
- Docker support

## Technology Stack

### Backend

- **Framework**: Flask/FastAPI
- **AI Agents**: LangChain + Groq LLM
- **ML**: Scikit-learn, TensorFlow
- **Data**: Pandas, NumPy
- **APIs**: yfinance, Alpha Vantage

### Frontend

- **Framework**: React
- **Visualization**: Plotly, Recharts
- **State Management**: Redux/Context API
- **HTTP Client**: Axios

### Infrastructure

- **Deployment**: Heroku/AWS
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Testing**: Pytest, Jest

## Getting Started

1. Follow setup instructions in `README.md`
2. Configure environment variables (`.env`)
3. Run data pipeline: `python scripts/fetch_historical_data.py`
4. Train models: `python scripts/train_models.py`
5. Start backend: `python src/api/app.py`
6. Start frontend: `cd frontend && npm start`

## Phase 2 Focus Areas

- **Weeks 3-6**: Implement `data/fetchers/` and `data/processors/`
- **Weeks 7-10**: Build `models/` and `agents/` modules
- Create `notebooks/` for experiments
- Document in `docs/architecture/`
