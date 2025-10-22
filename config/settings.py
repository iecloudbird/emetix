"""
Configuration settings for JobHedge Investor
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

# API Keys (load from environment variables)
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CACHE_DIR = DATA_DIR / 'cache'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
VALUATION_MODEL_PATH = MODELS_DIR / 'valuation_model.pkl'
RISK_MODEL_PATH = MODELS_DIR / 'risk_model.pkl'
PORTFOLIO_MODEL_PATH = MODELS_DIR / 'portfolio_model.pkl'

# ML Model Configuration
ML_CONFIG = {
    'valuation': {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5
    },
    'risk': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}

# Stock Analysis Thresholds
RISK_THRESHOLDS = {
    'low_beta': 1.0,
    'undervalued_pct': 20.0,  # >20% undervalued
    'max_debt_equity': 2.0,
    'min_pe_ratio': 5.0,
    'max_pe_ratio': 30.0
}

# Scoring Weights
SCORING_WEIGHTS = {
    'fundamental': 0.40,
    'technical': 0.30,
    'ml_prediction': 0.30
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('PORT', 5000)),
    'debug': os.getenv('FLASK_ENV', 'development') == 'development'
}

# Rate Limiting
RATE_LIMIT = {
    'default': '100 per hour',
    'premium': '1000 per hour'
}

# Cache Configuration
CACHE_CONFIG = {
    'ttl': 3600,  # 1 hour in seconds
    'max_size': 1000
}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'app.log'

# Database (if needed)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///JobHedge.db')
