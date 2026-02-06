"""
Configuration settings for Emetix
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
GOOGLE_GEMINI_API_KEY = os.getenv('GOOGLE_GEMINI_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')

# LLM Provider Configuration
# Options: "gemini", "groq", "auto"
# - gemini: Use Google Gemini (1M tokens/min, but 1500 RPD limit)
# - groq: Use Groq (6K tokens/min, unlimited RPD)
# - auto: Try Gemini first, fallback to Groq on rate limit
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')  # Using Gemini for better token allowance

# MongoDB Atlas Configuration (primary storage for watchlists, education)
# Get connection string from: https://cloud.mongodb.com/
MONGODB_URI = os.getenv('MONGODB_URI', '') 
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'emetix_pipeline')


def get_env(key: str, default: str = '') -> str:
    """Helper function to get environment variables"""
    return os.getenv(key, default)

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

