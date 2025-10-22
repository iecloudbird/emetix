"""Initialize agents package - Multi-Agent System for Stock Analysis"""
from .risk_agent import RiskAgent
from .valuation_agent import ValuationAgent
from .data_fetcher_agent import DataFetcherAgent
from .sentiment_analyzer_agent import SentimentAnalyzerAgent
from .fundamentals_analyzer_agent import FundamentalsAnalyzerAgent
from .watchlist_manager_agent import WatchlistManagerAgent
from .supervisor_agent import SupervisorAgent

__all__ = [
    'RiskAgent',
    'ValuationAgent',
    'DataFetcherAgent',
    'SentimentAnalyzerAgent',
    'FundamentalsAnalyzerAgent',
    'WatchlistManagerAgent',
    'SupervisorAgent'
]
