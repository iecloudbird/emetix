"""
Alpha Vantage API fetcher
For additional financial data
"""
import requests
import pandas as pd
from typing import Optional
from config.settings import ALPHA_VANTAGE_API_KEY
from config.logging_config import get_logger

logger = get_logger(__name__)


class AlphaVantageFetcher:
    """Fetch data from Alpha Vantage API"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = ALPHA_VANTAGE_API_KEY):
        self.api_key = api_key
        self.logger = logger
    
    def fetch_income_statement(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch income statement data
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with income statement or None
        """
        try:
            params = {
                'function': 'INCOME_STATEMENT',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            if 'annualReports' in data:
                df = pd.DataFrame(data['annualReports'])
                self.logger.info(f"Fetched income statement for {ticker}")
                return df
            else:
                self.logger.warning(f"No income statement data for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching income statement for {ticker}: {str(e)}")
            return None
    
    def fetch_balance_sheet(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch balance sheet data"""
        try:
            params = {
                'function': 'BALANCE_SHEET',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            if 'annualReports' in data:
                df = pd.DataFrame(data['annualReports'])
                return df
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching balance sheet: {str(e)}")
            return None
    
    def fetch_company_overview(self, ticker: str) -> Optional[dict]:
        """Fetch company overview"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            return data if data else None
            
        except Exception as e:
            self.logger.error(f"Error fetching company overview: {str(e)}")
            return None
