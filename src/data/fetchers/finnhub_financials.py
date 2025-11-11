"""
Finnhub Financial Data Fetcher for LSTM Training
Fetches quarterly financial data from Finnhub API

Data Coverage:
- Financial statements (quarterly)
- Company metrics
- Basic financials

Rate Limits (Free Tier):
- 60 API calls per minute
- No daily limit
- Much more generous than Alpha Vantage
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from config.logging_config import get_logger
from config.settings import RAW_DATA_DIR

load_dotenv()
logger = get_logger(__name__)


class FinnhubFinancialsFetcher:
    """
    Fetch financial statements from Finnhub for LSTM-DCF training
    
    Advantages over Alpha Vantage:
    - 60 calls/minute (vs 5/minute)
    - No daily limit (vs 25/day)
    - Faster data collection
    - Good for expanding dataset
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY not found. Set it in .env file.")
        
        self.base_url = "https://finnhub.io/api/v1"
        
        # Rate limiting (conservative: 50/min to be safe)
        self.calls_per_minute = 50
        self.calls_this_minute = []
        
        # Caching
        self.cache_dir = RAW_DATA_DIR / "finnhub_financials"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Finnhub Financial Fetcher initialized")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Rate limit: {self.calls_per_minute} calls/minute")
    
    def _wait_for_rate_limit(self):
        """Smart rate limiting: 50 calls/min"""
        current_time = time.time()
        
        # Clean up old timestamps (older than 60 seconds)
        self.calls_this_minute = [t for t in self.calls_this_minute if current_time - t < 60]
        
        # If at limit, wait
        if len(self.calls_this_minute) >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.calls_this_minute[0]) + 1
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.calls_this_minute = []
        
        # Record call
        self.calls_this_minute.append(current_time)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting"""
        self._wait_for_rate_limit()
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params)
        
        if response.status_code == 429:
            logger.warning("Rate limit hit, waiting 60s...")
            time.sleep(60)
            return self._make_request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def _get_cache_path(self, ticker: str, data_type: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{ticker}_{data_type}.json"
    
    def _load_from_cache(self, ticker: str, data_type: str) -> Optional[Dict]:
        """Load from cache if available"""
        import json
        cache_path = self._get_cache_path(ticker, data_type)
        
        if cache_path.exists():
            logger.debug(f"Loading {ticker} {data_type} from cache")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _save_to_cache(self, data: Dict, ticker: str, data_type: str):
        """Save to cache"""
        import json
        cache_path = self._get_cache_path(ticker, data_type)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Cached {ticker} {data_type}")
    
    def fetch_financials_reported(self, ticker: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Fetch reported financials (as-reported)
        Returns quarterly data
        """
        if use_cache:
            cached = self._load_from_cache(ticker, 'financials_reported')
            if cached is not None:
                return cached
        
        try:
            data = self._make_request('stock/financials-reported', {'symbol': ticker, 'freq': 'quarterly'})
            
            if not data or 'data' not in data:
                logger.warning(f"No financials data for {ticker}")
                return None
            
            self._save_to_cache(data, ticker, 'financials_reported')
            logger.info(f"✓ Fetched reported financials for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {e}")
            return None
    
    def fetch_basic_financials(self, ticker: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Fetch basic financial metrics
        Returns latest metrics
        """
        if use_cache:
            cached = self._load_from_cache(ticker, 'basic_financials')
            if cached is not None:
                return cached
        
        try:
            data = self._make_request('stock/metric', {'symbol': ticker, 'metric': 'all'})
            
            if not data:
                logger.warning(f"No basic financials for {ticker}")
                return None
            
            self._save_to_cache(data, ticker, 'basic_financials')
            logger.info(f"✓ Fetched basic financials for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching basic financials for {ticker}: {e}")
            return None
    
    def parse_financials_to_dataframe(self, data: Dict) -> Optional[pd.DataFrame]:
        """
        Parse Finnhub financials to DataFrame format
        Extract: Revenue, CapEx, D&A, EBIT, Tax, Assets
        """
        if not data or 'data' not in data:
            return None
        
        records = []
        
        for report in data['data']:
            try:
                # Extract report date
                report_date = report.get('filedDate') or report.get('acceptedDate')
                if not report_date:
                    continue
                
                # Get the report data
                report_data = report.get('report', {})
                
                # Try to find key financial metrics
                # Note: Finnhub structure varies by company, need to handle dynamically
                bs = report_data.get('bs', [])  # Balance Sheet
                ic = report_data.get('ic', [])  # Income Statement
                cf = report_data.get('cf', [])  # Cash Flow
                
                # Helper to find value by label
                def find_value(items, keywords):
                    if not items:
                        return None
                    for item in items:
                        label = item.get('label', '').lower()
                        if any(kw in label for kw in keywords):
                            return item.get('value')
                    return None
                
                # Extract components
                revenue = find_value(ic, ['revenue', 'sales', 'total revenue'])
                ebit = find_value(ic, ['operating income', 'ebit', 'income from operations'])
                tax_expense = find_value(ic, ['income tax', 'tax expense', 'provision for income tax'])
                
                capex = find_value(cf, ['capital expenditure', 'capex', 'purchase of property'])
                da = find_value(cf, ['depreciation', 'amortization', 'depreciation and amortization'])
                
                total_assets = find_value(bs, ['total assets', 'assets'])
                
                if revenue is None:
                    continue
                
                record = {
                    'date': report_date,
                    'revenue': float(revenue) if revenue else 0,
                    'ebit': float(ebit) if ebit else 0,
                    'tax_expense': float(tax_expense) if tax_expense else 0,
                    'capex': abs(float(capex)) if capex else 0,
                    'da': float(da) if da else 0,
                    'total_assets': float(total_assets) if total_assets else 0
                }
                
                records.append(record)
                
            except Exception as e:
                logger.debug(f"Error parsing report: {e}")
                continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def prepare_lstm_training_data(
        self,
        ticker: str,
        min_quarters: int = 20,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Prepare training data for LSTM in same format as Alpha Vantage fetcher
        
        Returns:
            Dictionary with:
            - ticker: Stock symbol
            - raw_data: Original quarterly data
            - normalized_data: Asset-normalized metrics
            - standardized_data: Standardized for training
            - normalization_params: For denormalization
            - quarters: Number of quarters
        """
        # Fetch financials
        financials_data = self.fetch_financials_reported(ticker, use_cache=use_cache)
        
        if not financials_data:
            return None
        
        # Parse to DataFrame
        df = self.parse_financials_to_dataframe(financials_data)
        
        if df is None or len(df) < min_quarters:
            logger.warning(f"Insufficient data for {ticker}: {len(df) if df is not None else 0} quarters (need {min_quarters})")
            return None
        
        # Calculate derived metrics
        df['tax_rate'] = df.apply(
            lambda row: abs(row['tax_expense'] / row['ebit']) if row['ebit'] != 0 else 0.21,
            axis=1
        )
        df['tax_rate'] = df['tax_rate'].clip(0, 0.50)  # Clamp between 0-50%
        df['nopat'] = df['ebit'] * (1 - df['tax_rate'])
        
        # Normalize by assets
        normalized_df = df.copy()
        for col in ['revenue', 'capex', 'da', 'nopat']:
            if col in df.columns and 'total_assets' in df.columns:
                normalized_df[f'{col}_norm'] = df[col] / (df['total_assets'] + 1e-6)  # Avoid division by zero
        
        # Standardize
        standardized_df = normalized_df.copy()
        params = {}
        
        for col in ['revenue_norm', 'capex_norm', 'da_norm', 'nopat_norm']:
            if col in normalized_df.columns:
                mean = normalized_df[col].mean()
                std = normalized_df[col].std()
                
                if std > 0:
                    standardized_df[f'{col}_std'] = (normalized_df[col] - mean) / std
                else:
                    standardized_df[f'{col}_std'] = 0
                
                params[col] = {'mean': mean, 'std': std}
        
        return {
            'ticker': ticker,
            'raw_data': df,
            'normalized_data': normalized_df,
            'standardized_data': standardized_df,
            'normalization_params': params,
            'quarters': len(df)
        }


# Example usage
if __name__ == "__main__":
    fetcher = FinnhubFinancialsFetcher()
    
    # Test fetch
    print("Testing Finnhub fetcher with AAPL:")
    print("=" * 60)
    
    data = fetcher.prepare_lstm_training_data('AAPL')
    
    if data:
        print(f"\n✅ Success!")
        print(f"Ticker: {data['ticker']}")
        print(f"Quarters: {data['quarters']}")
        print(f"\nRaw data (last 5):")
        print(data['raw_data'][['date', 'revenue', 'capex', 'da', 'nopat']].tail())
        print(f"\nStandardized data (last 5):")
        print(data['standardized_data'][['date', 'revenue_norm_std', 'capex_norm_std', 'da_norm_std', 'nopat_norm_std']].tail())
    else:
        print("❌ Failed to fetch data")
