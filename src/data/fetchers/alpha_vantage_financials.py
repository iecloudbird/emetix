"""
Alpha Vantage Financial Statements Fetcher for LSTM-DCF
Fetches quarterly financial data with proper rate limiting and caching

Data Coverage:
- 81 quarters (~20 years) per stock
- Income Statement: Revenue, EBIT, Tax Expense
- Cash Flow: CapEx, Depreciation & Amortization
- Balance Sheet: Total Assets (for normalization)

Rate Limits (Free Tier):
- 25 API calls per day
- 5 calls per minute
- For 500 stocks: ~60 days collection time
"""
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from alpha_vantage.fundamentaldata import FundamentalData
from dotenv import load_dotenv
from config.logging_config import get_logger
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

load_dotenv()
logger = get_logger(__name__)


class AlphaVantageFinancialsFetcher:
    """
    Fetch financial statements from Alpha Vantage for LSTM-DCF training
    
    Implements:
    - Smart rate limiting (5 calls/min, 25 calls/day)
    - Caching to avoid re-fetching
    - Data validation and cleaning
    - Normalization by assets (per article methodology)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found. Set it in .env file.")
        
        self.fd = FundamentalData(key=self.api_key, output_format='pandas')
        
        # Rate limiting
        self.calls_per_minute = 5
        self.calls_per_day = 25
        self.daily_calls = 0
        self.last_call_time = 0
        self.calls_this_minute = []
        
        # Caching
        self.cache_dir = RAW_DATA_DIR / "financial_statements"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Alpha Vantage Financial Fetcher initialized")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _wait_for_rate_limit(self):
        """Smart rate limiting: 5 calls/min, 25 calls/day"""
        current_time = time.time()
        
        # Check daily limit
        if self.daily_calls >= self.calls_per_day:
            logger.warning(f"Daily limit reached ({self.calls_per_day} calls). Stopping.")
            raise Exception("Daily API call limit reached. Try again tomorrow.")
        
        # Check minute limit
        self.calls_this_minute = [t for t in self.calls_this_minute if current_time - t < 60]
        
        if len(self.calls_this_minute) >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.calls_this_minute[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        # Record call
        self.calls_this_minute.append(current_time)
        self.daily_calls += 1
        self.last_call_time = current_time
    
    def _get_cache_path(self, ticker: str, statement_type: str) -> Path:
        """Get cache file path for a statement"""
        return self.cache_dir / f"{ticker}_{statement_type}.csv"
    
    def _load_from_cache(self, ticker: str, statement_type: str) -> Optional[pd.DataFrame]:
        """Load statement from cache if available"""
        cache_path = self._get_cache_path(ticker, statement_type)
        
        if cache_path.exists():
            logger.debug(f"Loading {ticker} {statement_type} from cache")
            return pd.read_csv(cache_path)
        
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, ticker: str, statement_type: str):
        """Save statement to cache"""
        cache_path = self._get_cache_path(ticker, statement_type)
        df.to_csv(cache_path, index=False)
        logger.debug(f"Cached {ticker} {statement_type}")
    
    def fetch_income_statement(self, ticker: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetch quarterly income statement"""
        if use_cache:
            cached = self._load_from_cache(ticker, 'income')
            if cached is not None:
                return cached
        
        try:
            self._wait_for_rate_limit()
            df, _ = self.fd.get_income_statement_quarterly(ticker)
            
            if df.empty:
                logger.warning(f"No income data for {ticker}")
                return None
            
            self._save_to_cache(df, ticker, 'income')
            logger.info(f"✓ Fetched income statement for {ticker} ({len(df)} quarters)")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching income for {ticker}: {e}")
            return None
    
    def fetch_cash_flow(self, ticker: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetch quarterly cash flow statement"""
        if use_cache:
            cached = self._load_from_cache(ticker, 'cashflow')
            if cached is not None:
                return cached
        
        try:
            self._wait_for_rate_limit()
            df, _ = self.fd.get_cash_flow_quarterly(ticker)
            
            if df.empty:
                logger.warning(f"No cash flow data for {ticker}")
                return None
            
            self._save_to_cache(df, ticker, 'cashflow')
            logger.info(f"✓ Fetched cash flow for {ticker} ({len(df)} quarters)")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching cash flow for {ticker}: {e}")
            return None
    
    def fetch_balance_sheet(self, ticker: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetch quarterly balance sheet"""
        if use_cache:
            cached = self._load_from_cache(ticker, 'balance')
            if cached is not None:
                return cached
        
        try:
            self._wait_for_rate_limit()
            df, _ = self.fd.get_balance_sheet_quarterly(ticker)
            
            if df.empty:
                logger.warning(f"No balance sheet data for {ticker}")
                return None
            
            self._save_to_cache(df, ticker, 'balance')
            logger.info(f"✓ Fetched balance sheet for {ticker} ({len(df)} quarters)")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            return None
    
    def extract_dcf_components(self, ticker: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Extract and combine DCF components from all statements
        
        Returns DataFrame with columns:
        - date: Fiscal quarter end
        - revenue: Total Revenue
        - capex: Capital Expenditures
        - da: Depreciation & Amortization
        - ebit: Operating Income
        - tax_rate: Effective tax rate
        - nopat: EBIT × (1 - tax_rate)
        - total_assets: For normalization
        """
        # Fetch all three statements
        income = self.fetch_income_statement(ticker, use_cache)
        cashflow = self.fetch_cash_flow(ticker, use_cache)
        balance = self.fetch_balance_sheet(ticker, use_cache)
        
        if income is None or cashflow is None or balance is None:
            logger.warning(f"Missing financial statements for {ticker}")
            return None
        
        # Merge on fiscal date
        data = []
        
        # Iterate through quarters (assuming all have same dates)
        for i in range(len(income)):
            try:
                date = income.iloc[i]['fiscalDateEnding']
                
                # Find matching rows in other statements
                cf_row = cashflow[cashflow['fiscalDateEnding'] == date]
                bal_row = balance[balance['fiscalDateEnding'] == date]
                
                if cf_row.empty or bal_row.empty:
                    continue
                
                # Extract values
                revenue = float(income.iloc[i].get('totalRevenue', 0))
                ebit = float(income.iloc[i].get('operatingIncome', 0))
                tax_expense = float(income.iloc[i].get('incomeTaxExpense', 0))
                pretax_income = float(income.iloc[i].get('incomeBeforeTax', ebit))
                
                capex = abs(float(cf_row.iloc[0].get('capitalExpenditures', 0)))
                da = float(cf_row.iloc[0].get('depreciationDepletionAndAmortization', 0))
                
                total_assets = float(bal_row.iloc[0].get('totalAssets', 1))
                
                # Calculate tax rate
                tax_rate = abs(tax_expense / pretax_income) if pretax_income != 0 else 0.21
                tax_rate = min(max(tax_rate, 0), 0.50)  # Clamp 0-50%
                
                # Calculate NOPAT
                nopat = ebit * (1 - tax_rate)
                
                data.append({
                    'date': pd.to_datetime(date),
                    'revenue': revenue,
                    'capex': capex,
                    'da': da,
                    'ebit': ebit,
                    'tax_rate': tax_rate,
                    'nopat': nopat,
                    'total_assets': total_assets
                })
                
            except Exception as e:
                logger.debug(f"Error processing quarter {i} for {ticker}: {e}")
                continue
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✓ Extracted {len(df)} quarters of DCF components for {ticker}")
        return df
    
    def prepare_lstm_training_data(
        self,
        ticker: str,
        min_quarters: int = 20,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Prepare complete LSTM training data for a stock
        
        Following article methodology:
        1. Extract DCF components
        2. Normalize by total assets
        3. Standardize (mean=0, std=1)
        
        Returns:
            Dictionary with:
            - ticker: Stock symbol
            - raw_data: Original values
            - normalized_data: Asset-normalized
            - standardized_data: Ready for LSTM
            - norm_params: For denormalization
        """
        # Get DCF components
        df = self.extract_dcf_components(ticker, use_cache)
        
        if df is None or len(df) < min_quarters:
            logger.warning(f"Insufficient data for {ticker}: {len(df) if df is not None else 0} quarters (need {min_quarters})")
            return None
        
        # Normalize by assets
        normalized_df = df.copy()
        for col in ['revenue', 'capex', 'da', 'nopat']:
            normalized_df[f'{col}_norm'] = df[col] / df['total_assets']
        
        # Standardize (mean=0, std=1)
        standardized_df = normalized_df.copy()
        norm_params = {}
        
        for col in ['revenue_norm', 'capex_norm', 'da_norm', 'nopat_norm']:
            mean = normalized_df[col].mean()
            std = normalized_df[col].std()
            
            standardized_df[f'{col}_std'] = (normalized_df[col] - mean) / std if std > 0 else 0
            norm_params[col] = {'mean': mean, 'std': std}
        
        return {
            'ticker': ticker,
            'raw_data': df,
            'normalized_data': normalized_df,
            'standardized_data': standardized_df,
            'norm_params': norm_params,
            'quarters': len(df)
        }
    
    def batch_fetch(
        self,
        tickers: List[str],
        max_stocks_per_session: int = 8,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Batch fetch financial data with daily limit management
        
        Args:
            tickers: List of stock symbols
            max_stocks_per_session: Max stocks to process (respects 25/day limit)
            use_cache: Use cached data when available
        
        Returns:
            Combined DataFrame ready for LSTM training
        """
        logger.info(f"Batch fetching {len(tickers)} stocks (max {max_stocks_per_session}/session)")
        
        all_data = []
        stocks_processed = 0
        
        for ticker in tickers:
            if stocks_processed >= max_stocks_per_session:
                logger.warning(f"Reached session limit ({max_stocks_per_session} stocks). Stopping.")
                break
            
            logger.info(f"\nProcessing {ticker} ({stocks_processed+1}/{min(len(tickers), max_stocks_per_session)})...")
            
            data = self.prepare_lstm_training_data(ticker, use_cache=use_cache)
            
            if data is None:
                continue
            
            # Extract standardized metrics for training
            std_df = data['standardized_data']
            
            training_df = pd.DataFrame({
                'ticker': ticker,
                'date': std_df['date'],
                'revenue_std': std_df['revenue_norm_std'],
                'capex_std': std_df['capex_norm_std'],
                'da_std': std_df['da_norm_std'],
                'nopat_std': std_df['nopat_norm_std']
            })
            
            all_data.append(training_df)
            stocks_processed += 1
        
        if not all_data:
            logger.error("No data collected!")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Batch fetch complete:")
        logger.info(f"  Stocks processed: {stocks_processed}")
        logger.info(f"  Total records: {len(combined_df)}")
        logger.info(f"  API calls used: {self.daily_calls}/25")
        logger.info(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"{'='*80}")
        
        return combined_df


if __name__ == "__main__":
    print("Alpha Vantage Financial Statements Fetcher")
    print("="*80)
    
    # Initialize
    fetcher = AlphaVantageFinancialsFetcher()
    
    # Test with single stock
    print("\n📊 Testing with AAPL...")
    data = fetcher.prepare_lstm_training_data('AAPL', use_cache=False)
    
    if data:
        print(f"\n✅ Success!")
        print(f"  Quarters: {data['quarters']}")
        print(f"\nRaw Data (latest 5 quarters):")
        print(data['raw_data'][['date', 'revenue', 'capex', 'da', 'nopat']].head())
        
        print(f"\nNormalized Data:")
        print(data['normalized_data'][['date', 'revenue_norm', 'capex_norm', 'da_norm', 'nopat_norm']].head())
        
        print(f"\nStandardized Data (ready for LSTM):")
        print(data['standardized_data'][['date', 'revenue_norm_std', 'capex_norm_std', 'da_norm_std', 'nopat_norm_std']].head())
    
    # Test batch fetch
    print("\n" + "="*80)
    print("📦 Testing batch fetch (3 stocks)...")
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    combined = fetcher.batch_fetch(test_tickers, max_stocks_per_session=3, use_cache=True)
    
    if not combined.empty:
        print(f"\n✅ Batch fetch successful!")
        print(f"  Records: {len(combined)}")
        print(f"  Stocks: {combined['ticker'].nunique()}")
        print(f"\nSample:")
        print(combined.head(10))
