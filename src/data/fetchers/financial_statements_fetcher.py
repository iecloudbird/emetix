"""
Financial Statements Fetcher for LSTM-DCF Training
Fetches time-series data for:
- Revenue
- Capital Expenditures (CapEx)
- Depreciation & Amortization (D&A)
- NOPAT (Net Operating Profit After Tax)

Based on: https://www.revocm.com/articles/lstm-networks-estimating-growth-rates-dcf-models
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)


class FinancialStatementsFetcher:
    """Fetch financial statement data for LSTM-DCF model training"""
    
    def __init__(self):
        self.logger = logger
    
    def fetch_quarterly_financials(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch quarterly financial statement data
        
        Returns DataFrame with columns:
        - date: Quarter end date
        - revenue: Total Revenue
        - capex: Capital Expenditures
        - da: Depreciation & Amortization
        - ebit: Operating Income (EBIT)
        - tax_rate: Effective Tax Rate
        - nopat: EBIT * (1 - tax_rate)
        - total_assets: For normalization
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly financials
            qtr_financials = stock.quarterly_financials
            qtr_cashflow = stock.quarterly_cashflow
            qtr_balance = stock.quarterly_balance_sheet
            
            if qtr_financials.empty or qtr_cashflow.empty:
                self.logger.warning(f"No financial data for {ticker}")
                return None
            
            # Extract components (yfinance returns columns as dates)
            data = []
            
            for date in qtr_financials.columns:
                try:
                    # Revenue
                    revenue = qtr_financials.loc['Total Revenue', date] if 'Total Revenue' in qtr_financials.index else 0
                    
                    # CapEx (Capital Expenditure - negative in cash flow statement)
                    capex = abs(qtr_cashflow.loc['Capital Expenditure', date]) if 'Capital Expenditure' in qtr_cashflow.index else 0
                    
                    # Depreciation & Amortization
                    da = qtr_cashflow.loc['Depreciation And Amortization', date] if 'Depreciation And Amortization' in qtr_cashflow.index else 0
                    
                    # EBIT (Operating Income)
                    ebit = qtr_financials.loc['Operating Income', date] if 'Operating Income' in qtr_financials.index else 0
                    
                    # Tax provision and pretax income for tax rate
                    tax_provision = qtr_financials.loc['Tax Provision', date] if 'Tax Provision' in qtr_financials.index else 0
                    pretax_income = qtr_financials.loc['Pretax Income', date] if 'Pretax Income' in qtr_financials.index else ebit
                    
                    # Effective tax rate
                    tax_rate = abs(tax_provision / pretax_income) if pretax_income != 0 else 0.21  # Default 21%
                    tax_rate = min(max(tax_rate, 0), 0.50)  # Clamp between 0-50%
                    
                    # NOPAT = EBIT * (1 - tax_rate)
                    nopat = ebit * (1 - tax_rate)
                    
                    # Total Assets for normalization
                    total_assets = qtr_balance.loc['Total Assets', date] if 'Total Assets' in qtr_balance.index else revenue * 2  # Fallback
                    
                    data.append({
                        'date': date,
                        'revenue': revenue,
                        'capex': capex,
                        'da': da,
                        'ebit': ebit,
                        'tax_rate': tax_rate,
                        'nopat': nopat,
                        'total_assets': total_assets
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error processing quarter {date} for {ticker}: {e}")
                    continue
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            
            self.logger.info(f"Fetched {len(df)} quarters of financial data for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching financials for {ticker}: {str(e)}")
            return None
    
    def normalize_by_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize financial metrics by total assets
        
        Following the article's approach:
        "normalize it by the asset value at that time, to account for the company's scale"
        """
        normalized_df = df.copy()
        
        for col in ['revenue', 'capex', 'da', 'nopat']:
            if col in df.columns:
                normalized_df[f'{col}_norm'] = df[col] / df['total_assets']
        
        return normalized_df
    
    def standardize_metrics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Standardize normalized metrics (mean=0, std=1)
        
        Returns:
            - Standardized DataFrame
            - Dictionary with mean/std for each metric
        """
        standardized_df = df.copy()
        params = {}
        
        for col in ['revenue_norm', 'capex_norm', 'da_norm', 'nopat_norm']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                standardized_df[f'{col}_std'] = (df[col] - mean) / std if std > 0 else 0
                
                params[col] = {'mean': mean, 'std': std}
        
        return standardized_df, params
    
    def prepare_training_data(
        self,
        ticker: str,
        min_quarters: int = 20  # ~5 years minimum
    ) -> Optional[Dict]:
        """
        Prepare complete training data for LSTM model
        
        Returns:
            Dictionary with:
            - ticker: Stock symbol
            - raw_data: Original quarterly data
            - normalized_data: Asset-normalized metrics
            - standardized_data: Standardized for training
            - normalization_params: For denormalization
        """
        # Fetch quarterly financials
        df = self.fetch_quarterly_financials(ticker)
        
        if df is None or len(df) < min_quarters:
            self.logger.warning(f"Insufficient data for {ticker}: {len(df) if df is not None else 0} quarters")
            return None
        
        # Normalize by assets
        normalized_df = self.normalize_by_assets(df)
        
        # Standardize
        standardized_df, params = self.standardize_metrics(normalized_df)
        
        return {
            'ticker': ticker,
            'raw_data': df,
            'normalized_data': normalized_df,
            'standardized_data': standardized_df,
            'normalization_params': params
        }
    
    def fetch_current_metrics(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Fetch current/latest financial metrics for valuation
        
        Returns:
            Dictionary with latest values for FCFF calculation
        """
        df = self.fetch_quarterly_financials(ticker)
        
        if df is None or len(df) == 0:
            return None
        
        # Get most recent quarter
        latest = df.iloc[-1]
        
        # Get stock info for shares outstanding
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            shares = info.get('sharesOutstanding', 1e9)
            market_cap = info.get('marketCap', 0)
        except:
            shares = 1e9
            market_cap = 0
        
        return {
            'revenue': latest['revenue'],
            'capex': latest['capex'],
            'da': latest['da'],
            'nopat': latest['nopat'],
            'ebit': latest['ebit'],
            'tax_rate': latest['tax_rate'],
            'total_assets': latest['total_assets'],
            'shares_outstanding': shares,
            'market_cap': market_cap
        }
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        min_quarters: int = 20
    ) -> pd.DataFrame:
        """
        Fetch and combine financial data for multiple tickers
        
        Returns:
            Combined DataFrame for LSTM training with columns:
            [ticker, date, revenue_std, capex_std, da_std, nopat_std]
        """
        all_data = []
        
        for ticker in tickers:
            self.logger.info(f"Processing {ticker}...")
            
            data = self.prepare_training_data(ticker, min_quarters)
            
            if data is None:
                continue
            
            # Extract standardized metrics
            std_df = data['standardized_data']
            
            # Keep only standardized columns
            training_df = pd.DataFrame({
                'ticker': ticker,
                'date': std_df['date'],
                'revenue_std': std_df['revenue_norm_std'],
                'capex_std': std_df['capex_norm_std'],
                'da_std': std_df['da_norm_std'],
                'nopat_std': std_df['nopat_norm_std']
            })
            
            all_data.append(training_df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined data: {len(combined_df)} records from {len(all_data)} tickers")
        
        return combined_df


# Example usage
if __name__ == "__main__":
    fetcher = FinancialStatementsFetcher()
    
    # Test single ticker
    print("Testing AAPL financial data fetch:")
    print("=" * 60)
    
    data = fetcher.prepare_training_data('AAPL')
    
    if data:
        print(f"\nRaw Data (last 5 quarters):")
        print(data['raw_data'].tail())
        
        print(f"\nNormalized Data (last 5 quarters):")
        print(data['normalized_data'][['date', 'revenue_norm', 'capex_norm', 'da_norm', 'nopat_norm']].tail())
        
        print(f"\nStandardized Data (last 5 quarters):")
        print(data['standardized_data'][['date', 'revenue_norm_std', 'capex_norm_std', 'da_norm_std', 'nopat_norm_std']].tail())
        
        print(f"\nNormalization Parameters:")
        for key, params in data['normalization_params'].items():
            print(f"  {key}: mean={params['mean']:.6f}, std={params['std']:.6f}")
    
    # Test current metrics
    print("\n" + "=" * 60)
    print("Current Metrics:")
    current = fetcher.fetch_current_metrics('AAPL')
    if current:
        for key, value in current.items():
            if 'rate' in key:
                print(f"  {key}: {value*100:.2f}%")
            elif value > 1e9:
                print(f"  {key}: ${value/1e9:.2f}B")
            else:
                print(f"  {key}: {value:,.0f}")
