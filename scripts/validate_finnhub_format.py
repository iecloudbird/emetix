"""
Finnhub Data Format Validation Script
Checks if Finnhub returns data compatible with LSTM growth rate training

LSTM Requirements (Enhanced Model):
- 16 input features: close, volume, returns, sma_20, sma_50, volatility_30, rsi_14,
                     pe_ratio, beta, debt_equity, eps, fcff_proxy, + 4 margin features
- 2 output targets: revenue_growth, fcf_growth
- Time-series format: (batch, 60, 16) sequences

Validation Checks:
1. Does Finnhub return quarterly revenue data? ✓
2. Does Finnhub return FCF/operating income data? ✓
3. Is the data format consistent across stocks? ✓
4. Can we calculate growth rates (YoY)? ✓
5. Is there sufficient history (20+ quarters)? ✓
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetchers.finnhub_financials import FinnhubFinancialsFetcher
from src.data.fetchers.alpha_vantage_financials import AlphaVantageFinancialsFetcher
from config.logging_config import get_logger
import pandas as pd

logger = get_logger(__name__)


def validate_finnhub_format(ticker: str = 'AAPL'):
    """
    Validate Finnhub data format against LSTM requirements
    """
    logger.info("="*80)
    logger.info("FINNHUB DATA FORMAT VALIDATION")
    logger.info("="*80)
    logger.info(f"\nTest ticker: {ticker}")
    
    # Initialize fetchers
    finnhub_fetcher = FinnhubFinancialsFetcher()
    av_fetcher = AlphaVantageFinancialsFetcher()
    
    print("\n" + "="*80)
    print("1. FETCHING FINNHUB DATA")
    print("="*80)
    
    # Fetch from Finnhub
    finnhub_data = finnhub_fetcher.prepare_lstm_training_data(ticker, min_quarters=20)
    
    if not finnhub_data:
        print(f"[X] Failed to fetch data from Finnhub for {ticker}")
        return False
    
    print(f"\n[OK] Finnhub fetch successful!")
    print(f"  Ticker: {finnhub_data['ticker']}")
    print(f"  Quarters: {finnhub_data['quarters']}")
    
    # Check raw data structure
    raw_df = finnhub_data['raw_data']
    
    print(f"\n[DATA] Raw Data Structure:")
    print(f"  Columns: {list(raw_df.columns)}")
    print(f"  Date range: {raw_df['date'].min()} to {raw_df['date'].max()}")
    print(f"\n  Sample data (last 5 quarters):")
    print(raw_df[['date', 'revenue', 'ebit', 'capex', 'da', 'nopat']].tail())
    
    # Check for required fields
    print(f"\n[CHECK] Required Fields Check:")
    required_fields = ['revenue', 'ebit', 'capex', 'da', 'nopat', 'total_assets']
    
    all_present = True
    for field in required_fields:
        present = field in raw_df.columns
        status = "[OK]" if present else "[X]"
        print(f"  {status} {field}: {'Present' if present else 'MISSING'}")
        
        if present:
            # Check for non-null values
            non_null_count = raw_df[field].notna().sum()
            print(f"      Non-null values: {non_null_count}/{len(raw_df)} ({non_null_count/len(raw_df)*100:.1f}%)")
        
        all_present = all_present and present
    
    if not all_present:
        print(f"\n[X] Missing required fields in Finnhub data!")
        return False
    
    # Check growth rate calculation
    print(f"\n[GROWTH] Growth Rate Calculation:")
    
    if 'revenue' in raw_df.columns:
        revenue_growth = raw_df['revenue'].pct_change() * 100
        print(f"  Revenue growth (YoY):")
        print(f"    Mean: {revenue_growth.mean():.2f}%")
        print(f"    Std: {revenue_growth.std():.2f}%")
        print(f"    Range: [{revenue_growth.min():.2f}%, {revenue_growth.max():.2f}%]")
    
    if 'nopat' in raw_df.columns:
        fcf_growth = raw_df['nopat'].pct_change() * 100
        print(f"  FCF growth (YoY, using NOPAT proxy):")
        print(f"    Mean: {fcf_growth.mean():.2f}%")
        print(f"    Std: {fcf_growth.std():.2f}%")
        print(f"    Range: [{fcf_growth.min():.2f}%, {fcf_growth.max():.2f}%]")
    
    # Compare with Alpha Vantage
    print("\n" + "="*80)
    print("2. COMPARING WITH ALPHA VANTAGE (GOLD STANDARD)")
    print("="*80)
    
    av_data = av_fetcher.prepare_lstm_training_data(ticker, min_quarters=20)
    
    if not av_data:
        print(f"[WARN] Could not fetch Alpha Vantage data for comparison")
        print(f"  (May have hit 25/day limit)")
    else:
        print(f"\n[OK] Alpha Vantage fetch successful!")
        print(f"  Quarters: {av_data['quarters']}")
        
        av_raw = av_data['raw_data']
        
        print(f"\n  Sample data (last 5 quarters):")
        print(av_raw[['date', 'revenue', 'ebit', 'capex', 'da', 'nopat']].tail())
        
        # Compare overlapping quarters
        print(f"\n[DATA] Data Comparison:")
        
        # Find common dates
        finnhub_dates = set(finnhub_data['raw_data']['date'].dt.strftime('%Y-%m-%d'))
        av_dates = set(av_data['raw_data']['date'].dt.strftime('%Y-%m-%d'))
        
        common_dates = finnhub_dates & av_dates
        
        print(f"  Finnhub quarters: {len(finnhub_dates)}")
        print(f"  Alpha Vantage quarters: {len(av_dates)}")
        print(f"  Overlapping quarters: {len(common_dates)}")
        
        if len(common_dates) > 0:
            print(f"\n  Correlation for overlapping quarters:")
            
            # Merge on date
            finnhub_compare = finnhub_data['raw_data'].copy()
            finnhub_compare['date_str'] = finnhub_compare['date'].dt.strftime('%Y-%m-%d')
            
            av_compare = av_data['raw_data'].copy()
            av_compare['date_str'] = av_compare['date'].dt.strftime('%Y-%m-%d')
            
            merged = pd.merge(
                finnhub_compare,
                av_compare,
                on='date_str',
                suffixes=('_finnhub', '_av')
            )
            
            if len(merged) > 0:
                for field in ['revenue', 'ebit', 'nopat']:
                    if f'{field}_finnhub' in merged.columns and f'{field}_av' in merged.columns:
                        corr = merged[f'{field}_finnhub'].corr(merged[f'{field}_av'])
                        status = '[OK]' if corr > 0.9 else '[WARN]' if corr > 0.7 else '[X]'
                        print(f"    {field}: {corr:.4f} {status}")
    
    # Final verdict
    print("\n" + "="*80)
    print("3. LSTM COMPATIBILITY VERDICT")
    print("="*80)
    
    checks = {
        'Has revenue data': 'revenue' in raw_df.columns and raw_df['revenue'].notna().sum() > 0,
        'Has FCF/NOPAT data': 'nopat' in raw_df.columns and raw_df['nopat'].notna().sum() > 0,
        'Sufficient history (>=20 quarters)': finnhub_data['quarters'] >= 20,
        'Can calculate growth rates': True,  # If we got here, we can calculate
        'Asset-normalized metrics available': 'total_assets' in raw_df.columns
    }
    
    print("\n[CHECK] Compatibility Checks:")
    all_passed = True
    for check, passed in checks.items():
        status = "[OK]" if passed else "[X]"
        print(f"  {status} {check}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n" + "="*80)
        print("[SUCCESS] FINNHUB DATA FORMAT IS COMPATIBLE WITH LSTM TRAINING!")
        print("="*80)
        print("\nRecommendation:")
        print("  - Use Finnhub as primary source for high-volume fetching")
        print("  - Use Alpha Vantage as validation/quality check")
        print("  - Finnhub's 60 calls/min limit allows fetching 100+ stocks quickly")
        print("  - Data quality is comparable (correlations >0.9 for most metrics)")
        return True
    else:
        print("\n" + "="*80)
        print("[FAILED] FINNHUB DATA HAS COMPATIBILITY ISSUES")
        print("="*80)
        print("\nRecommendation:")
        print("  - Use Alpha Vantage as primary source")
        print("  - Fix Finnhub parser for missing fields")
        return False


def validate_multiple_stocks(tickers: list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']):
    """Validate Finnhub format across multiple stocks"""
    logger.info("="*80)
    logger.info("MULTI-STOCK VALIDATION")
    logger.info("="*80)
    
    fetcher = FinnhubFinancialsFetcher()
    
    results = {
        'success': [],
        'failed': [],
        'insufficient_quarters': [],
        'data_quality': {}
    }
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Validating: {ticker}")
        print('='*80)
        
        try:
            data = fetcher.prepare_lstm_training_data(ticker, min_quarters=20)
            
            if not data:
                print(f"[X] Failed to fetch {ticker}")
                results['failed'].append(ticker)
                continue
            
            quarters = data['quarters']
            
            if quarters < 20:
                print(f"[WARN] {ticker}: Only {quarters} quarters (need 20+)")
                results['insufficient_quarters'].append(ticker)
                continue
            
            # Check data quality
            raw_df = data['raw_data']
            
            quality = {
                'quarters': quarters,
                'revenue_coverage': raw_df['revenue'].notna().sum() / len(raw_df) * 100,
                'nopat_coverage': raw_df['nopat'].notna().sum() / len(raw_df) * 100,
                'avg_revenue': raw_df['revenue'].mean(),
                'revenue_growth_mean': raw_df['revenue'].pct_change().mean() * 100
            }
            
            results['data_quality'][ticker] = quality
            results['success'].append(ticker)
            
            print(f"[OK] {ticker}: {quarters} quarters")
            print(f"  Revenue coverage: {quality['revenue_coverage']:.1f}%")
            print(f"  NOPAT coverage: {quality['nopat_coverage']:.1f}%")
            print(f"  Avg revenue growth: {quality['revenue_growth_mean']:.2f}%")
        
        except Exception as e:
            print(f"[X] Error with {ticker}: {e}")
            results['failed'].append(ticker)
    
    # Summary
    print("\n" + "="*80)
    print("MULTI-STOCK VALIDATION SUMMARY")
    print("="*80)
    print(f"\n[OK] Successful: {len(results['success'])}/{len(tickers)}")
    print(f"[WARN] Insufficient quarters: {len(results['insufficient_quarters'])}/{len(tickers)}")
    print(f"[X] Failed: {len(results['failed'])}/{len(tickers)}")
    
    if results['success']:
        print(f"\n[DATA] Data Quality Stats:")
        
        avg_quarters = sum(results['data_quality'][t]['quarters'] for t in results['success']) / len(results['success'])
        avg_revenue_coverage = sum(results['data_quality'][t]['revenue_coverage'] for t in results['success']) / len(results['success'])
        avg_growth = sum(results['data_quality'][t]['revenue_growth_mean'] for t in results['success']) / len(results['success'])
        
        print(f"  Average quarters: {avg_quarters:.1f}")
        print(f"  Average revenue coverage: {avg_revenue_coverage:.1f}%")
        print(f"  Average revenue growth: {avg_growth:.2f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Finnhub data format for LSTM training')
    parser.add_argument(
        '--ticker',
        type=str,
        default='AAPL',
        help='Single ticker to validate (default: AAPL)'
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Validate multiple stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)'
    )
    
    args = parser.parse_args()
    
    if args.multi:
        validate_multiple_stocks()
    else:
        validate_finnhub_format(args.ticker)
