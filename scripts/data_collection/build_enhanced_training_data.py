"""
Build Enhanced LSTM-DCF Training Dataset
Uses REAL financial statement data instead of price-based proxies
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def build_ticker_dataset(ticker: str, financial_statements_dir: Path) -> dict:
    """
    Build dataset for a single ticker from financial statements.
    
    Returns:
        dict with 'success', 'data', 'error' keys
    """
    result = {
        'ticker': ticker,
        'success': False,
        'data': None,
        'error': None
    }
    
    try:
        # Load financial statements
        income_path = financial_statements_dir / f'{ticker}_income.csv'
        cashflow_path = financial_statements_dir / f'{ticker}_cashflow.csv'
        balance_path = financial_statements_dir / f'{ticker}_balance.csv'
        
        if not all([income_path.exists(), cashflow_path.exists(), balance_path.exists()]):
            result['error'] = "Missing files"
            return result
        
        # Load only required columns for efficiency
        income_df = pd.read_csv(income_path)
        cashflow_df = pd.read_csv(cashflow_path)
        balance_df = pd.read_csv(balance_path)
        
        # Extract core columns
        income_subset = income_df[['fiscalDateEnding', 'totalRevenue', 'operatingIncome', 
                                     'depreciationAndAmortization', 'ebitda', 'netIncome']].copy()
        cashflow_subset = cashflow_df[['fiscalDateEnding', 'operatingCashflow', 'capitalExpenditures']].copy()
        balance_subset = balance_df[['fiscalDateEnding', 'totalAssets', 'totalLiabilities']].copy()
        
        # Merge
        df = income_subset.merge(cashflow_subset, on='fiscalDateEnding', how='inner') \
                          .merge(balance_subset, on='fiscalDateEnding', how='inner')
        
        # Sort by date
        df['date'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Convert to numeric
        df['revenue'] = pd.to_numeric(df['totalRevenue'], errors='coerce')
        df['operating_cf'] = pd.to_numeric(df['operatingCashflow'], errors='coerce')
        df['capex'] = pd.to_numeric(df['capitalExpenditures'], errors='coerce').abs()
        df['da'] = pd.to_numeric(df['depreciationAndAmortization'], errors='coerce')
        df['total_assets'] = pd.to_numeric(df['totalAssets'], errors='coerce')
        df['net_income'] = pd.to_numeric(df['netIncome'], errors='coerce')
        df['ebitda'] = pd.to_numeric(df['ebitda'], errors='coerce')
        df['operating_income'] = pd.to_numeric(df['operatingIncome'], errors='coerce')
        df['total_liabilities'] = pd.to_numeric(df['totalLiabilities'], errors='coerce')
        
        # Calculate FCF (KEY METRIC!)
        df['fcf'] = df['operating_cf'] - df['capex']
        
        # === PROFITABILITY MARGINS ===
        df['operating_margin'] = (df['operating_income'] / df['revenue']) * 100
        df['net_margin'] = (df['net_income'] / df['revenue']) * 100
        df['fcf_margin'] = (df['fcf'] / df['revenue']) * 100
        df['ebitda_margin'] = (df['ebitda'] / df['revenue']) * 100
        
        # === ASSET EFFICIENCY ===
        df['asset_turnover'] = df['revenue'] / df['total_assets']
        df['roa'] = (df['net_income'] / df['total_assets']) * 100
        
        # === NORMALIZE BY ASSETS (per research paper) ===
        df['revenue_per_asset'] = df['revenue'] / df['total_assets']
        df['capex_per_asset'] = df['capex'] / df['total_assets']
        df['da_per_asset'] = df['da'] / df['total_assets']
        df['fcf_per_asset'] = df['fcf'] / df['total_assets']
        df['ebitda_per_asset'] = df['ebitda'] / df['total_assets']
        
        # === GROWTH RATES (TARGET VARIABLES) ===
        df['revenue_growth'] = df['revenue'].pct_change() * 100
        df['fcf_growth'] = df['fcf'].pct_change() * 100
        df['capex_growth'] = df['capex'].pct_change() * 100
        df['da_growth'] = df['da'].pct_change() * 100
        df['ebitda_growth'] = df['ebitda'].pct_change() * 100
        
        # Year-over-year growth (4 quarters)
        df['revenue_growth_yoy'] = df['revenue'].pct_change(periods=4) * 100
        df['fcf_growth_yoy'] = df['fcf'].pct_change(periods=4) * 100
        
        # Drop rows with missing critical data
        critical_cols = ['revenue', 'fcf', 'total_assets']
        df = df.dropna(subset=critical_cols)
        
        # Select final features
        feature_cols = [
            'date', 'ticker',
            # Core metrics
            'revenue', 'capex', 'da', 'fcf', 'operating_cf', 'ebitda',
            'total_assets', 'net_income', 'operating_income',
            # Margins
            'operating_margin', 'net_margin', 'fcf_margin', 'ebitda_margin',
            # Asset efficiency
            'asset_turnover', 'roa',
            # Normalized
            'revenue_per_asset', 'capex_per_asset', 'da_per_asset', 'fcf_per_asset', 'ebitda_per_asset',
            # Growth rates (TARGET!)
            'revenue_growth', 'capex_growth', 'da_growth', 'fcf_growth', 'ebitda_growth',
            'revenue_growth_yoy', 'fcf_growth_yoy'
        ]
        
        df['ticker'] = ticker
        result['success'] = True
        result['data'] = df[feature_cols]
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    """Build enhanced training dataset from all available stocks"""
    
    logger.info("=" * 100)
    logger.info("BUILDING ENHANCED LSTM-DCF TRAINING DATASET")
    logger.info("=" * 100)
    
    # Setup paths
    financial_statements_dir = RAW_DATA_DIR / 'financial_statements'
    output_dir = PROCESSED_DATA_DIR / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available tickers
    files = list(financial_statements_dir.glob('*_income.csv'))
    tickers = sorted(set([f.name.split('_')[0] for f in files]))
    
    logger.info(f"\n📊 Found {len(tickers)} stocks with financial statements")
    logger.info(f"   Output: {output_dir / 'lstm_dcf_training_enhanced.csv'}")
    
    # Process all tickers
    all_data = []
    success_count = 0
    failed_tickers = []
    
    for ticker in tqdm(tickers, desc="Processing stocks"):
        result = build_ticker_dataset(ticker, financial_statements_dir)
        
        if result['success']:
            all_data.append(result['data'])
            success_count += 1
        else:
            failed_tickers.append((ticker, result['error']))
    
    # Combine all data
    if not all_data:
        logger.error("❌ No data processed successfully!")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save
    output_path = output_dir / 'lstm_dcf_training_enhanced.csv'
    combined_df.to_csv(output_path, index=False)
    
    # Summary
    logger.info(f"\n{'=' * 100}")
    logger.info(f"✅ DATASET BUILT SUCCESSFULLY")
    logger.info(f"{'=' * 100}")
    logger.info(f"\n📊 Statistics:")
    logger.info(f"   Stocks processed: {success_count}/{len(tickers)}")
    logger.info(f"   Total records: {len(combined_df):,}")
    logger.info(f"   Features: {combined_df.shape[1]}")
    logger.info(f"   Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
    logger.info(f"   Avg quarters per stock: {len(combined_df) / success_count:.1f}")
    
    logger.info(f"\n📋 Feature Summary:")
    logger.info(f"   Core metrics: revenue, capex, da, fcf, operating_cf, ebitda (9 features)")
    logger.info(f"   Margins: operating, net, fcf, ebitda (4 features)")
    logger.info(f"   Asset efficiency: turnover, roa (2 features)")
    logger.info(f"   Normalized: revenue/asset, capex/asset, etc. (5 features)")
    logger.info(f"   Growth rates: revenue, fcf, ebitda (7 features)")
    
    logger.info(f"\n🎯 Target Variables (Growth Rates):")
    for col in ['revenue_growth', 'fcf_growth', 'ebitda_growth']:
        stats = combined_df[col].dropna()
        logger.info(f"   {col:20s}: Mean={stats.mean():6.1f}%, Std={stats.std():6.1f}%, "
                   f"Range=[{stats.min():6.1f}%, {stats.max():6.1f}%]")
    
    logger.info(f"\n💾 Saved to: {output_path}")
    
    if failed_tickers:
        logger.warning(f"\n⚠️  Failed tickers ({len(failed_tickers)}):")
        for ticker, error in failed_tickers[:10]:
            logger.warning(f"   {ticker}: {error}")
    
    logger.info(f"\n{'=' * 100}")
    logger.info(f"✅ READY FOR LSTM TRAINING!")
    logger.info(f"{'=' * 100}")
    logger.info(f"\nNext step: python scripts/train_lstm_dcf_enhanced.py")


if __name__ == '__main__':
    main()
