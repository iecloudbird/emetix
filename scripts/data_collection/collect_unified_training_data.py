"""
Unified Continuous Data Collection for LSTM-DCF and RF Ensemble Training
=========================================================================

This script collects aligned training data for both ML models:

LSTM-DCF Enhanced (16 features - Quarterly Financials):
- revenue, capex, da, fcf, operating_cf, ebitda
- total_assets, net_income, operating_income
- operating_margin, net_margin, fcf_margin, ebitda_margin
- revenue_per_asset, fcf_per_asset, ebitda_per_asset

RF Ensemble (14 features - Risk/Sentiment):
- beta, debt_to_equity, 30d_volatility, volume_zscore, short_percent
- rsi_14, sentiment_mean, sentiment_std, news_volume, relevance_mean
- pe_ratio, revenue_growth, current_ratio, return_on_equity

Usage:
    # Continuous collection (runs daily/weekly)
    python scripts/data_collection/collect_unified_training_data.py --mode continuous

    # Full batch collection
    python scripts/data_collection/collect_unified_training_data.py --mode full --batch-size 100

    # LSTM only (quarterly financials)
    python scripts/data_collection/collect_unified_training_data.py --mode lstm --batch-size 50

    # RF only (risk/sentiment)
    python scripts/data_collection/collect_unified_training_data.py --mode rf --batch-size 100

    # Validate existing data quality
    python scripts/data_collection/collect_unified_training_data.py --validate

    # Show collection progress
    python scripts/data_collection/collect_unified_training_data.py --status
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


# Comprehensive ticker universe (170 tickers for diverse training)
SP500_CORE = [
    # Technology (30 tickers)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'AVGO', 'CSCO', 
    'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'TXN', 'QCOM', 'IBM',
    'NOW', 'INTU', 'AMAT', 'ADI', 'LRCX', 'MU', 'SNPS', 'CDNS',
    'KLAC', 'MCHP', 'FTNT', 'PANW', 'CRWD', 'NXPI',
    
    # Healthcare (25 tickers)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
    'BMY', 'AMGN', 'MDT', 'GILD', 'CVS', 'ISRG', 'VRTX', 'REGN',
    'SYK', 'BSX', 'ZTS', 'BDX', 'CI', 'HUM', 'ELV', 'MCK',
    
    # Consumer Discretionary (20 tickers)
    'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'COST', 
    'WMT', 'ORLY', 'AZO', 'ROST', 'TJX', 'LULU', 'BBY', 'DG',
    'DLTR', 'POOL', 'ULTA', 'DECK',
    
    # Consumer Staples (15 tickers)
    'PG', 'KO', 'PEP', 'MDLZ', 'MO', 'PM', 'CL', 'EL', 'GIS',
    'KHC', 'SYY', 'STZ', 'HSY', 'K', 'KMB',
    
    # Financial (20 tickers)
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
    'AXP', 'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC', 'TFC',
    'CME', 'ICE', 'SPGI', 'MCO',
    
    # Industrial (20 tickers)
    'CAT', 'DE', 'UNP', 'HON', 'UPS', 'RTX', 'BA', 'LMT',
    'GE', 'MMM', 'EMR', 'ITW', 'ETN', 'PH', 'CMI', 'PCAR',
    'FDX', 'GWW', 'FAST', 'ODFL',
    
    # Energy (10 tickers)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    'OXY', 'HAL',
    
    # Utilities (8 tickers)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
    
    # Materials (8 tickers)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE',
    
    # Real Estate (8 tickers)
    'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'SPG', 'O', 'WELL',
    
    # Communication (6 tickers)
    'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
]


class UnifiedDataCollector:
    """
    Collects training data for both LSTM-DCF and RF Ensemble models
    with proper feature alignment.
    """
    
    # LSTM-DCF requires 16 quarterly financial features
    LSTM_FEATURES = [
        'revenue', 'capex', 'da', 'fcf', 'operating_cf', 'ebitda',
        'total_assets', 'net_income', 'operating_income',
        'operating_margin', 'net_margin', 'fcf_margin', 'ebitda_margin',
        'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
    ]
    
    # RF Ensemble requires 14 risk/sentiment features
    RF_FEATURES = [
        'beta', 'debt_to_equity', '30d_volatility', 'volume_zscore', 'short_percent',
        'rsi_14', 'sentiment_mean', 'sentiment_std', 'news_volume', 'relevance_mean',
        'pe_ratio', 'revenue_growth', 'current_ratio', 'return_on_equity'
    ]
    
    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or SP500_CORE
        
        # Output directories
        self.lstm_data_dir = PROCESSED_DATA_DIR / "lstm_dcf_training"
        self.rf_data_dir = PROCESSED_DATA_DIR / "rf_training"
        self.raw_financials_dir = RAW_DATA_DIR / "quarterly_financials"
        self.raw_risk_dir = RAW_DATA_DIR / "risk_sentiment"
        
        for d in [self.lstm_data_dir, self.rf_data_dir, 
                  self.raw_financials_dir, self.raw_risk_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = PROCESSED_DATA_DIR / "unified_collection_progress.json"
        self.progress = self._load_progress()
        
        logger.info(f"UnifiedDataCollector initialized with {len(self.tickers)} tickers")
    
    def _load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'started': datetime.now().isoformat(),
            'lstm_collected': [],
            'rf_collected': [],
            'failed': {},
            'last_full_run': None,
            'quality_stats': {}
        }
    
    def _save_progress(self):
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)
    
    def collect_lstm_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Collect quarterly financial features for LSTM-DCF training.
        
        The LSTM model expects sequences of quarterly financial data:
        - Minimum 4 quarters (1 year) for basic training
        - Ideally 8+ quarters (2 years) for better sequences
        - 16 financial features per quarter
        
        Returns:
            DataFrame with quarterly features or None if insufficient data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly financial statements
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow
            
            if income_stmt.empty or cash_flow.empty:
                logger.debug(f"{ticker}: No quarterly financials")
                return None
            
            n_quarters = min(len(income_stmt.columns), len(cash_flow.columns), 20)
            
            # Relaxed to 4 quarters minimum (yfinance often returns only 4-6)
            if n_quarters < 4:
                logger.debug(f"{ticker}: Only {n_quarters} quarters (need 4+)")
                return None
            
            records = []
            
            for i in range(n_quarters):
                try:
                    col_is = income_stmt.columns[i] if i < len(income_stmt.columns) else None
                    col_cf = cash_flow.columns[i] if i < len(cash_flow.columns) else None
                    col_bs = balance_sheet.columns[i] if i < len(balance_sheet.columns) else None
                    
                    if col_is is None or col_cf is None:
                        continue
                    
                    # Extract financial statement items
                    revenue = self._safe_get(income_stmt, 'Total Revenue', col_is, 0)
                    net_income = self._safe_get(income_stmt, 'Net Income', col_is, 0)
                    operating_income = self._safe_get(income_stmt, 'Operating Income', col_is, 0)
                    ebitda = self._safe_get(income_stmt, 'EBITDA', col_is,
                              self._safe_get(income_stmt, 'Normalized EBITDA', col_is, 0))
                    
                    operating_cf = self._safe_get(cash_flow, 'Operating Cash Flow', col_cf,
                                    self._safe_get(cash_flow, 'Cash Flow From Continuing Operating Activities', col_cf, 0))
                    capex = abs(self._safe_get(cash_flow, 'Capital Expenditure', col_cf, 0))
                    da = self._safe_get(cash_flow, 'Depreciation And Amortization', col_cf,
                          self._safe_get(cash_flow, 'Depreciation', col_cf, 0))
                    
                    fcf = operating_cf - capex
                    total_assets = self._safe_get(balance_sheet, 'Total Assets', col_bs, 1) if col_bs else 1
                    
                    # Calculate margins
                    operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
                    net_margin = (net_income / revenue * 100) if revenue > 0 else 0
                    fcf_margin = (fcf / revenue * 100) if revenue > 0 else 0
                    ebitda_margin = (ebitda / revenue * 100) if revenue > 0 else 0
                    
                    # Asset efficiency
                    revenue_per_asset = revenue / total_assets if total_assets > 0 else 0
                    fcf_per_asset = fcf / total_assets if total_assets > 0 else 0
                    ebitda_per_asset = ebitda / total_assets if total_assets > 0 else 0
                    
                    records.append({
                        'quarter_date': col_is,
                        'ticker': ticker,
                        'revenue': revenue,
                        'capex': capex,
                        'da': da,
                        'fcf': fcf,
                        'operating_cf': operating_cf,
                        'ebitda': ebitda,
                        'total_assets': total_assets,
                        'net_income': net_income,
                        'operating_income': operating_income,
                        'operating_margin': operating_margin,
                        'net_margin': net_margin,
                        'fcf_margin': fcf_margin,
                        'ebitda_margin': ebitda_margin,
                        'revenue_per_asset': revenue_per_asset,
                        'fcf_per_asset': fcf_per_asset,
                        'ebitda_per_asset': ebitda_per_asset
                    })
                    
                except Exception as e:
                    logger.debug(f"{ticker} Q{i} extraction error: {e}")
                    continue
            
            if len(records) < 4:
                logger.debug(f"{ticker}: Only {len(records)} valid quarters")
                return None
            
            df = pd.DataFrame(records)
            df['quarter_date'] = pd.to_datetime(df['quarter_date'])
            df = df.sort_values('quarter_date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.debug(f"{ticker} LSTM feature extraction failed: {e}")
            return None
    
    def collect_rf_features(self, ticker: str) -> Optional[Dict]:
        """
        Collect risk and sentiment features for RF Ensemble training.
        
        Returns:
            Dictionary with RF features or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'marketCap' not in info:
                return None
            
            # Get historical prices for volatility
            hist = stock.history(period="3mo")
            if hist.empty:
                return None
            
            # Calculate features
            returns = hist['Close'].pct_change().dropna()
            volatility_30d = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else 0.3
            
            # Volume z-score
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].tail(5).mean()
            volume_zscore = (recent_volume - avg_volume) / (hist['Volume'].std() + 1e-8)
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = rsi.iloc[-1] if not rsi.empty else 50
            
            features = {
                'ticker': ticker,
                'collection_date': datetime.now().isoformat(),
                'beta': info.get('beta', 1.0) or 1.0,
                'debt_to_equity': (info.get('debtToEquity', 0) or 0) / 100,
                '30d_volatility': float(volatility_30d),
                'volume_zscore': float(volume_zscore),
                'short_percent': (info.get('shortPercentOfFloat', 0) or 0) * 100,
                'rsi_14': float(rsi_14) if not pd.isna(rsi_14) else 50,
                'sentiment_mean': 0,  # Would come from news API
                'sentiment_std': 0.3,
                'news_volume': 0,
                'relevance_mean': 0.5,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'revenue_growth': (info.get('revenueGrowth', 0) or 0) * 100,
                'current_ratio': info.get('currentRatio', 1.0) or 1.0,
                'return_on_equity': (info.get('returnOnEquity', 0) or 0) * 100,
                # Additional context
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', hist['Close'].iloc[-1])
            }
            
            return features
            
        except Exception as e:
            logger.debug(f"{ticker} RF feature extraction failed: {e}")
            return None
    
    def _safe_get(self, df: pd.DataFrame, row_name: str, col, default=0):
        """Safely extract value from DataFrame"""
        try:
            if row_name in df.index and col in df.columns:
                val = df.loc[row_name, col]
                if pd.notna(val):
                    return float(val)
        except:
            pass
        return default
    
    def collect_batch(self, mode: str = 'full', batch_size: int = 50, 
                      resume: bool = True, max_workers: int = 4):
        """
        Collect training data in batch mode.
        
        Args:
            mode: 'full' (both), 'lstm' (quarterly only), 'rf' (risk only)
            batch_size: Number of stocks per batch
            resume: Continue from previous progress
            max_workers: Parallel workers
        """
        logger.info("=" * 80)
        logger.info(f"UNIFIED DATA COLLECTION - Mode: {mode.upper()}")
        logger.info("=" * 80)
        
        # Determine pending tickers
        if resume:
            if mode in ['full', 'lstm']:
                lstm_pending = [t for t in self.tickers 
                               if t not in self.progress['lstm_collected']]
            else:
                lstm_pending = []
            
            if mode in ['full', 'rf']:
                rf_pending = [t for t in self.tickers 
                             if t not in self.progress['rf_collected']]
            else:
                rf_pending = []
        else:
            lstm_pending = self.tickers if mode in ['full', 'lstm'] else []
            rf_pending = self.tickers if mode in ['full', 'rf'] else []
        
        pending = list(set(lstm_pending + rf_pending))[:batch_size]
        
        if not pending:
            print("\n" + "="*60)
            print("✅ DATA COLLECTION STATUS: COMPLETE")
            print("="*60)
            print(f"All {len(self.tickers)} tickers already collected.")
            print(f"LSTM: {len(self.progress['lstm_collected'])} tickers")
            print(f"RF: {len(self.progress['rf_collected'])} tickers")
            print("\nTo force recollection, use: --no-resume")
            print("To check data quality, use: --validate")
            print("="*60 + "\n")
            return
        
        logger.info(f"\nBatch size: {len(pending)} tickers")
        logger.info(f"LSTM pending: {len(lstm_pending)}, RF pending: {len(rf_pending)}")
        
        lstm_success = 0
        rf_success = 0
        failed = 0
        
        lstm_records = []
        rf_records = []
        
        for ticker in tqdm(pending, desc="Collecting data"):
            try:
                # LSTM features (quarterly financials)
                if mode in ['full', 'lstm'] and ticker not in self.progress['lstm_collected']:
                    lstm_df = self.collect_lstm_features(ticker)
                    if lstm_df is not None:
                        # Save individual file
                        output_file = self.raw_financials_dir / f"{ticker}_quarterly.csv"
                        lstm_df.to_csv(output_file, index=False)
                        lstm_records.extend(lstm_df.to_dict('records'))
                        self.progress['lstm_collected'].append(ticker)
                        lstm_success += 1
                    else:
                        self.progress['failed'][ticker] = self.progress['failed'].get(ticker, {})
                        self.progress['failed'][ticker]['lstm'] = 'Insufficient quarterly data'
                
                # RF features (risk/sentiment)
                if mode in ['full', 'rf'] and ticker not in self.progress['rf_collected']:
                    rf_data = self.collect_rf_features(ticker)
                    if rf_data is not None:
                        # Save individual file
                        output_file = self.raw_risk_dir / f"{ticker}_risk.json"
                        with open(output_file, 'w') as f:
                            json.dump(rf_data, f, indent=2, default=str)
                        rf_records.append(rf_data)
                        self.progress['rf_collected'].append(ticker)
                        rf_success += 1
                    else:
                        self.progress['failed'][ticker] = self.progress['failed'].get(ticker, {})
                        self.progress['failed'][ticker]['rf'] = 'Feature extraction failed'
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"{ticker} failed: {e}")
                self.progress['failed'][ticker] = str(e)
                failed += 1
        
        # Save consolidated training data
        if lstm_records:
            lstm_df = pd.DataFrame(lstm_records)
            output_path = self.lstm_data_dir / "quarterly_features_combined.csv"
            
            # Append to existing or create new
            if output_path.exists():
                existing = pd.read_csv(output_path)
                lstm_df = pd.concat([existing, lstm_df]).drop_duplicates(
                    subset=['ticker', 'quarter_date'], keep='last'
                )
            
            lstm_df.to_csv(output_path, index=False)
            logger.info(f"LSTM training data saved: {len(lstm_df)} records")
        
        if rf_records:
            rf_df = pd.DataFrame(rf_records)
            output_path = self.rf_data_dir / "risk_features_combined.csv"
            
            if output_path.exists():
                existing = pd.read_csv(output_path)
                rf_df = pd.concat([existing, rf_df]).drop_duplicates(
                    subset=['ticker', 'collection_date'], keep='last'
                )
            
            rf_df.to_csv(output_path, index=False)
            logger.info(f"RF training data saved: {len(rf_df)} records")
        
        # Update quality stats
        self.progress['quality_stats'] = {
            'lstm_tickers': len(self.progress['lstm_collected']),
            'rf_tickers': len(self.progress['rf_collected']),
            'total_tickers': len(self.tickers),
            'lstm_coverage': f"{len(self.progress['lstm_collected'])}/{len(self.tickers)}",
            'rf_coverage': f"{len(self.progress['rf_collected'])}/{len(self.tickers)}"
        }
        
        self._save_progress()
        
        # Summary
        # Print clear CLI summary
        print("\n" + "="*60)
        print("✅ BATCH COLLECTION COMPLETE")
        print("="*60)
        print(f"Mode: {mode.upper()}")
        print(f"\nResults:")
        print(f"  LSTM (Quarterly Financials): +{lstm_success} new → {len(self.progress['lstm_collected'])}/{len(self.tickers)} total")
        print(f"  RF (Risk/Sentiment): +{rf_success} new → {len(self.progress['rf_collected'])}/{len(self.tickers)} total")
        print(f"  Failed: {failed}")
        if lstm_records:
            print(f"\nLSTM data saved: {len(lstm_records)} records")
        if rf_records:
            print(f"RF data saved: {len(rf_records)} records")
        print("="*60 + "\n")
        
        logger.info(f"Batch complete: LSTM +{lstm_success}, RF +{rf_success}, Failed {failed}")
    
    def continuous_collection(self, interval_hours: int = 24):
        """
        Run continuous collection - designed for scheduled execution.
        Collects new data and updates existing records.
        """
        logger.info("=" * 80)
        logger.info("CONTINUOUS COLLECTION MODE")
        logger.info("=" * 80)
        
        # Check last run
        last_run = self.progress.get('last_full_run')
        if last_run:
            last_run_dt = datetime.fromisoformat(last_run)
            hours_since = (datetime.now() - last_run_dt).total_seconds() / 3600
            
            if hours_since < interval_hours:
                print("\n" + "="*60)
                print("⏰ CONTINUOUS COLLECTION: SKIPPED (INTERVAL NOT MET)")
                print("="*60)
                print(f"Last full run: {hours_since:.1f} hours ago")
                print(f"Interval: {interval_hours} hours")
                print(f"Next run in: {interval_hours - hours_since:.1f} hours")
                print("\nTo force run now, use: --mode full --no-resume")
                print("="*60 + "\n")
                return
            else:
                print(f"\nLast run was {hours_since:.1f} hours ago. Running collection...")
        
        # Run full collection
        self.collect_batch(mode='full', batch_size=len(self.tickers), resume=True)
        
        self.progress['last_full_run'] = datetime.now().isoformat()
        self._save_progress()
        
        print("\n" + "="*60)
        print("✅ CONTINUOUS COLLECTION: COMPLETE")
        print("="*60)
        print(f"Next scheduled run: {interval_hours} hours from now")
        print(f"Use --status to check progress")
        print("="*60 + "\n")
    
    def validate_data(self) -> Dict:
        """
        Validate collected data quality.
        
        Returns:
            Quality report dictionary
        """
        logger.info("=" * 80)
        logger.info("DATA QUALITY VALIDATION")
        logger.info("=" * 80)
        
        report = {
            'lstm': {'valid': 0, 'invalid': 0, 'issues': []},
            'rf': {'valid': 0, 'invalid': 0, 'issues': []}
        }
        
        # Validate LSTM data
        lstm_combined = self.lstm_data_dir / "quarterly_features_combined.csv"
        if lstm_combined.exists():
            df = pd.read_csv(lstm_combined)
            
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker]
                
                if len(ticker_df) < 4:
                    report['lstm']['issues'].append(f"{ticker}: Only {len(ticker_df)} quarters")
                    report['lstm']['invalid'] += 1
                elif ticker_df[self.LSTM_FEATURES].isna().sum().sum() > len(ticker_df):  # Allow some NaNs
                    report['lstm']['issues'].append(f"{ticker}: Too many NaN values")
                    report['lstm']['invalid'] += 1
                else:
                    report['lstm']['valid'] += 1
            
            logger.info(f"\nLSTM Data:")
            logger.info(f"  Total tickers: {df['ticker'].nunique()}")
            logger.info(f"  Valid: {report['lstm']['valid']}")
            logger.info(f"  Invalid: {report['lstm']['invalid']}")
        else:
            logger.warning("No LSTM combined data found")
        
        # Validate RF data
        rf_combined = self.rf_data_dir / "risk_features_combined.csv"
        if rf_combined.exists():
            df = pd.read_csv(rf_combined)
            
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker]
                
                # Check for required features
                missing = [f for f in self.RF_FEATURES if f not in ticker_df.columns]
                if missing:
                    report['rf']['issues'].append(f"{ticker}: Missing {missing}")
                    report['rf']['invalid'] += 1
                else:
                    report['rf']['valid'] += 1
            
            logger.info(f"\nRF Data:")
            logger.info(f"  Total tickers: {df['ticker'].nunique()}")
            logger.info(f"  Valid: {report['rf']['valid']}")
            logger.info(f"  Invalid: {report['rf']['invalid']}")
        else:
            logger.warning("No RF combined data found")
        
        return report
    
    def show_status(self):
        """Display collection progress status"""
        logger.info("=" * 80)
        logger.info("COLLECTION STATUS")
        logger.info("=" * 80)
        
        logger.info(f"\nUniverse: {len(self.tickers)} tickers")
        logger.info(f"Started: {self.progress.get('started', 'N/A')}")
        logger.info(f"Last updated: {self.progress.get('last_updated', 'N/A')}")
        
        lstm_count = len(self.progress.get('lstm_collected', []))
        rf_count = len(self.progress.get('rf_collected', []))
        
        logger.info(f"\nLSTM (Quarterly Financials):")
        logger.info(f"  Collected: {lstm_count}/{len(self.tickers)} ({lstm_count/len(self.tickers)*100:.1f}%)")
        
        logger.info(f"\nRF (Risk/Sentiment):")
        logger.info(f"  Collected: {rf_count}/{len(self.tickers)} ({rf_count/len(self.tickers)*100:.1f}%)")
        
        failed = self.progress.get('failed', {})
        if failed:
            logger.info(f"\nFailed tickers: {len(failed)}")
            for ticker, error in list(failed.items())[:5]:
                logger.info(f"  {ticker}: {error}")
            if len(failed) > 5:
                logger.info(f"  ... and {len(failed) - 5} more")
        
        # Check data files
        lstm_file = self.lstm_data_dir / "quarterly_features_combined.csv"
        rf_file = self.rf_data_dir / "risk_features_combined.csv"
        
        # Print clear CLI summary
        print("\n" + "="*60)
        print("📊 DATA COLLECTION STATUS")
        print("="*60)
        print(f"Universe: {len(self.tickers)} tickers")
        print(f"Started: {self.progress.get('started', 'N/A')}")
        print(f"Last updated: {self.progress.get('last_updated', 'N/A')}")
        print(f"\nProgress:")
        print(f"  LSTM (Quarterly Financials): {lstm_count}/{len(self.tickers)} ({lstm_count/len(self.tickers)*100:.1f}%)")
        print(f"  RF (Risk/Sentiment): {rf_count}/{len(self.tickers)} ({rf_count/len(self.tickers)*100:.1f}%)")
        print(f"\nData Files:")
        print(f"  LSTM: {'✅ Exists' if lstm_file.exists() else '❌ Missing'} ({lstm_file.name})")
        print(f"  RF: {'✅ Exists' if rf_file.exists() else '❌ Missing'} ({rf_file.name})")
        
        if failed:
            print(f"\nFailed tickers: {len(failed)}")
        
        print("\nNext steps:")
        if lstm_count < len(self.tickers) or rf_count < len(self.tickers):
            print("  Run: --mode full --batch-size 50   (collect more data)")
        print("  Run: --validate                      (check data quality)")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Collection for LSTM-DCF and RF Ensemble Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full collection for both models
    python collect_unified_training_data.py --mode full --batch-size 100

    # LSTM only (quarterly financials)
    python collect_unified_training_data.py --mode lstm --batch-size 50

    # RF only (risk/sentiment features)
    python collect_unified_training_data.py --mode rf --batch-size 100

    # Continuous/scheduled collection
    python collect_unified_training_data.py --mode continuous

    # Check progress
    python collect_unified_training_data.py --status

    # Validate data quality
    python collect_unified_training_data.py --validate
        """
    )
    
    parser.add_argument('--mode', choices=['full', 'lstm', 'rf', 'continuous'],
                        default='full', help='Collection mode')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of stocks per batch')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh (ignore previous progress)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing data quality')
    parser.add_argument('--status', action='store_true',
                        help='Show collection progress')
    
    args = parser.parse_args()
    
    collector = UnifiedDataCollector()
    
    if args.status:
        collector.show_status()
    elif args.validate:
        report = collector.validate_data()
        print(json.dumps(report, indent=2))
    elif args.mode == 'continuous':
        collector.continuous_collection()
    else:
        collector.collect_batch(
            mode=args.mode,
            batch_size=args.batch_size,
            resume=not args.no_resume
        )


if __name__ == "__main__":
    main()
