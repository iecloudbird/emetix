"""
Comprehensive Stock Data Fetcher for LSTM Training & Backtesting
Fetches S&P 500 + NYSE stocks for:
1. LSTM-DCF Growth Forecasting Training
2. Backtesting Historical Performance

Features:
- Full S&P 500 coverage (~500 stocks)
- Major NYSE stocks (~100 additional)
- Time-series data for LSTM (60-day sequences with 16 features)
- Historical prices for backtesting
- Quarterly fundamentals for growth rate validation
- Lean storage (only essential features)
- Data quality validation

Usage:
    # Fetch time-series for LSTM training
    python scripts/data_collection/fetch_comprehensive_training_data.py --mode timeseries --batch-size 100
    
    # Fetch fundamentals for validation
    python scripts/data_collection/fetch_comprehensive_training_data.py --mode fundamentals --batch-size 50
    
    # Create backtest dataset
    python scripts/data_collection/fetch_comprehensive_training_data.py --mode backtest --lookback-years 5
    
    # Validate data quality
    python scripts/data_collection/fetch_comprehensive_training_data.py --validate
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
from typing import List, Dict, Optional
from tqdm import tqdm

from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.fetchers.unified_financials_fetcher import UnifiedFinancialsFetcher
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


# Full S&P 500 ticker list (500 stocks)
SP500_TICKERS = [
    # Technology (80 stocks)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ADBE',
    'CSCO', 'ACN', 'CRM', 'TXN', 'ORCL', 'QCOM', 'INTC', 'AMD', 'IBM', 'INTU',
    'NOW', 'AMAT', 'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
    'NXPI', 'ADSK', 'MRVL', 'ON', 'ANSS', 'MPWR', 'PANW', 'CRWD', 'DDOG', 'WDAY',
    'ZS', 'OKTA', 'NET', 'SNOW', 'MDB', 'TEAM', 'HUBS', 'ZM', 'DOCU', 'TWLO',
    'PLTR', 'U', 'PATH', 'RBLX', 'GTLB', 'S', 'CFLT', 'DT', 'IOT', 'ESTC',
    'DASH', 'ABNB', 'UBER', 'LYFT', 'PINS', 'SNAP', 'SPOT', 'ROKU', 'NFLX', 'DIS',
    'PYPL', 'SQ', 'COIN', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA', 'CARG',
    
    # Healthcare (70 stocks)
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
    'AMGN', 'GILD', 'VRTX', 'REGN', 'CVS', 'CI', 'HUM', 'ELV', 'CNC', 'MOH',
    'ISRG', 'SYK', 'BDX', 'BSX', 'EW', 'ZBH', 'BAX', 'DXCM', 'ILMN', 'IDXX',
    'A', 'RMD', 'ZTS', 'HOLX', 'PODD', 'ALGN', 'INCY', 'EXAS', 'TECH', 'PEN',
    'BIIB', 'MRNA', 'BNTX', 'NVAX', 'SGEN', 'BMRN', 'ALNY', 'IONS', 'RARE', 'FOLD',
    'ARWR', 'BLUE', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'SGMO', 'FATE', 'CLLS',
    'MDT', 'GEHC', 'WAT', 'MTD', 'IQV', 'CRL', 'LH', 'DGX', 'TECH', 'RVTY',
    
    # Financials (70 stocks)
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'KEY', 'RF', 'CFG', 'HBAN',
    'CMA', 'ZION', 'FRC', 'SIVB', 'WTFC', 'EWBC', 'WAL', 'GBCI', 'BANR', 'COLB',
    'SPGI', 'CME', 'ICE', 'MSCI', 'MCO', 'NDAQ', 'CBOE', 'MKTX', 'VIRT', 'IBKR',
    'MMC', 'AJG', 'WTW', 'BRO', 'AON', 'CB', 'TRV', 'PGR', 'ALL', 'AIG',
    'MET', 'PRU', 'AFL', 'AMP', 'CINF', 'AIZ', 'WRB', 'RNR', 'RE', 'GL',
    'BRK.B', 'BRK.A', 'AXS', 'APAM', 'TROW', 'BEN', 'IVZ', 'AMG', 'SEIC', 'HLI',
    
    # Consumer Discretionary (60 stocks)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'MAR',
    'GM', 'F', 'CMG', 'YUM', 'DRI', 'ORLY', 'AZO', 'GPC', 'AAP', 'BBWI',
    'RL', 'PVH', 'HBI', 'UA', 'UAA', 'CROCS', 'DECK', 'BIRK', 'ONON', 'HOKA',
    'TGT', 'COST', 'WMT', 'DG', 'DLTR', 'BIG', 'FIVE', 'OLLI', 'PSMT', 'RCII',
    'ROST', 'TJX', 'BURL', 'URBN', 'ANF', 'AEO', 'GPS', 'EXPR', 'ZUMZ', 'GES',
    'DHI', 'LEN', 'PHM', 'TOL', 'KBH', 'BZH', 'MTH', 'TMHC', 'MHO', 'LGIH',
    
    # Consumer Staples (35 stocks)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
    'GIS', 'K', 'CPB', 'CAG', 'SJM', 'HSY', 'MKSI', 'LW', 'POST', 'BGFV',
    'KR', 'SYY', 'MNST', 'KDP', 'TAP', 'STZ', 'BF.B', 'SAM', 'CELH', 'FIZZ',
    'CHD', 'CLX', 'SPB', 'COTY', 'ELF',
    
    # Energy (30 stocks)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL',
    'BKR', 'FANG', 'DVN', 'HES', 'MRO', 'APA', 'CTRA', 'OVV', 'PR', 'MGY',
    'PXD', 'EQT', 'AR', 'MTDR', 'SM', 'RRC', 'CHRD', 'VTLE', 'NOG', 'CRGY',
    
    # Industrials (70 stocks)
    'BA', 'HON', 'UNP', 'RTX', 'LMT', 'CAT', 'DE', 'GE', 'MMM', 'ITW',
    'EMR', 'ETN', 'NOC', 'GD', 'LHX', 'TDG', 'CARR', 'OTIS', 'PWR', 'FLR',
    'UPS', 'FDX', 'NSC', 'CSX', 'JBHT', 'ODFL', 'KNX', 'XPO', 'CHRW', 'LSTR',
    'WM', 'RSG', 'CWST', 'GFL', 'WCN', 'CLH', 'MEG', 'RVTY', 'CECO', 'NVRI',
    'PH', 'ROK', 'DOV', 'AME', 'GNRC', 'HUBB', 'AOS', 'UFPI', 'BLDR', 'AZEK',
    'IR', 'VLTO', 'CMI', 'PCAR', 'FAST', 'WWD', 'AGCO', 'ALSN', 'LEA', 'VC',
    'J', 'JELD', 'AWI', 'BECN', 'CSL', 'HI', 'AIRC', 'AIR', 'B', 'GVA',
    
    # Materials (30 stocks)
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
    'STLD', 'RS', 'IP', 'PKG', 'AMCR', 'AVY', 'SEE', 'SON', 'WLK', 'OLN',
    'ALB', 'FMC', 'CE', 'CF', 'MOS', 'IFF', 'EMN', 'DOW', 'LYB', 'WLK',
    
    # Real Estate (30 stocks)
    'PLD', 'AMT', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'AVB', 'EQR',
    'VICI', 'INVH', 'EXR', 'VTR', 'MAA', 'ESS', 'SPG', 'SUI', 'CPT', 'UDR',
    'KIM', 'REG', 'BXP', 'VNO', 'SLG', 'ARE', 'DRE', 'EPRT', 'NNN', 'ADC',
    
    # Utilities (30 stocks)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED',
    'WEC', 'ES', 'ETR', 'DTE', 'PPL', 'AEE', 'CMS', 'CNP', 'AWK', 'ATO',
    'FE', 'EVRG', 'LNT', 'NI', 'PNW', 'OGE', 'SJW', 'MSEX', 'AVA', 'NWE',
    
    # Communication Services (25 stocks)
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS',
    'LBRDA', 'LBRDK', 'DISH', 'SIRI', 'LUMN', 'CABO', 'ATUS', 'LILAK', 'WOW', 'FYBR',
    'TTWO', 'EA', 'ATVI', 'ZNGA', 'RBLX'
]

# Major NYSE stocks not in S&P 500 (100 additional)
NYSE_ADDITIONAL = [
    # Emerging Tech
    'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'LPRO', 'TOST', 'BILL', 'DKNG',
    
    # International
    'BABA', 'PDD', 'JD', 'BIDU', 'NIO', 'XPEV', 'LI', 'TME', 'BILI', 'IQ',
    
    # Energy/Renewables
    'PLUG', 'FCEL', 'BE', 'SEDG', 'ENPH', 'RUN', 'NOVA', 'SPWR', 'CSIQ', 'JKS',
    
    # Biotech
    'SANA', 'FATE', 'CLLS', 'BCYC', 'RLAY', 'ARCT', 'VERV', 'PRTA', 'SDGR', 'CGEM',
    
    # Industrial/Transport
    'RIVN', 'LCID', 'FSR', 'RIDE', 'GOEV', 'WKHS', 'BLNK', 'EVGO', 'CHPT', 'QS',
    
    # Consumer
    'BYND', 'TTCF', 'VERY', 'KIND', 'SMPL', 'HAIN', 'FDP', 'BGS', 'FARM', 'APPH',
    
    # Finance/Fintech
    'SFM', 'VOYA', 'RGA', 'LNC', 'FNF', 'FAF', 'EWBC', 'PACW', 'WAL', 'BANR',
    
    # Healthcare Services
    'CNC', 'MOH', 'HQY', 'EHC', 'THC', 'UHS', 'AMED', 'ENSG', 'PNTG', 'AGL',
    
    # REIT
    'AMH', 'CUBE', 'LSI', 'REXR', 'FR', 'IRM', 'COLD', 'STAG', 'TRNO', 'NSA',
    
    # Other
    'ZIM', 'MATX', 'SBLK', 'DAC', 'GSL', 'EGLE', 'SB', 'CMRE', 'NMM', 'SHIP'
]


class ComprehensiveDataFetcher:
    """Fetch comprehensive data for LSTM training and backtesting"""
    
    def __init__(self):
        self.yfinance_fetcher = YFinanceFetcher()
        self.unified_fetcher = UnifiedFinancialsFetcher()
        
        # Directories
        self.timeseries_dir = RAW_DATA_DIR / "timeseries"
        self.fundamentals_dir = RAW_DATA_DIR / "fundamentals"
        self.backtest_dir = PROCESSED_DATA_DIR / "backtesting"
        
        for dir in [self.timeseries_dir, self.fundamentals_dir, self.backtest_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "comprehensive_fetch_progress.json"
        self.progress = self._load_progress()
        
        # Ticker universe
        self.all_tickers = SP500_TICKERS + NYSE_ADDITIONAL
        logger.info(f"Comprehensive Fetcher initialized with {len(self.all_tickers)} tickers")
        logger.info(f"  S&P 500: {len(SP500_TICKERS)}")
        logger.info(f"  NYSE Additional: {len(NYSE_ADDITIONAL)}")
    
    def _load_progress(self) -> Dict:
        """Load fetch progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'started': datetime.now().isoformat(),
            'timeseries_fetched': [],
            'fundamentals_fetched': [],
            'backtest_created': False,
            'failed_tickers': {},
            'data_quality': {}
        }
    
    def _save_progress(self):
        """Save progress"""
        self.progress['last_updated'] = datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def fetch_timeseries_batch(self, batch_size: int = 100, resume: bool = True):
        """
        Fetch time-series data for LSTM training
        
        LSTM requires 16 features × 60-day sequences:
        - Price data: close, volume, returns, sma_20, sma_50, volatility_30, rsi_14
        - Fundamentals: pe_ratio, beta, debt_equity, eps
        - Derived: fcff_proxy (approximation for LSTM)
        
        Args:
            batch_size: Number of stocks to fetch
            resume: Continue from previous progress
        """
        logger.info("="*80)
        logger.info("TIMESERIES FETCH - LSTM TRAINING DATA")
        logger.info("="*80)
        
        # Get pending tickers
        if resume:
            fetched = set(self.progress['timeseries_fetched'])
            pending = [t for t in self.all_tickers if t not in fetched]
        else:
            pending = self.all_tickers.copy()
        
        if not pending:
            logger.info("✅ All timeseries data already fetched!")
            return
        
        batch = pending[:batch_size]
        
        logger.info(f"\nBatch: {len(batch)} tickers")
        logger.info(f"Progress: {len(self.progress['timeseries_fetched'])}/{len(self.all_tickers)} ({len(self.progress['timeseries_fetched'])/len(self.all_tickers)*100:.1f}%)")
        logger.info(f"Remaining: {len(pending)}")
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for ticker in tqdm(batch, desc="Fetching timeseries"):
            # Check if timeseries file already exists
            output_file = self.timeseries_dir / f"{ticker}_timeseries.csv"
            if output_file.exists():
                logger.debug(f"Skipping {ticker}: timeseries file already exists")
                if ticker not in self.progress['timeseries_fetched']:
                    self.progress['timeseries_fetched'].append(ticker)
                skipped_count += 1
                continue
            
            try:
                # Fetch stock data (includes 60+ days history with all features)
                stock_data = self.yfinance_fetcher.fetch_stock_data(ticker)
                
                if stock_data is None or stock_data.empty:
                    logger.warning(f"No data for {ticker}")
                    self.progress['failed_tickers'][ticker] = "No stock data"
                    failed_count += 1
                    continue
                
                # Fetch historical prices (for sequences)
                # Using 6mo period to get 180 days of data (sufficient for 60-day sequences)
                historical = self.yfinance_fetcher.fetch_historical_prices(
                    ticker, 
                    period='6mo'  # 6 months = ~180 days
                )
                
                if historical is None or historical.empty or len(historical) < 60:
                    logger.warning(f"Insufficient history for {ticker}: {len(historical) if historical is not None else 0} days")
                    self.progress['failed_tickers'][ticker] = f"Insufficient history: {len(historical) if historical is not None else 0} days"
                    failed_count += 1
                    continue
                
                # Calculate 16 features (match LSTM training data format)
                df = historical.copy()
                df['ticker'] = ticker
                
                # Technical indicators
                df['returns'] = df['Close'].pct_change()
                df['sma_20'] = df['Close'].rolling(20).mean()
                df['sma_50'] = df['Close'].rolling(50).mean()
                df['volatility_30'] = df['returns'].rolling(30).std()
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Fundamentals (from stock_data dict)
                df['pe_ratio'] = stock_data.get('pe_ratio', 0)
                df['beta'] = stock_data.get('beta', 1.0)
                df['debt_equity'] = stock_data.get('debt_to_equity', 0)
                df['eps'] = stock_data.get('eps', 0)
                
                # FCF proxy (approximation: use price-based proxy for time-series)
                df['fcff_proxy'] = df['Close'] * df['Volume'] * 0.001  # Simple approximation
                
                # Rename for consistency with training data
                df = df.rename(columns={
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Select 16 features (match LSTM input)
                features = [
                    'close', 'volume', 'returns', 'sma_20', 'sma_50', 
                    'volatility_30', 'rsi_14', 'pe_ratio', 'beta', 
                    'debt_equity', 'eps', 'fcff_proxy', 'ticker'
                ]
                
                df = df[features].dropna()
                
                if len(df) < 60:
                    logger.warning(f"After feature calc, {ticker} has only {len(df)} rows (need 60+)")
                    self.progress['failed_tickers'][ticker] = f"Insufficient rows after features: {len(df)}"
                    failed_count += 1
                    continue
                
                # Save to CSV (lean format)
                output_file = self.timeseries_dir / f"{ticker}_timeseries.csv"
                df.to_csv(output_file, index_label='date')
                
                # Update progress
                self.progress['timeseries_fetched'].append(ticker)
                self.progress['data_quality'][ticker] = {
                    'timeseries_rows': len(df),
                    'date_range': f"{df.index.min()} to {df.index.max()}",
                    'features': len(features) - 1  # Exclude ticker column
                }
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                self.progress['failed_tickers'][ticker] = str(e)
                failed_count += 1
        
        self._save_progress()
        
        logger.info("\n" + "="*80)
        logger.info("✅ TIMESERIES BATCH COMPLETE")
        logger.info("="*80)
        logger.info(f"  Successful: {success_count}/{len(batch)}")
        logger.info(f"  Failed: {failed_count}/{len(batch)}")
        logger.info(f"  Overall: {len(self.progress['timeseries_fetched'])}/{len(self.all_tickers)} ({len(self.progress['timeseries_fetched'])/len(self.all_tickers)*100:.1f}%)")
    
    def fetch_fundamentals_batch(self, batch_size: int = 50, resume: bool = True):
        """
        Fetch quarterly fundamentals for growth rate validation
        
        Uses UnifiedFinancialsFetcher (Alpha Vantage + Finnhub)
        """
        logger.info("="*80)
        logger.info("FUNDAMENTALS FETCH - GROWTH RATE VALIDATION")
        logger.info("="*80)
        
        # Get pending tickers
        if resume:
            fetched = set(self.progress['fundamentals_fetched'])
            pending = [t for t in self.all_tickers if t not in fetched]
        else:
            pending = self.all_tickers.copy()
        
        if not pending:
            logger.info("✅ All fundamentals data already fetched!")
            return
        
        batch = pending[:batch_size]
        
        logger.info(f"\nBatch: {len(batch)} tickers")
        logger.info(f"Progress: {len(self.progress['fundamentals_fetched'])}/{len(self.all_tickers)}")
        
        # Use unified fetcher (smart source selection)
        results = self.unified_fetcher.fetch_batch_smart(
            batch,
            min_quarters=20,
            use_cache=True,
            alpha_vantage_limit=25
        )
        
        # Update progress
        for ticker, data in results.items():
            self.progress['fundamentals_fetched'].append(ticker)
            
            if ticker in self.progress['data_quality']:
                self.progress['data_quality'][ticker]['fundamentals_quarters'] = data['quarters']
                self.progress['data_quality'][ticker]['data_source'] = data.get('source', 'Unknown')
            else:
                self.progress['data_quality'][ticker] = {
                    'fundamentals_quarters': data['quarters'],
                    'data_source': data.get('source', 'Unknown')
                }
        
        # Track failures
        failed = set(batch) - set(results.keys())
        for ticker in failed:
            self.progress['failed_tickers'][ticker] = "Fundamentals fetch failed"
        
        self._save_progress()
        
        logger.info(f"\n✅ Fetched: {len(results)}/{len(batch)}")
        logger.info(f"  Overall: {len(self.progress['fundamentals_fetched'])}/{len(self.all_tickers)}")
    
    def create_backtest_dataset(self, lookback_years: int = 5):
        """
        Create backtesting dataset with historical prices + actual growth rates
        
        For each stock:
        - Historical daily prices (5 years)
        - Quarterly growth rates (actual)
        - LSTM predictions at each quarter (for comparison)
        
        This allows validation: "Did LSTM correctly predict growth?"
        """
        logger.info("="*80)
        logger.info("BACKTEST DATASET CREATION")
        logger.info("="*80)
        
        logger.info(f"\nLookback: {lookback_years} years")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years*365)
        
        backtest_records = []
        
        # Only use stocks with both timeseries + fundamentals
        valid_tickers = set(self.progress['timeseries_fetched']) & set(self.progress['fundamentals_fetched'])
        
        logger.info(f"Valid tickers (have both timeseries + fundamentals): {len(valid_tickers)}")
        
        for ticker in tqdm(list(valid_tickers), desc="Creating backtest dataset"):
            try:
                # Fetch historical prices for backtesting
                # Use max period to get full history (will be filtered by lookback_years later)
                period_map = {5: '5y', 10: '10y', 15: 'max', 20: 'max'}
                period = period_map.get(lookback_years, '5y' if lookback_years <= 5 else 'max')
                
                historical = self.yfinance_fetcher.fetch_historical_prices(
                    ticker,
                    period=period
                )
                
                if historical is None or historical.empty:
                    logger.warning(f"No historical data for {ticker}")
                    continue
                
                # Load fundamentals from financial_statements cache
                financial_statements_dir = RAW_DATA_DIR / "financial_statements"
                income_file = financial_statements_dir / f"{ticker}_income.csv"
                cashflow_file = financial_statements_dir / f"{ticker}_cashflow.csv"
                
                if not income_file.exists() or not cashflow_file.exists():
                    logger.debug(f"Missing financial statements for {ticker}")
                    continue
                
                # Load financial statements
                income_df = pd.read_csv(income_file)
                cashflow_df = pd.read_csv(cashflow_file)
                
                # Merge income and cashflow on date
                if 'fiscalDateEnding' not in income_df.columns:
                    continue
                
                raw_df = income_df.merge(
                    cashflow_df[['fiscalDateEnding', 'operatingCashflow', 'capitalExpenditures']], 
                    on='fiscalDateEnding',
                    how='inner'
                )
                
                # Extract quarterly growth rates (actual)
                if 'totalRevenue' in raw_df.columns:
                    raw_df['date'] = pd.to_datetime(raw_df['fiscalDateEnding'])
                    raw_df['revenue'] = pd.to_numeric(raw_df['totalRevenue'], errors='coerce')
                    
                # Calculate revenue growth (year-over-year)
                raw_df = raw_df.sort_values('date')
                raw_df['revenue_growth'] = raw_df['revenue'].pct_change()
                
                # Calculate FCF growth
                if 'operatingCashflow' in raw_df.columns and 'capitalExpenditures' in raw_df.columns:
                    raw_df['fcf'] = pd.to_numeric(raw_df['operatingCashflow'], errors='coerce') - abs(pd.to_numeric(raw_df['capitalExpenditures'], errors='coerce'))
                    raw_df['fcf_growth'] = raw_df['fcf'].pct_change()
                
                # Merge with historical prices
                for idx, row in raw_df.iterrows():
                    quarter_date = row['date']
                    
                    # Make quarter_date timezone-aware to match historical index
                    if historical.index.tz is not None:
                        quarter_date = quarter_date.tz_localize('UTC').tz_convert(historical.index.tz)
                    
                    # Find closest price
                    closest_price = historical[historical.index <= quarter_date].tail(1)
                    
                    if closest_price.empty:
                        continue
                    
                    # Convert quarter_date back to naive for output
                    output_date = row['date'] if not hasattr(row['date'], 'tz_localize') else row['date']
                    
                    record = {
                        'ticker': ticker,
                        'date': output_date.strftime('%Y-%m-%d'),
                        'price': float(closest_price['Close'].values[0]),
                        'revenue': float(row.get('revenue', 0)) if pd.notna(row.get('revenue', 0)) else 0,
                        'revenue_growth': float(row.get('revenue_growth', 0)) if pd.notna(row.get('revenue_growth', 0)) else 0,
                        'fcf_growth': float(row.get('fcf_growth', 0)) if 'fcf_growth' in raw_df.columns and pd.notna(row.get('fcf_growth', 0)) else 0,
                        'pe_ratio': 0,  # Will be filled from timeseries
                        'beta': 0,
                        'debt_equity': 0
                    }
                    
                    backtest_records.append(record)
            
            except Exception as e:
                logger.error(f"Error creating backtest for {ticker}: {e}")
                continue
        
        if not backtest_records:
            logger.error("No backtest records created!")
            return
        
        # Save backtest dataset
        backtest_df = pd.DataFrame(backtest_records)
        output_file = self.backtest_dir / "lstm_dcf_backtest_data.csv"
        backtest_df.to_csv(output_file, index=False)
        
        self.progress['backtest_created'] = True
        self._save_progress()
        
        logger.info("\n" + "="*80)
        logger.info("✅ BACKTEST DATASET CREATED")
        logger.info("="*80)
        logger.info(f"  File: {output_file}")
        logger.info(f"  Records: {len(backtest_df):,}")
        logger.info(f"  Stocks: {backtest_df['ticker'].nunique()}")
        logger.info(f"  Date range: {backtest_df['date'].min()} to {backtest_df['date'].max()}")
    
    def validate_data_quality(self):
        """Validate data quality and print summary"""
        logger.info("="*80)
        logger.info("DATA QUALITY VALIDATION")
        logger.info("="*80)
        
        # Check timeseries
        timeseries_complete = len(self.progress['timeseries_fetched'])
        timeseries_total = len(self.all_tickers)
        
        logger.info(f"\n📊 Timeseries Data:")
        logger.info(f"  Complete: {timeseries_complete}/{timeseries_total} ({timeseries_complete/timeseries_total*100:.1f}%)")
        
        # Check fundamentals
        fundamentals_complete = len(self.progress['fundamentals_fetched'])
        
        logger.info(f"\n📊 Fundamentals Data:")
        logger.info(f"  Complete: {fundamentals_complete}/{timeseries_total} ({fundamentals_complete/timeseries_total*100:.1f}%)")
        
        # Check data quality metrics
        logger.info(f"\n📊 Data Quality:")
        
        sufficient_timeseries = 0
        sufficient_fundamentals = 0
        
        for ticker, quality in self.progress['data_quality'].items():
            if quality.get('timeseries_rows', 0) >= 60:
                sufficient_timeseries += 1
            
            if quality.get('fundamentals_quarters', 0) >= 20:
                sufficient_fundamentals += 1
        
        if timeseries_complete > 0:
            logger.info(f"  Timeseries (>=60 rows): {sufficient_timeseries}/{timeseries_complete} ({sufficient_timeseries/timeseries_complete*100:.1f}%)")
        else:
            logger.info(f"  Timeseries (>=60 rows): 0/0 (No data yet)")
        
        if fundamentals_complete > 0:
            logger.info(f"  Fundamentals (>=20 quarters): {sufficient_fundamentals}/{fundamentals_complete} ({sufficient_fundamentals/fundamentals_complete*100:.1f}%)")
        else:
            logger.info(f"  Fundamentals (>=20 quarters): 0/0 (No data yet)")
        
        # Check failures
        failed_count = len(self.progress['failed_tickers'])
        
        if failed_count > 0:
            logger.info(f"\n⚠️  Failed Tickers: {failed_count}")
            logger.info("  Top 10 failures:")
            
            for i, (ticker, reason) in enumerate(list(self.progress['failed_tickers'].items())[:10]):
                logger.info(f"    {i+1}. {ticker}: {reason}")
        
        # Backtest status
        logger.info(f"\n📊 Backtesting:")
        logger.info(f"  Dataset created: {'✅ Yes' if self.progress.get('backtest_created', False) else '❌ No'}")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive data fetcher for LSTM training & backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['timeseries', 'fundamentals', 'backtest', 'validate'],
        default='timeseries',
        help='Fetch mode: timeseries (LSTM), fundamentals (validation), backtest (historical), or validate (check quality)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for fetching (default: 100 for timeseries, 50 for fundamentals)'
    )
    
    parser.add_argument(
        '--lookback-years',
        type=int,
        default=5,
        help='Years of historical data for backtesting (default: 5)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh (ignore previous progress)'
    )
    
    args = parser.parse_args()
    
    fetcher = ComprehensiveDataFetcher()
    
    if args.mode == 'timeseries':
        fetcher.fetch_timeseries_batch(
            batch_size=args.batch_size,
            resume=not args.no_resume
        )
    
    elif args.mode == 'fundamentals':
        batch_size = min(args.batch_size, 50)  # Cap at 50 for fundamentals (API limits)
        fetcher.fetch_fundamentals_batch(
            batch_size=batch_size,
            resume=not args.no_resume
        )
    
    elif args.mode == 'backtest':
        fetcher.create_backtest_dataset(lookback_years=args.lookback_years)
    
    elif args.mode == 'validate':
        fetcher.validate_data_quality()


if __name__ == "__main__":
    main()
