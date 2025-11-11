"""
Unified Financial Data Fetcher for LSTM Training
Combines Alpha Vantage and Finnhub to maximize data coverage

Strategy:
1. Try Alpha Vantage first (more reliable, standardized format)
2. If Alpha Vantage fails or has insufficient data, try Finnhub
3. If both available, merge and deduplicate
4. Expand ticker universe beyond Alpha Vantage's 25/day limit

Benefits:
- More stocks: Not limited by Alpha Vantage's 25 calls/day
- Better coverage: Fills gaps when one API fails
- Faster collection: Finnhub has 60 calls/min vs 5/min
- Redundancy: Multiple data sources increase reliability
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import Dict, Optional, List
import pandas as pd
from datetime import datetime

from src.data.fetchers.alpha_vantage_financials import AlphaVantageFinancialsFetcher
from src.data.fetchers.finnhub_financials import FinnhubFinancialsFetcher
from config.logging_config import get_logger
from config.settings import PROCESSED_DATA_DIR

logger = get_logger(__name__)


class UnifiedFinancialsFetcher:
    """
    Unified fetcher combining Alpha Vantage and Finnhub
    
    Fetching Strategy:
    - Primary: Alpha Vantage (more complete, 81 quarters)
    - Secondary: Finnhub (faster, no daily limit)
    - Fallback: Use whichever source has data
    """
    
    def __init__(self):
        # Initialize both fetchers
        try:
            self.alpha_vantage = AlphaVantageFinancialsFetcher()
            self.av_available = True
            logger.info("✓ Alpha Vantage fetcher initialized")
        except Exception as e:
            logger.warning(f"Alpha Vantage unavailable: {e}")
            self.av_available = False
        
        try:
            self.finnhub = FinnhubFinancialsFetcher()
            self.finnhub_available = True
            logger.info("✓ Finnhub fetcher initialized")
        except Exception as e:
            logger.warning(f"Finnhub unavailable: {e}")
            self.finnhub_available = False
        
        if not self.av_available and not self.finnhub_available:
            raise ValueError("No financial data APIs available. Check API keys in .env")
        
        logger.info(f"Unified Fetcher: AV={self.av_available}, Finnhub={self.finnhub_available}")
    
    def fetch_with_fallback(
        self,
        ticker: str,
        min_quarters: int = 20,
        use_cache: bool = True,
        prefer_source: str = 'alpha_vantage'
    ) -> Optional[Dict]:
        """
        Fetch training data with fallback strategy
        
        Args:
            ticker: Stock ticker
            min_quarters: Minimum quarters required
            use_cache: Use cached data if available
            prefer_source: 'alpha_vantage' or 'finnhub'
        
        Returns:
            Dictionary with training data or None
        """
        sources = []
        
        # Determine order based on preference
        if prefer_source == 'alpha_vantage':
            if self.av_available:
                sources.append(('Alpha Vantage', self.alpha_vantage))
            if self.finnhub_available:
                sources.append(('Finnhub', self.finnhub))
        else:
            if self.finnhub_available:
                sources.append(('Finnhub', self.finnhub))
            if self.av_available:
                sources.append(('Alpha Vantage', self.alpha_vantage))
        
        # Try each source
        for source_name, fetcher in sources:
            try:
                logger.info(f"Trying {source_name} for {ticker}...")
                
                data = fetcher.prepare_lstm_training_data(ticker, min_quarters, use_cache)
                
                if data and data['quarters'] >= min_quarters:
                    logger.info(f"✅ {source_name}: {ticker} - {data['quarters']} quarters")
                    data['source'] = source_name
                    return data
                else:
                    quarters = data['quarters'] if data else 0
                    logger.info(f"⚠️ {source_name}: {ticker} - insufficient data ({quarters} quarters)")
                    
            except Exception as e:
                logger.warning(f"❌ {source_name} failed for {ticker}: {e}")
                continue
        
        logger.warning(f"❌ All sources failed for {ticker}")
        return None
    
    def fetch_batch_smart(
        self,
        tickers: List[str],
        min_quarters: int = 20,
        use_cache: bool = True,
        alpha_vantage_limit: int = 25
    ) -> Dict[str, Dict]:
        """
        Smart batch fetching with intelligent source selection
        
        Strategy:
        1. Check cache first for all tickers
        2. Use Alpha Vantage for first N tickers (respects daily limit)
        3. Use Finnhub for remaining tickers (no daily limit)
        
        Args:
            tickers: List of ticker symbols
            min_quarters: Minimum quarters required
            use_cache: Use cached data
            alpha_vantage_limit: Max Alpha Vantage calls (default 25/day)
        
        Returns:
            Dictionary of {ticker: training_data}
        """
        results = {}
        av_calls_used = 0
        
        logger.info(f"Smart batch fetch: {len(tickers)} tickers")
        logger.info(f"Alpha Vantage limit: {alpha_vantage_limit} calls/day")
        logger.info(f"Finnhub: unlimited (60/min)")
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            # Determine which source to use
            if self.av_available and av_calls_used < alpha_vantage_limit:
                # Use Alpha Vantage (better quality)
                prefer_source = 'alpha_vantage'
            elif self.finnhub_available:
                # Use Finnhub (no daily limit)
                prefer_source = 'finnhub'
            else:
                logger.warning(f"No sources available for {ticker}")
                continue
            
            data = self.fetch_with_fallback(
                ticker,
                min_quarters=min_quarters,
                use_cache=use_cache,
                prefer_source=prefer_source
            )
            
            if data:
                results[ticker] = data
                
                # Track Alpha Vantage usage
                if data.get('source') == 'Alpha Vantage' and not use_cache:
                    av_calls_used += 3  # 3 calls per stock (income, cashflow, balance)
            
            # Log progress
            if (i % 10) == 0:
                logger.info(f"\nProgress: {len(results)}/{i} successful")
                logger.info(f"Alpha Vantage calls used: {av_calls_used}/{alpha_vantage_limit}")
        
        logger.info(f"\n✅ Batch complete: {len(results)}/{len(tickers)} stocks fetched")
        logger.info(f"Alpha Vantage calls: {av_calls_used}/{alpha_vantage_limit}")
        
        return results
    
    def create_combined_dataset(
        self,
        tickers: List[str],
        min_quarters: int = 20,
        alpha_vantage_limit: int = 25
    ) -> pd.DataFrame:
        """
        Create combined training dataset from multiple sources
        
        Returns:
            DataFrame with columns: [ticker, date, revenue_std, capex_std, da_std, nopat_std, source]
        """
        logger.info("Creating combined dataset from multiple sources...")
        
        results = self.fetch_batch_smart(
            tickers,
            min_quarters=min_quarters,
            use_cache=True,
            alpha_vantage_limit=alpha_vantage_limit
        )
        
        if not results:
            logger.error("No data fetched!")
            return pd.DataFrame()
        
        all_data = []
        
        for ticker, data in results.items():
            std_df = data['standardized_data']
            
            training_df = pd.DataFrame({
                'ticker': ticker,
                'date': std_df['date'],
                'revenue_std': std_df['revenue_norm_std'],
                'capex_std': std_df['capex_norm_std'],
                'da_std': std_df['da_norm_std'],
                'nopat_std': std_df['nopat_norm_std'],
                'source': data.get('source', 'Unknown')
            })
            
            all_data.append(training_df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"\n📊 Combined Dataset Statistics:")
        logger.info(f"  Total records: {len(combined_df):,}")
        logger.info(f"  Unique stocks: {combined_df['ticker'].nunique()}")
        logger.info(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"\n📈 Data Sources:")
        logger.info(f"{combined_df['source'].value_counts()}")
        
        return combined_df


# Example usage
if __name__ == "__main__":
    import sys
    
    fetcher = UnifiedFinancialsFetcher()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("\n" + "="*80)
    print("TESTING UNIFIED FETCHER")
    print("="*80)
    
    # Test single fetch with fallback
    print("\n1. Testing single fetch with fallback:")
    data = fetcher.fetch_with_fallback('AAPL', min_quarters=20)
    if data:
        print(f"   ✅ {data['ticker']}: {data['quarters']} quarters from {data['source']}")
    
    # Test batch fetch
    print("\n2. Testing smart batch fetch:")
    results = fetcher.fetch_batch_smart(test_tickers, alpha_vantage_limit=10)
    print(f"   ✅ Fetched {len(results)}/{len(test_tickers)} stocks")
    
    for ticker, data in results.items():
        print(f"   - {ticker}: {data['quarters']} quarters from {data['source']}")
    
    # Test combined dataset
    print("\n3. Creating combined dataset:")
    df = fetcher.create_combined_dataset(test_tickers, alpha_vantage_limit=10)
    
    if not df.empty:
        print(f"   ✅ Dataset created: {len(df)} records, {df['ticker'].nunique()} stocks")
        print(f"\n   Data sources:")
        print(df['source'].value_counts())
    
    print("\n" + "="*80)
