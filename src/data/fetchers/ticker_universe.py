"""
US Stock Universe Fetcher

Fetches complete list of US-traded stocks from public sources:
- NASDAQ FTP (nasdaqtraded.txt) - Free, updated daily
- NYSE symbols
- Filters: Common stocks only, excludes ETFs, ADRs, warrants, etc.

This replaces hardcoded ticker lists with dynamic universe scanning.
"""
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import json

from config.logging_config import get_logger
from config.settings import CACHE_DIR

logger = get_logger(__name__)


class TickerUniverseFetcher:
    """
    Fetches and caches US stock ticker universe from public sources.
    
    Sources:
    1. NASDAQ FTP - All NASDAQ-traded securities (includes NYSE, AMEX, ARCA, BATS)
    
    Exchange codes in nasdaqtraded.txt:
    - Q = NASDAQ
    - N = NYSE
    - A = AMEX (NYSE American)
    - P = NYSE ARCA
    - Z = BATS
    
    Filters applied:
    - Common stocks only (excludes preferred, warrants, units)
    - Excludes ETFs, ETNs, closed-end funds
    - Excludes ADRs (optional)
    - Exchange filtering (optional)
    """
    
    NASDAQ_FTP_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    
    # Exchange code mapping
    EXCHANGE_CODES = {
        'NASDAQ': 'Q',
        'NYSE': 'N',
        'AMEX': 'A',
        'ARCA': 'P',
        'BATS': 'Z'
    }
    
    # Cache settings
    CACHE_FILE = "ticker_universe.json"
    CACHE_EXPIRY_HOURS = 24  # Refresh daily
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR / "ticker_universe"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / self.CACHE_FILE
        
    def get_all_us_tickers(
        self,
        include_otc: bool = False,
        min_market_cap: Optional[float] = None,
        exclude_adrs: bool = True,
        max_tickers: Optional[int] = None,
        force_refresh: bool = False,
        exchanges: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get all US stock tickers.
        
        Args:
            include_otc: Include OTC/Pink Sheet stocks (default False)
            min_market_cap: Minimum market cap filter (applied later via yfinance)
            exclude_adrs: Exclude American Depositary Receipts
            max_tickers: Limit number of tickers (for testing)
            force_refresh: Force refresh from source
            exchanges: List of exchanges to include: ['NASDAQ', 'NYSE', 'AMEX', 'ARCA', 'BATS']
                       Default None = all major exchanges (NASDAQ, NYSE, AMEX)
            
        Returns:
            List of ticker symbols
        """
        # Default to major exchanges if not specified
        if exchanges is None:
            exchanges = ['NASDAQ', 'NYSE', 'AMEX']
        
        # Generate cache key based on exchange selection
        cache_key = f"tickers_{'_'.join(sorted(exchanges))}"
        
        # Check cache first
        if not force_refresh:
            cached = self._load_cache()
            if cached and cached.get('cache_key') == cache_key:
                tickers = cached.get('tickers', [])
                if max_tickers:
                    tickers = tickers[:max_tickers]
                logger.info(f"Loaded {len(tickers)} tickers from cache")
                return tickers
        
        # Fetch fresh data
        tickers = self._fetch_nasdaq_tickers(include_otc, exclude_adrs, exchanges)
        
        # Save to cache with key
        self._save_cache(tickers, cache_key)
        
        if max_tickers:
            tickers = tickers[:max_tickers]
            
        logger.info(f"Fetched {len(tickers)} US tickers from {', '.join(exchanges)}")
        return tickers
    
    def _fetch_nasdaq_tickers(
        self, 
        include_otc: bool, 
        exclude_adrs: bool,
        exchanges: List[str]
    ) -> List[str]:
        """Fetch tickers from NASDAQ FTP with exchange filtering."""
        try:
            logger.info(f"Fetching ticker list from NASDAQ (exchanges: {exchanges})...")
            response = requests.get(self.NASDAQ_FTP_URL, timeout=30)
            response.raise_for_status()
            
            # Build allowed exchange codes
            allowed_codes = set()
            for ex in exchanges:
                if ex.upper() in self.EXCHANGE_CODES:
                    allowed_codes.add(self.EXCHANGE_CODES[ex.upper()])
            
            # Parse the pipe-delimited file
            lines = response.text.strip().split('\n')
            
            # Header is first line
            # Format: Nasdaq Traded|Symbol|Security Name|Listing Exchange|...
            tickers = []
            
            for line in lines[1:]:  # Skip header
                if '|' not in line:
                    continue
                    
                parts = line.split('|')
                if len(parts) < 10:
                    continue
                
                nasdaq_traded = parts[0]  # Y/N
                symbol = parts[1]
                security_name = parts[2]
                listing_exchange = parts[3]  # Q=NASDAQ, N=NYSE, A=AMEX, P=ARCA, Z=BATS
                market_category = parts[4] if len(parts) > 4 else ''
                etf = parts[5] if len(parts) > 5 else 'N'
                round_lot = parts[6] if len(parts) > 6 else '100'
                test_issue = parts[7] if len(parts) > 7 else 'N'
                financial_status = parts[8] if len(parts) > 8 else 'N'
                
                # Apply filters
                # Exchange filter - skip if not in allowed exchanges
                if listing_exchange not in allowed_codes:
                    continue
                
                # Skip test issues
                if test_issue == 'Y':
                    continue
                    
                # Skip ETFs
                if etf == 'Y':
                    continue
                
                # Skip non-traded
                if nasdaq_traded != 'Y':
                    continue
                
                # Skip preferred stocks, warrants, units (contain special chars)
                if any(char in symbol for char in [' ', '.', '$', '^', '-']):
                    # Allow simple tickers with dots like BRK.A, BRK.B
                    if '.' in symbol and len(symbol.split('.')[1]) == 1:
                        pass  # Allow class shares
                    else:
                        continue
                
                # Skip if symbol too long (usually complex securities)
                if len(symbol) > 5:
                    continue
                
                # Skip ADRs if requested
                if exclude_adrs and 'ADR' in security_name.upper():
                    continue
                
                # Skip certain keywords indicating non-common stocks
                # Note: Use word boundaries or specific patterns to avoid false positives
                # e.g., 'UNIT' should not match 'UNITED'
                skip_patterns = [
                    'WARRANT',
                    ' UNIT ',  # Space-bounded to not match UNITED
                    ' UNITS',  # Space-bounded
                    'PREFERRED',
                    'RIGHTS',
                    'ACQUISITION CORP',  # SPACs
                    'ACQUISITION CO',
                ]
                name_upper = ' ' + security_name.upper() + ' '  # Add spaces for boundary matching
                if any(pattern in name_upper for pattern in skip_patterns):
                    continue
                
                tickers.append(symbol)
            
            # Remove duplicates and sort
            tickers = sorted(list(set(tickers)))
            
            logger.info(f"Filtered to {len(tickers)} common stocks")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ tickers: {e}")
            # Return fallback list
            return self._get_fallback_tickers()
    
    def _get_fallback_tickers(self) -> List[str]:
        """Return fallback S&P 500 tickers if fetch fails"""
        logger.warning("Using fallback S&P 500 ticker list")
        return [
            # Top S&P 500 by market cap
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'XOM', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX',
            'LLY', 'MRK', 'ABBV', 'PEP', 'KO', 'AVGO', 'COST', 'TMO', 'MCD',
            'CSCO', 'ACN', 'ABT', 'DHR', 'NEE', 'WFC', 'LIN', 'CRM', 'BMY',
            'TXN', 'AMD', 'PM', 'VZ', 'T', 'COP', 'RTX', 'ORCL', 'HON', 'UPS',
        ]
    
    def _load_cache(self) -> Optional[Dict]:
        """Load cached ticker list if not expired"""
        try:
            if not self.cache_path.exists():
                return None
                
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            
            # Check expiry
            cached_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            if datetime.now() - cached_time > timedelta(hours=self.CACHE_EXPIRY_HOURS):
                logger.info("Ticker cache expired")
                return None
                
            return data
            
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return None
    
    def _save_cache(self, tickers: List[str], cache_key: str = "default"):
        """Save ticker list to cache"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'count': len(tickers),
                'cache_key': cache_key,
                'tickers': tickers
            }
            with open(self.cache_path, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Cached {len(tickers)} tickers")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_universe_by_exchange(self, exchange: str = 'ALL') -> List[str]:
        """
        Get tickers filtered by exchange.
        
        Args:
            exchange: 'NASDAQ', 'NYSE', 'AMEX', or 'ALL'
        """
        # For now, return all - exchange filtering can be added later
        return self.get_all_us_tickers()
    
    def get_sector_tickers(self, sector: str) -> List[str]:
        """
        Get tickers for a specific sector.
        Requires yfinance lookup - expensive operation.
        """
        # TODO: Implement sector filtering via batch yfinance lookup
        raise NotImplementedError("Sector filtering not yet implemented")


# Convenience function
def get_us_stock_universe(max_tickers: Optional[int] = None, force_refresh: bool = False) -> List[str]:
    """Quick access to US ticker universe"""
    fetcher = TickerUniverseFetcher()
    return fetcher.get_all_us_tickers(max_tickers=max_tickers, force_refresh=force_refresh)


if __name__ == "__main__":
    # Test the fetcher
    fetcher = TickerUniverseFetcher()
    tickers = fetcher.get_all_us_tickers(force_refresh=True)
    print(f"Total US tickers: {len(tickers)}")
    print(f"Sample (first 20): {tickers[:20]}")
    print(f"Sample (last 20): {tickers[-20:]}")
