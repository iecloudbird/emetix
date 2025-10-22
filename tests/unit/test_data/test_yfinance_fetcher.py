"""
Unit tests for YFinanceFetcher
"""
import pytest
import pandas as pd
from src.data.fetchers import YFinanceFetcher


@pytest.fixture
def fetcher():
    """Create a YFinanceFetcher instance"""
    return YFinanceFetcher()


class TestYFinanceFetcher:
    """Test suite for YFinanceFetcher"""
    
    def test_fetch_stock_data_valid_ticker(self, fetcher):
        """Test fetching data for a valid ticker"""
        result = fetcher.fetch_stock_data('AAPL')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'ticker' in result.columns
        assert 'pe_ratio' in result.columns
        assert result['ticker'].iloc[0] == 'AAPL'
    
    def test_fetch_stock_data_invalid_ticker(self, fetcher):
        """Test fetching data for an invalid ticker"""
        result = fetcher.fetch_stock_data('INVALID123')
        
        # Should return None or empty DataFrame
        assert result is None or result.empty
    
    def test_fetch_multiple_stocks(self, fetcher):
        """Test fetching data for multiple stocks"""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        result = fetcher.fetch_multiple_stocks(tickers)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert len(result) <= len(tickers)
    
    def test_fetch_historical_prices(self, fetcher):
        """Test fetching historical price data"""
        result = fetcher.fetch_historical_prices('AAPL', period='1mo')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Close' in result.columns
        assert 'Volume' in result.columns


@pytest.mark.integration
class TestYFinanceFetcherIntegration:
    """Integration tests requiring live API calls"""
    
    def test_data_quality(self, fetcher):
        """Test that fetched data meets quality standards"""
        result = fetcher.fetch_stock_data('AAPL')
        
        if result is not None and not result.empty:
            # Check for reasonable values
            pe_ratio = result['pe_ratio'].iloc[0]
            beta = result['beta'].iloc[0]
            
            assert pe_ratio >= 0 or pe_ratio is None
            assert -5 <= beta <= 5  # Beta typically in reasonable range
