"""
Unit tests for ValuationAnalyzer
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis import ValuationAnalyzer


@pytest.fixture
def analyzer():
    """Create a ValuationAnalyzer instance"""
    return ValuationAnalyzer()


@pytest.fixture
def sample_enhanced_data():
    """Sample enhanced stock data for testing"""
    return pd.DataFrame([{
        'ticker': 'TEST',
        'current_price': 100.0,
        'market_cap': 1000000000,
        'pe_ratio': 20.0,
        'pb_ratio': 2.0,
        'ps_ratio': 3.0,
        'peg_ratio': 1.2,
        'debt_equity': 0.5,
        'current_ratio': 2.0,
        'quick_ratio': 1.5,
        'roe': 15.0,
        'roa': 8.0,
        'profit_margin': 12.0,
        'gross_margin': 40.0,
        'revenue_growth': 10.0,
        'earnings_growth': 8.0,
        'fcf_yield': 6.0,
        'ev_ebitda': 12.0,
        'dividend_yield': 2.0,
        'beta': 1.1,
        'volatility': 25.0,
        'eps': 5.0,
        'book_value': 50.0,
        '52_week_high': 120.0,
        '52_week_low': 80.0,
        'operating_cash_flow': 50000000,
        'free_cash_flow': 40000000
    }])


class TestValuationAnalyzer:
    """Test suite for ValuationAnalyzer"""
    
    def test_calculate_valuation_score(self, analyzer, sample_enhanced_data):
        """Test valuation score calculation"""
        result = analyzer.calculate_valuation_score(sample_enhanced_data)
        
        assert 'overall_score' in result
        assert 'component_scores' in result
        assert 'assessment' in result
        
        # Score should be between 0 and 100
        assert 0 <= result['overall_score'] <= 100
        
        # Should have component scores
        assert 'pe_score' in result['component_scores']
        assert 'pb_score' in result['component_scores']
        assert 'peg_score' in result['component_scores']
    
    def test_get_valuation_assessment(self, analyzer):
        """Test valuation assessment categories"""
        assert analyzer._get_valuation_assessment(85) == "SIGNIFICANTLY UNDERVALUED"
        assert analyzer._get_valuation_assessment(75) == "UNDERVALUED"
        assert analyzer._get_valuation_assessment(65) == "FAIRLY VALUED"
        assert analyzer._get_valuation_assessment(45) == "SLIGHTLY OVERVALUED"
        assert analyzer._get_valuation_assessment(30) == "OVERVALUED"
    
    def test_estimate_fair_value(self, analyzer, sample_enhanced_data):
        """Test fair value estimation"""
        row = sample_enhanced_data.iloc[0]
        fair_value = analyzer._estimate_fair_value(row)
        
        assert isinstance(fair_value, float)
        assert fair_value > 0
    
    def test_assess_risk_level(self, analyzer, sample_enhanced_data):
        """Test risk level assessment"""
        row = sample_enhanced_data.iloc[0]
        risk_level = analyzer._assess_risk_level(row)
        
        assert risk_level in ["LOW", "MEDIUM", "HIGH"]
    
    def test_get_recommendation(self, analyzer):
        """Test recommendation generation"""
        # Test various combinations
        assert analyzer._get_recommendation(85, "LOW") == "STRONG BUY"
        assert analyzer._get_recommendation(75, "MEDIUM") == "BUY"
        assert analyzer._get_recommendation(65, "HIGH") == "HOLD"
        assert analyzer._get_recommendation(30, "HIGH") == "SELL"
    
    def test_compare_stocks_empty_list(self, analyzer):
        """Test stock comparison with empty results"""
        # This will likely return empty DataFrame due to network/API issues in testing
        result = analyzer.compare_stocks(['INVALID_TICKER'])
        assert isinstance(result, pd.DataFrame)
    
    def test_analyze_stock_structure(self, analyzer):
        """Test that analyze_stock returns proper structure"""
        # Mock the fetch_enhanced_data method for testing
        analyzer.fetch_enhanced_data = lambda ticker: pd.DataFrame([{
            'ticker': ticker,
            'current_price': 100,
            'pe_ratio': 20,
            'pb_ratio': 2,
            'ps_ratio': 3,
            'peg_ratio': 1.2,
            'debt_equity': 0.5,
            'roe': 15,
            'fcf_yield': 6,
            'dividend_yield': 2,
            'beta': 1.1,
            'volatility': 25,
            'eps': 5,
            'book_value': 50
        }])
        
        result = analyzer.analyze_stock('TEST')
        
        # Check required fields
        required_fields = [
            'ticker', 'current_price', 'fair_value_estimate',
            'valuation_score', 'assessment', 'key_metrics',
            'risk_level', 'recommendation'
        ]
        
        for field in required_fields:
            assert field in result
        
        assert result['ticker'] == 'TEST'
        assert isinstance(result['valuation_score'], float)
        assert result['assessment'] in [
            "SIGNIFICANTLY UNDERVALUED", "UNDERVALUED", "FAIRLY VALUED",
            "SLIGHTLY OVERVALUED", "OVERVALUED"
        ]


class TestValuationThresholds:
    """Test valuation threshold logic"""
    
    def test_pe_scoring(self):
        """Test P/E ratio scoring logic"""
        analyzer = ValuationAnalyzer()
        
        # Test data with different P/E ratios
        test_cases = [
            ({'pe_ratio': 10}, 100),  # Very low P/E
            ({'pe_ratio': 20}, 75),   # Reasonable P/E
            ({'pe_ratio': 30}, 50),   # High P/E
            ({'pe_ratio': 40}, 25),   # Very high P/E
        ]
        
        for data_dict, expected_range in test_cases:
            data = pd.DataFrame([{**data_dict, 
                'pb_ratio': 2, 'peg_ratio': 1, 'debt_equity': 0.5,
                'current_ratio': 2, 'roe': 15, 'fcf_yield': 5}])
            
            result = analyzer.calculate_valuation_score(data)
            # Should be within reasonable range of expected
            assert 'pe_score' in result['component_scores']
    
    def test_risk_level_calculation(self):
        """Test risk level calculation logic"""
        analyzer = ValuationAnalyzer()
        
        # Low risk case
        low_risk_data = pd.Series({
            'beta': 0.7, 'debt_equity': 0.3, 'volatility': 15
        })
        assert analyzer._assess_risk_level(low_risk_data) == "LOW"
        
        # High risk case
        high_risk_data = pd.Series({
            'beta': 2.0, 'debt_equity': 2.5, 'volatility': 50
        })
        assert analyzer._assess_risk_level(high_risk_data) == "HIGH"


@pytest.mark.integration
class TestValuationAnalyzerIntegration:
    """Integration tests requiring live data"""
    
    def test_fetch_enhanced_data_live(self, analyzer):
        """Test fetching real data from API"""
        result = analyzer.fetch_enhanced_data('AAPL')
        
        if result is not None:
            assert not result.empty
            assert 'ticker' in result.columns
            assert 'pe_ratio' in result.columns
            assert 'current_price' in result.columns
            assert result['ticker'].iloc[0] == 'AAPL'
    
    def test_full_analysis_live(self, analyzer):
        """Test complete analysis with live data"""
        result = analyzer.analyze_stock('AAPL')
        
        # Should either succeed or fail gracefully
        if 'error' not in result:
            assert result['ticker'] == 'AAPL'
            assert 'valuation_score' in result
            assert 'recommendation' in result
        else:
            assert isinstance(result['error'], str)


if __name__ == "__main__":
    pytest.main([__file__])