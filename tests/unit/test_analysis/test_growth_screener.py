"""
Unit tests for GrowthScreener
"""
import pandas as pd
from src.analysis import GrowthScreener


def test_growth_screener_initialization():
    """Test GrowthScreener initialization"""
    screener = GrowthScreener()
    assert screener is not None
    assert hasattr(screener, 'default_criteria')
    assert 'min_revenue_growth' in screener.default_criteria


def test_calculate_opportunity_score():
    """Test opportunity score calculation"""
    screener = GrowthScreener()
    
    # Sample data for testing
    test_data = {
        'revenue_growth': 20.0,
        'ytd_return': -10.0,
        'peg_ratio': 1.0,
        'roe': 15.0,
        'debt_equity': 0.5,
        'gross_margin': 45.0
    }
    
    score = screener._calculate_opportunity_score(test_data)
    
    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_get_fail_reasons():
    """Test fail reasons generation"""
    screener = GrowthScreener()
    
    # All checks failed
    failed_checks = {
        'revenue_growth': False,
        'ytd_return': False,
        'peg_ratio': False,
        'pe_ratio': False,
        'debt_equity': False,
        'roe': False,
        'gross_margin': False,
        'positive_fcf': False,
        'market_cap': False
    }
    
    reasons = screener._get_fail_reasons(failed_checks)
    assert isinstance(reasons, list)
    assert len(reasons) > 0


def test_assess_growth_momentum():
    """Test growth momentum assessment"""
    screener = GrowthScreener()
    
    # Strong growth case
    strong_data = {'revenue_growth': 30, 'earnings_growth': 25}
    assert screener._assess_growth_momentum(strong_data) == "STRONG"
    
    # Weak growth case
    weak_data = {'revenue_growth': 12, 'earnings_growth': 8}
    assert screener._assess_growth_momentum(weak_data) == "WEAK"


def test_assess_valuation_attractiveness():
    """Test valuation attractiveness assessment"""
    screener = GrowthScreener()
    
    # Very attractive case
    attractive_data = {'peg_ratio': 0.8, 'pe_ratio': 20}
    assert screener._assess_valuation_attractiveness(attractive_data) == "VERY ATTRACTIVE"
    
    # Expensive case
    expensive_data = {'peg_ratio': 2.5, 'pe_ratio': 35}
    assert screener._assess_valuation_attractiveness(expensive_data) == "EXPENSIVE"


def test_identify_risk_factors():
    """Test risk factor identification"""
    screener = GrowthScreener()
    
    # High risk data
    risky_data = {
        'debt_equity': 2.0,
        'current_ratio': 0.8,
        'operating_margin': 3.0,
        'beta': 2.0,
        'ytd_return': -40.0
    }
    
    risks = screener._identify_risk_factors(risky_data)
    assert isinstance(risks, list)
    assert len(risks) > 0
    assert "High debt levels" in risks


if __name__ == "__main__":
    # Run basic tests
    test_growth_screener_initialization()
    test_calculate_opportunity_score()
    test_get_fail_reasons()
    test_assess_growth_momentum()
    test_assess_valuation_attractiveness()
    test_identify_risk_factors()
    print("All basic tests passed!")