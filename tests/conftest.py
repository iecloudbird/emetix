"""
Pytest configuration and fixtures
"""
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def sample_stock_data():
    """Sample stock data for testing"""
    import pandas as pd
    
    return pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'pe_ratio': [25.5, 30.2, 28.1],
        'debt_equity': [1.2, 0.5, 0.3],
        'beta': [1.1, 0.9, 1.05],
        'current_price': [150.0, 320.0, 125.0],
        'eps': [6.0, 10.5, 4.5],
        'revenue_growth': [0.15, 0.12, 0.18]
    })


@pytest.fixture(scope="session")
def sample_risk_labels():
    """Sample risk labels for testing"""
    import pandas as pd
    return pd.Series([1, 0, 1])  # Medium, Low, Medium


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
