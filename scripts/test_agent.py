"""
Test script to verify the agent setup
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents import RiskAgent
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_logger(__name__)


def test_risk_agent():
    """Test the Risk Agent"""
    logger.info("Testing Risk Agent...")
    
    try:
        # Initialize agent
        agent = RiskAgent()
        
        # Test single stock assessment
        logger.info("\n=== Testing Single Stock Assessment ===")
        result = agent.assess_risk("AAPL")
        print(f"\nTicker: {result['ticker']}")
        print(f"Analysis:\n{result['analysis']}")
        
        # Test stock comparison
        logger.info("\n=== Testing Stock Comparison ===")
        comparison = agent.compare_stocks(["AAPL", "TSLA"])
        print(f"\nComparison:\n{comparison['comparison']}")
        
        logger.info("Risk Agent test complete!")
        
    except Exception as e:
        logger.error(f"Error testing agent: {str(e)}")
        print(f"\nNote: Make sure GROQ_API_KEY is set in your environment")
        print("You can get a free API key from: https://console.groq.com")


if __name__ == "__main__":
    test_risk_agent()
