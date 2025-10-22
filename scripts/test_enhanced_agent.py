"""
Test Script for Enhanced Valuation Agent
Tests LSTM-DCF and RF Ensemble integration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.enhanced_valuation_agent import EnhancedValuationAgent
from config.logging_config import get_logger

logger = get_logger(__name__)

def main():
    print("=" * 80)
    print("ENHANCED VALUATION AGENT TEST")
    print("=" * 80)
    print()
    
    # Initialize agent
    print("Initializing Enhanced Valuation Agent...")
    agent = EnhancedValuationAgent()
    print("✓ Agent initialized\n")
    
    # Test queries
    test_queries = [
        {
            "name": "Traditional Valuation",
            "query": "What is the comprehensive valuation for AAPL?"
        },
        {
            "name": "LSTM-DCF Analysis",
            "query": "Use LSTM-DCF hybrid valuation to analyze Microsoft (MSFT)"
        },
        {
            "name": "RF Multi-Metric Analysis",
            "query": "Perform Random Forest multi-metric analysis on TSLA"
        },
        {
            "name": "Consensus Valuation",
            "query": "Give me a consensus valuation for GOOGL using all available models"
        },
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='* 80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*80}")
        print(f"Query: {test['query']}\n")
        
        try:
            result = agent.analyze(test['query'])
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print("ALL TESTS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
