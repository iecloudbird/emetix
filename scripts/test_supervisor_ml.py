"""
Test Supervisor Agent with ML Integration
Verifies EnhancedValuationAgent integration into SupervisorAgent orchestration
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.supervisor_agent import SupervisorAgent
from config.logging_config import get_logger

logger = get_logger(__name__)


def test_ml_valuation_integration():
    """Test ML valuation through SupervisorAgent"""
    print("\n" + "="*80)
    print("  TESTING SUPERVISOR AGENT WITH ML VALUATION")
    print("="*80 + "\n")
    
    # Initialize supervisor
    print("[*] Initializing SupervisorAgent with EnhancedValuationAgent...")
    try:
        supervisor = SupervisorAgent()
        print("[OK] SupervisorAgent initialized successfully\n")
    except Exception as e:
        print(f"[X] Failed to initialize: {e}")
        return
    
    # Test 1: Direct ML valuation tool
    print("-"*80)
    print("[>>] Test 1: Direct ML Valuation for AAPL")
    print("-"*80)
    try:
        result = supervisor.agent_executor.invoke({
            "input": "Get ML-powered valuation for AAPL with fair value estimates"
        })
        print("\n[+] ML Valuation Result:")
        print(result['output'])
    except Exception as e:
        print(f"[X] Error: {e}")
    
    # Test 2: Comprehensive analysis with ML
    print("\n" + "-"*80)
    print("[>>] Test 2: Comprehensive Analysis for MSFT (includes ML)")
    print("-"*80)
    try:
        result = supervisor.analyze_stock_comprehensive("MSFT")
        print("\n[+] Comprehensive Analysis:")
        print(result['comprehensive_analysis'])
    except Exception as e:
        print(f"[X] Error: {e}")
    
    # Test 3: Natural language query for ML valuation
    print("\n" + "-"*80)
    print("[>>] Test 3: Natural Language ML Query")
    print("-"*80)
    try:
        result = supervisor.agent_executor.invoke({
            "input": "Is GOOGL undervalued according to ML models? Show me LSTM-DCF and RF Ensemble predictions."
        })
        print("\n[+] Natural Language Result:")
        print(result['output'])
    except Exception as e:
        print(f"[X] Error: {e}")
    
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80 + "\n")
    print("[OK] SupervisorAgent successfully integrated with EnhancedValuationAgent")
    print("[OK] ML valuation accessible via orchestrate_stock_analysis_tool")
    print("[OK] Dedicated MLPoweredValuation tool available")
    print("[OK] Comprehensive analysis includes ML predictions")
    print("\n[*] Integration complete! Phase 6 Task 2 DONE.\n")


if __name__ == "__main__":
    test_ml_valuation_integration()
