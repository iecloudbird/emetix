"""Check FCF status for problematic stocks"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.lstm.test_lstm_dcf_real_stocks import LSTMDCFRealWorldTester
import logging

logging.basicConfig(level=logging.WARNING)

tester = LSTMDCFRealWorldTester()

print("\n" + "="*80)
print("FCF Analysis for Problematic Stocks")
print("="*80 + "\n")

for ticker in ['BA', 'AAPL', 'AMZN', 'AMD']:
    df = tester.fetch_financial_data(ticker)
    if df is not None:
        current_fcf = df['fcf'].iloc[-1]
        last_4q_avg = df['fcf'].iloc[-4:].mean()
        last_8q_avg = df['fcf'].iloc[-8:].mean()
        
        print(f"{ticker}:")
        print(f"  Current FCF: ${current_fcf/1e9:.2f}B")
        print(f"  Last 4Q Avg: ${last_4q_avg/1e9:.2f}B")
        print(f"  Last 8Q Avg: ${last_8q_avg/1e9:.2f}B")
        
        if current_fcf < 0:
            print(f"  ⚠️  NEGATIVE FCF - DCF will produce negative value!")
        if abs(current_fcf) < 1e8:  # Less than $100M
            print(f"  ⚠️  Near-zero FCF - DCF will be very low!")
        print()
