"""Debug why hybrid growth isn't improving AAPL valuation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.lstm.test_lstm_dcf_real_stocks import LSTMDCFRealWorldTester
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tester = LSTMDCFRealWorldTester()

# Get data for AAPL
df = tester.fetch_financial_data('AAPL')
growth_result = tester.predict_growth('AAPL')

print("\n" + "="*80)
print("Debugging AAPL Hybrid Growth Logic")
print("="*80 + "\n")

# Check conditions
trailing_fcf = df['fcf'].iloc[-4:].mean()
current_fcf = trailing_fcf

raw_fcf_growth = growth_result['annual_fcf_growth'] / 100
revenue_growth = growth_result['annual_revenue_growth'] / 100

print(f"Inputs:")
print(f"  Current FCF (trailing 4Q): ${current_fcf/1e9:.2f}B")
print(f"  Raw FCF Growth: {raw_fcf_growth*100:.2f}%")
print(f"  Revenue Growth: {revenue_growth*100:.2f}%")
print()

print(f"Condition Checks:")
print(f"  revenue_growth > 0.02? {revenue_growth > 0.02} ({revenue_growth:.4f} > 0.02)")
print(f"  current_fcf > 1e9? {current_fcf > 1e9} (${current_fcf/1e9:.2f}B > $1B)")
print()

if revenue_growth > 0.02 and current_fcf > 1e9:
    fcf_conversion_rate = 0.50 if revenue_growth < 0.10 else 0.60
    fcf_growth_floor = revenue_growth * fcf_conversion_rate
    
    print(f"Hybrid Calculation:")
    print(f"  FCF Conversion Rate: {fcf_conversion_rate*100:.0f}%")
    print(f"  FCF Growth Floor: {fcf_growth_floor*100:.2f}%")
    print(f"  raw_fcf_growth < fcf_growth_floor? {raw_fcf_growth < fcf_growth_floor}")
    print()
    
    if raw_fcf_growth < fcf_growth_floor:
        print(f"✅ Hybrid would apply:")
        print(f"   {raw_fcf_growth*100:.2f}% → {fcf_growth_floor*100:.2f}%")
        
        # Calculate what DCF would be with hybrid growth
        wacc = 0.08
        terminal_growth = 0.03
        shares = 14.78e9  # AAPL shares
        
        fcf_forecasts = []
        fcf = current_fcf
        for year in range(1, 11):
            fcf = fcf * (1 + fcf_growth_floor)
            pv_fcf = fcf / (1 + wacc) ** year
            fcf_forecasts.append(pv_fcf)
        
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / (1 + wacc) ** 10
        
        enterprise_value = sum(fcf_forecasts) + pv_terminal
        fair_value = enterprise_value / shares
        
        print(f"\n   Fair Value with Hybrid: ${fair_value:.2f}")
        print(f"   Current Price: $269.43")
        print(f"   Margin: {((fair_value - 269.43)/269.43)*100:+.1f}%")
    else:
        print(f"❌ Hybrid would NOT apply (raw FCF growth {raw_fcf_growth*100:.2f}% >= floor {fcf_growth_floor*100:.2f}%)")
else:
    print(f"❌ Conditions not met for hybrid growth")
