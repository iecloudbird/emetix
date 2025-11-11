"""Debug DCF calculations for specific stocks"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.lstm.test_lstm_dcf_real_stocks import LSTMDCFRealWorldTester
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_stock(ticker: str):
    """Debug DCF for a specific stock"""
    print(f"\n{'='*80}")
    print(f"🔍 Debugging {ticker}")
    print('='*80)
    
    tester = LSTMDCFRealWorldTester()
    
    # Get growth predictions
    growth_result = tester.predict_growth(ticker)
    if not growth_result:
        print(f"❌ Failed to get growth predictions for {ticker}")
        return
    
    # Get stock info
    stock = yf.Ticker(ticker)
    info = stock.info
    
    current_price = info.get('currentPrice', 0)
    shares = info.get('sharesOutstanding', 0)
    
    # Extract values
    current_fcf = growth_result['current_fcf']
    annual_fcf_growth = growth_result['annual_fcf_growth'] / 100 if growth_result['annual_fcf_growth'] else 0
    
    print(f"\n📊 Inputs:")
    print(f"   Current FCF: ${current_fcf/1e9:.2f}B")
    print(f"   FCF Growth Rate: {annual_fcf_growth*100:.2f}%")
    print(f"   Shares Outstanding: {shares/1e9:.2f}B")
    print(f"   Current Price: ${current_price:.2f}")
    
    # Manual DCF
    wacc = 0.08
    terminal_growth = 0.03
    
    print(f"\n💰 DCF Calculation (WACC={wacc*100}%, Terminal={terminal_growth*100}%):")
    
    fcf_forecasts = []
    fcf = current_fcf
    
    for year in range(1, 11):
        fcf = fcf * (1 + annual_fcf_growth)
        pv_fcf = fcf / (1 + wacc) ** year
        fcf_forecasts.append(pv_fcf)
        if year <= 5:
            print(f"   Year {year}: FCF ${fcf/1e9:.2f}B → PV ${pv_fcf/1e9:.2f}B")
    
    sum_pv_fcf = sum(fcf_forecasts)
    print(f"   Sum of 10Y PV FCF: ${sum_pv_fcf/1e9:.2f}B")
    
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 10
    
    print(f"   Terminal FCF (Year 11): ${terminal_fcf/1e9:.2f}B")
    print(f"   Terminal Value: ${terminal_value/1e9:.2f}B")
    print(f"   PV Terminal Value: ${pv_terminal/1e9:.2f}B")
    
    enterprise_value = sum_pv_fcf + pv_terminal
    fair_value_per_share = enterprise_value / shares
    margin = ((fair_value_per_share - current_price) / current_price) * 100
    
    print(f"\n✅ Results:")
    print(f"   Enterprise Value: ${enterprise_value/1e9:.2f}B")
    print(f"   Fair Value/Share: ${fair_value_per_share:.2f}")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Margin of Safety: {margin:+.1f}%")
    
    # Sanity checks
    print(f"\n🔍 Sanity Checks:")
    print(f"   Terminal Value % of Total: {(pv_terminal/enterprise_value)*100:.1f}%")
    print(f"   P/FCF Current: {(current_price * shares) / current_fcf:.1f}x")
    print(f"   Implied P/FCF Fair: {enterprise_value / current_fcf:.1f}x")
    
    # Check for issues
    if current_fcf < 0:
        print(f"   ⚠️  WARNING: Negative FCF!")
    if annual_fcf_growth > 0.50:
        print(f"   ⚠️  WARNING: Growth rate > 50% seems high!")
    if fair_value_per_share > current_price * 10:
        print(f"   ⚠️  WARNING: Fair value 10x+ current price!")
    if fair_value_per_share < current_price * 0.1:
        print(f"   ⚠️  WARNING: Fair value < 10% of current price!")

if __name__ == "__main__":
    # Debug problematic stocks
    debug_stock('AAPL')   # Should be reasonable
    debug_stock('MSFT')   # Showing $61k (broken)
    debug_stock('PG')     # Showing $1.2M (very broken)
