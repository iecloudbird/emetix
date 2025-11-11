"""Diagnose LSTM-DCF fair value calculation issue"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yfinance as yf
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.data.processors.time_series_processor import TimeSeriesProcessor

def diagnose_msft():
    """Diagnose why MSFT fair value is $12,100 instead of ~$500"""
    
    ticker = 'MSFT'
    
    print("="*70)
    print("LSTM-DCF VALUATION DIAGNOSIS FOR MSFT")
    print("="*70)
    
    # Get actual company info
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    shares = info.get('sharesOutstanding', 1)
    market_cap = info.get('marketCap', current_price * shares)
    
    print(f"\n1. ACTUAL COMPANY DATA:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Shares Outstanding: {shares:,.0f}")
    print(f"   Market Cap: ${market_cap/1e12:.2f}T")
    
    # Load model and processor
    model = LSTMDCFModel(input_size=12)
    model.load_model('models/lstm_dcf_final.pth')
    proc = TimeSeriesProcessor()
    
    # Fetch data and create sequences
    ts_data = proc.fetch_sequential_data(ticker, period='5y')
    if ts_data is None or ts_data.empty:
        print("ERROR: Could not fetch time-series data")
        return
    
    X, _ = proc.create_sequences(ts_data, target_col='close')
    if len(X) == 0:
        print("ERROR: Could not create sequences")
        return
    
    seq = torch.tensor(X[-1:], dtype=torch.float32)
    
    print(f"\n2. LSTM FORECASTS:")
    
    # Forecast FCFF WITH shares scaling
    fcff_with_shares = model.forecast_fcff(
        seq, periods=10, 
        scaler=proc.scaler, 
        fcff_feature_idx=-1,
        shares_outstanding=shares
    )
    
    print(f"   WITH shares scaling ({shares:,.0f} shares):")
    print(f"   Year 1-3: {[f'${f/1e9:.2f}B' for f in fcff_with_shares[:3]]}")
    print(f"   Per-share equivalent: {[f'${f/shares:.2f}' for f in fcff_with_shares[:3]]}")
    
    # Forecast FCFF WITHOUT shares scaling (per-share FCFF)
    fcff_per_share = model.forecast_fcff(
        seq, periods=10, 
        scaler=proc.scaler, 
        fcff_feature_idx=-1,
        shares_outstanding=1.0  # No scaling
    )
    
    print(f"\n   WITHOUT shares scaling (per-share FCFF):")
    print(f"   Year 1-3: {[f'${f:.2f}' for f in fcff_per_share[:3]]}")
    
    print(f"\n3. DCF VALUATION COMPARISON:")
    
    # DCF with aggregate FCFF
    dcf_aggregate = model.dcf_valuation(fcff_with_shares, shares)
    print(f"\n   Using AGGREGATE FCFF (current implementation):")
    print(f"   Enterprise Value: ${dcf_aggregate['enterprise_value']/1e12:.2f}T")
    print(f"   Fair Value per Share: ${dcf_aggregate['fair_value']:.2f}")
    print(f"   ❌ WRONG: ${dcf_aggregate['fair_value']:.2f} vs actual ${current_price:.2f}")
    
    # DCF with per-share FCFF (correct approach)
    dcf_per_share = model.dcf_valuation(fcff_per_share, 1.0)  # Treat as per-share
    print(f"\n   Using PER-SHARE FCFF (correct approach):")
    print(f"   Fair Value per Share: ${dcf_per_share['fair_value']:.2f}")
    print(f"   Gap vs Market: {((dcf_per_share['fair_value'] - current_price)/current_price)*100:+.2f}%")
    
    # Calculate what enterprise value SHOULD be
    correct_enterprise_value = dcf_per_share['fair_value'] * shares
    print(f"   Implied Enterprise Value: ${correct_enterprise_value/1e12:.2f}T")
    print(f"   ✓ More reasonable (cf. Market Cap ${market_cap/1e12:.2f}T)")
    
    print(f"\n4. ROOT CAUSE:")
    print(f"   The model predicts PER-SHARE FCFF (e.g., ${fcff_per_share[0]:.2f})")
    print(f"   We multiply by shares to get aggregate FCFF (${fcff_with_shares[0]/1e9:.2f}B)")
    print(f"   Then DCF divides by shares again → DOUBLE COUNTING!")
    print(f"   ")
    print(f"   Formula: fair_value = enterprise_value / shares")
    print(f"            enterprise_value = sum(FCFF_forecasts / (1+WACC)^t)")
    print(f"   ")
    print(f"   If FCFF = per_share_fcff × shares, then:")
    print(f"   fair_value = (per_share_fcff × shares) / shares = per_share_fcff × 1")
    print(f"   ")
    print(f"   But our DCF sums 10 years of massive aggregate FCFF, giving:")
    print(f"   fair_value = ${dcf_aggregate['fair_value']:.2f} (way too high!)")
    
    print(f"\n5. SOLUTION:")
    print(f"   Option A: Keep FCFF as per-share, don't multiply by shares")
    print(f"   Option B: Use aggregate FCFF but don't divide by shares in DCF")
    print(f"   Option C: Train model on aggregate FCFF (requires retraining)")
    print(f"   ")
    print(f"   RECOMMENDED: Option A - treat predictions as per-share FCFF")

if __name__ == "__main__":
    diagnose_msft()
