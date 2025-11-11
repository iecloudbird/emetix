"""
Deep Analysis: Can LSTM Learn Structural Patterns or Does It Need More Data?
===========================================================================

Analyzes:
1. Model's ability to learn structural patterns vs noise
2. Learning curve: Does accuracy improve with more data?
3. Pattern detection: Can it identify specific company behaviors?
4. Comparison: Simple baseline vs LSTM predictions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.lstm.backtest_lstm_predictions import LSTMBacktester
from config.settings import PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def analyze_learning_patterns():
    """Analyze if LSTM learned meaningful patterns"""
    
    print("\n" + "="*80)
    print("🔬 PATTERN LEARNING ANALYSIS")
    print("="*80 + "\n")
    
    # Load backtest results
    backtester = LSTMBacktester()
    df = backtester.load_historical_data()
    
    # Test 1: Compare LSTM vs Simple Baseline
    print("TEST 1: LSTM vs Baseline Models")
    print("-" * 80)
    
    # Get predictions for 20 tickers
    results = backtester.backtest_all(max_tickers=20)
    
    predictions = results['all_predictions']
    actuals = results['all_actuals']
    
    # Baseline 1: Mean predictor (always predict historical mean)
    mean_predictor_rev = np.full_like(actuals[:, 0], actuals[:, 0].mean())
    mean_predictor_fcf = np.full_like(actuals[:, 1], actuals[:, 1].mean())
    
    # Baseline 2: Previous value predictor (assume no change)
    # For this, we'd need to track sequences, so we'll use mean as baseline
    
    # Compare MAE
    lstm_mae_rev = mean_absolute_error(actuals[:, 0], predictions[:, 0])
    baseline_mae_rev = mean_absolute_error(actuals[:, 0], mean_predictor_rev)
    
    lstm_mae_fcf = mean_absolute_error(actuals[:, 1], predictions[:, 1])
    baseline_mae_fcf = mean_absolute_error(actuals[:, 1], mean_predictor_fcf)
    
    print(f"\nRevenue Growth Prediction:")
    print(f"  LSTM MAE:           {lstm_mae_rev:.2f}%")
    print(f"  Baseline MAE:       {baseline_mae_rev:.2f}%")
    print(f"  Improvement:        {((baseline_mae_rev - lstm_mae_rev) / baseline_mae_rev * 100):.1f}%")
    print(f"  LSTM Better?        {'✅ YES' if lstm_mae_rev < baseline_mae_rev else '❌ NO'}")
    
    print(f"\nFCF Growth Prediction:")
    print(f"  LSTM MAE:           {lstm_mae_fcf:.2f}%")
    print(f"  Baseline MAE:       {baseline_mae_fcf:.2f}%")
    print(f"  Improvement:        {((baseline_mae_fcf - lstm_mae_fcf) / baseline_mae_fcf * 100):.1f}%")
    print(f"  LSTM Better?        {'✅ YES' if lstm_mae_fcf < baseline_mae_fcf else '❌ NO'}")
    
    # Test 2: Analyze company-specific patterns
    print("\n\nTEST 2: Company-Specific Pattern Detection")
    print("-" * 80)
    
    ticker_results = results['ticker_results']
    
    # Group by performance
    ticker_results['revenue_improvement'] = (
        ticker_results['revenue_growth_mean_actual'] - ticker_results['revenue_growth_mae']
    )
    
    best_tickers = ticker_results.nsmallest(5, 'revenue_growth_mae')[['ticker', 'revenue_growth_mae', 'revenue_growth_direction_accuracy']]
    worst_tickers = ticker_results.nlargest(5, 'revenue_growth_mae')[['ticker', 'revenue_growth_mae', 'revenue_growth_direction_accuracy']]
    
    print("\n✅ Best Predicted Companies (Lowest MAE):")
    print(best_tickers.to_string(index=False))
    
    print("\n❌ Worst Predicted Companies (Highest MAE):")
    print(worst_tickers.to_string(index=False))
    
    # Check if best performers have structural characteristics
    print("\n📊 Pattern Analysis:")
    best_mae = best_tickers['revenue_growth_mae'].mean()
    worst_mae = worst_tickers['revenue_growth_mae'].mean()
    print(f"  Best companies MAE:   {best_mae:.2f}%")
    print(f"  Worst companies MAE:  {worst_mae:.2f}%")
    print(f"  Difference:           {worst_mae - best_mae:.2f}%")
    
    # Test 3: Check if model learned temporal patterns
    print("\n\nTEST 3: Temporal Pattern Learning")
    print("-" * 80)
    
    # Analyze if predictions improve over time (model adapts to recent patterns)
    # For each ticker, split predictions into early vs late periods
    
    ticker = 'AAPL'  # Analyze AAPL as example
    ticker_data = df[df['ticker'] == ticker].sort_values('date')
    
    if len(ticker_data) >= 120:  # Need enough data
        result = backtester.backtest_ticker(ticker, df)
        
        if result:
            n = len(result['predictions'])
            mid = n // 2
            
            early_preds = result['predictions'][:mid]
            early_actuals = result['actuals'][:mid]
            late_preds = result['predictions'][mid:]
            late_actuals = result['actuals'][mid:]
            
            early_mae = mean_absolute_error(early_actuals[:, 0], early_preds[:, 0])
            late_mae = mean_absolute_error(late_actuals[:, 0], late_preds[:, 0])
            
            print(f"\n{ticker} Temporal Analysis:")
            print(f"  Early period MAE:   {early_mae:.2f}%")
            print(f"  Late period MAE:    {late_mae:.2f}%")
            print(f"  Improvement:        {((early_mae - late_mae) / early_mae * 100):.1f}%")
            print(f"  Learned patterns?   {'✅ YES (improving)' if late_mae < early_mae else '❌ NO (getting worse)'}")
    
    # Test 4: Analyze prediction distribution
    print("\n\nTEST 4: Prediction Distribution Analysis")
    print("-" * 80)
    
    print(f"\nRevenue Growth:")
    print(f"  Actual range:       [{actuals[:, 0].min():.1f}%, {actuals[:, 0].max():.1f}%]")
    print(f"  Predicted range:    [{predictions[:, 0].min():.1f}%, {predictions[:, 0].max():.1f}%]")
    print(f"  Actual std:         {actuals[:, 0].std():.2f}%")
    print(f"  Predicted std:      {predictions[:, 0].std():.2f}%")
    
    # Check if model is too conservative (low variance)
    variance_ratio_rev = predictions[:, 0].std() / actuals[:, 0].std()
    print(f"  Variance ratio:     {variance_ratio_rev:.2f} (1.0 = perfect, <1 = conservative)")
    
    if variance_ratio_rev < 0.5:
        print(f"  ⚠️  Model too conservative (predicting near mean)")
    
    print(f"\nFCF Growth:")
    print(f"  Actual range:       [{actuals[:, 1].min():.1f}%, {actuals[:, 1].max():.1f}%]")
    print(f"  Predicted range:    [{predictions[:, 1].min():.1f}%, {predictions[:, 1].max():.1f}%]")
    print(f"  Actual std:         {actuals[:, 1].std():.2f}%")
    print(f"  Predicted std:      {predictions[:, 1].std():.2f}%")
    
    variance_ratio_fcf = predictions[:, 1].std() / actuals[:, 1].std()
    print(f"  Variance ratio:     {variance_ratio_fcf:.2f}")
    
    if variance_ratio_fcf < 0.5:
        print(f"  ⚠️  Model too conservative (predicting near mean)")
    
    # Test 5: Data size analysis
    print("\n\nTEST 5: Data Size Impact Analysis")
    print("-" * 80)
    
    print(f"\nCurrent Training Data:")
    print(f"  Total samples:      {len(df):,}")
    print(f"  Unique tickers:     {len(df['ticker'].unique())}")
    print(f"  Avg per ticker:     {len(df) / len(df['ticker'].unique()):.1f} quarters")
    print(f"  Model parameters:   ~339,000")
    print(f"  Samples/param:      {len(df) / 339000:.2f} (rule of thumb: >10 needed)")
    
    # Calculate if we need more data
    recommended_samples = 339000 * 10  # 10x parameters
    current_samples = len(df)
    
    print(f"\nData Sufficiency Analysis:")
    print(f"  Recommended:        {recommended_samples:,} samples")
    print(f"  Current:            {current_samples:,} samples")
    print(f"  Ratio:              {current_samples / recommended_samples:.1%}")
    
    if current_samples < recommended_samples:
        print(f"  ⚠️  UNDERFITTED: Need {recommended_samples - current_samples:,} more samples")
        print(f"     This translates to ~{(recommended_samples - current_samples) / 60:.0f} more tickers")
    else:
        print(f"  ✅ SUFFICIENT DATA")
    
    # Conclusion
    print("\n\n" + "="*80)
    print("🎯 CONCLUSION & RECOMMENDATIONS")
    print("="*80)
    
    print("\n📊 Pattern Learning Capability:")
    if lstm_mae_rev < baseline_mae_rev:
        print("  ✅ LSTM beats baseline for revenue → Can learn patterns")
    else:
        print("  ❌ LSTM worse than baseline → Not learning effectively")
    
    if lstm_mae_fcf < baseline_mae_fcf:
        print("  ✅ LSTM beats baseline for FCF → Can learn patterns")
    else:
        print("  ❌ LSTM worse than baseline for FCF → High noise overwhelms patterns")
    
    print("\n📈 Data Requirements:")
    if current_samples < recommended_samples:
        improvement_potential = (1 - current_samples / recommended_samples) * 100
        print(f"  ⚠️  Current data: {current_samples / recommended_samples:.0%} of recommended")
        print(f"  💡 Potential improvement with more data: ~{improvement_potential:.0f}%")
        print(f"  🎯 Target: {(recommended_samples - current_samples) / 60:.0f} more tickers (60 quarters each)")
    else:
        print(f"  ✅ Data is sufficient - bottleneck is signal/noise ratio")
    
    print("\n🔍 Key Insights:")
    
    if variance_ratio_rev < 0.5:
        print("  • Model predicts conservatively (low variance) → Playing it safe")
    
    if variance_ratio_fcf < 0.3:
        print("  • FCF predictions collapse to near-mean → Cannot capture volatility")
    
    print(f"  • Revenue direction accuracy: {results['overall_metrics']['revenue_growth']['direction_accuracy']:.1f}% → Useful signal")
    print(f"  • FCF direction accuracy: {results['overall_metrics']['fcf_growth']['direction_accuracy']:.1f}% → Barely useful")
    
    print("\n💡 Recommendations:")
    
    if lstm_mae_rev < baseline_mae_rev:
        print("  1. ✅ Keep LSTM for REVENUE predictions (beats baseline)")
    else:
        print("  1. ❌ LSTM not effective even for revenue (use simpler model)")
    
    if current_samples < recommended_samples:
        print(f"  2. 📊 Collect {(recommended_samples - current_samples) / 60:.0f} more tickers to improve accuracy")
    
    if variance_ratio_fcf < 0.3:
        print("  3. ❌ Don't use LSTM for FCF predictions (too noisy)")
        print("     → Use revenue-based proxy instead")
    
    if results['overall_metrics']['revenue_growth']['direction_accuracy'] > 60:
        print("  4. ✅ Use LSTM for DIRECTIONAL signals (buy/hold/sell)")
        print("     → Don't rely on exact growth percentages")
    
    print("\n🚀 Next Steps:")
    print("  • Proceed to RF Ensemble (different signal type)")
    print("  • Use consensus approach (multiple models)")
    print("  • Consider collecting more tickers if budget allows")
    print("  • Focus on directional accuracy, not precise values")


if __name__ == "__main__":
    analyze_learning_patterns()
