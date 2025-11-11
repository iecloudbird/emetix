"""
Comprehensive Training Data Quality Analysis
Identifies issues in LSTM training data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.settings import PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def analyze_data_quality():
    """Comprehensive data quality analysis"""
    print("\n" + "="*80)
    print("🔍 TRAINING DATA QUALITY ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    data_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_enhanced.csv"
    df = pd.read_csv(data_path)
    
    print(f"📊 Dataset Overview:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total tickers: {df['ticker'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Columns: {len(df.columns)}")
    
    # Missing values
    print(f"\n❌ Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"   {col}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print("   ✅ No missing values")
    
    # Infinite values
    print(f"\n♾️  Infinite Values:")
    growth_cols = [col for col in df.columns if 'growth' in col.lower()]
    
    total_inf = 0
    for col in growth_cols:
        inf_count = np.isinf(df[col]).sum()
        total_inf += inf_count
        if inf_count > 0:
            print(f"   {col}: {inf_count:,} ({inf_count/len(df)*100:.1f}%)")
    
    if total_inf == 0:
        print("   ✅ No infinite values")
    
    # Growth rate distributions
    print(f"\n📈 Growth Rate Distributions (excluding inf/nan):")
    print(f"   {'Column':<25} {'Count':>8} {'Min':>8} {'25%':>8} {'Median':>8} {'75%':>8} {'Max':>8}")
    print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for col in ['revenue_growth', 'fcf_growth', 'ebitda_growth']:
        if col in df.columns:
            valid = df[col][np.isfinite(df[col])]
            if len(valid) > 0:
                print(f"   {col:<25} {len(valid):>8,} {valid.min():>8.1f} {valid.quantile(0.25):>8.1f} "
                      f"{valid.median():>8.1f} {valid.quantile(0.75):>8.1f} {valid.max():>8.1f}")
    
    # Extreme values (outliers)
    print(f"\n⚠️  Extreme Values (growth > 1000% or < -100%):")
    for col in ['revenue_growth', 'fcf_growth']:
        if col in df.columns:
            extreme = df[(df[col] > 1000) | (df[col] < -100)]
            if len(extreme) > 0:
                print(f"   {col}: {len(extreme):,} extreme values")
                print(f"      Sample tickers: {extreme['ticker'].unique()[:5].tolist()}")
    
    # Zero/near-zero base values (cause of inf)
    print(f"\n🔍 Zero/Near-Zero Base Values (causes division by zero):")
    value_cols = ['revenue', 'fcf', 'ebitda']
    for col in value_cols:
        if col in df.columns:
            near_zero = df[(df[col].abs() < 1e6) & (df[col].abs() > 0)]  # < $1M but not exactly 0
            zero = df[df[col] == 0]
            if len(near_zero) > 0 or len(zero) > 0:
                print(f"   {col}:")
                print(f"      Exactly zero: {len(zero):,}")
                print(f"      Near-zero (<$1M): {len(near_zero):,}")
    
    # Ticker-level analysis
    print(f"\n📊 Ticker-Level Issues:")
    problematic_tickers = []
    
    for ticker in df['ticker'].unique()[:20]:  # Check first 20
        ticker_data = df[df['ticker'] == ticker]
        
        # Check for too many inf values
        inf_pct = np.isinf(ticker_data['revenue_growth']).sum() / len(ticker_data) * 100
        if inf_pct > 50:
            problematic_tickers.append(f"{ticker} ({inf_pct:.0f}% inf)")
    
    if problematic_tickers:
        print(f"   Tickers with >50% inf values:")
        for t in problematic_tickers[:10]:
            print(f"      {t}")
    else:
        print(f"   ✅ No major ticker-level issues in sample")
    
    # Feature completeness
    print(f"\n📋 Feature Completeness:")
    feature_cols = [col for col in df.columns if col not in ['date', 'ticker']]
    complete_rows = df[feature_cols].notna().all(axis=1).sum()
    print(f"   Rows with all features: {complete_rows:,} ({complete_rows/len(df)*100:.1f}%)")
    print(f"   Rows with missing data: {len(df) - complete_rows:,} ({(len(df) - complete_rows)/len(df)*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 Recommended Cleaning Actions:")
    actions = []
    
    if total_inf > 0:
        actions.append("1. Cap growth rates at reasonable range (-100% to +500%)")
    
    if len(missing) > 0:
        actions.append("2. Forward-fill or drop rows with missing values")
    
    if len(extreme) > 0:
        actions.append("3. Investigate and handle extreme outliers")
    
    actions.append("4. Filter out sequences with insufficient history (< 60 quarters)")
    actions.append("5. Standardize features with robust scaler (handles outliers better)")
    
    for action in actions:
        print(f"   {action}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return df


def visualize_issues(df):
    """Create visualizations of data issues"""
    print("📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Growth rate distribution (excluding inf)
    ax1 = axes[0, 0]
    valid_revenue = df['revenue_growth'][np.isfinite(df['revenue_growth'])]
    valid_revenue_clipped = valid_revenue.clip(-100, 200)  # Clip for visualization
    ax1.hist(valid_revenue_clipped, bins=50, alpha=0.7, color='blue')
    ax1.set_xlabel('Revenue Growth (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Revenue Growth Distribution (clipped -100% to 200%)')
    ax1.axvline(0, color='red', linestyle='--', label='Zero')
    ax1.legend()
    
    # 2. FCF growth distribution
    ax2 = axes[0, 1]
    valid_fcf = df['fcf_growth'][np.isfinite(df['fcf_growth'])]
    valid_fcf_clipped = valid_fcf.clip(-100, 200)
    ax2.hist(valid_fcf_clipped, bins=50, alpha=0.7, color='green')
    ax2.set_xlabel('FCF Growth (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('FCF Growth Distribution (clipped -100% to 200%)')
    ax2.axvline(0, color='red', linestyle='--', label='Zero')
    ax2.legend()
    
    # 3. Missing/Inf values per ticker
    ax3 = axes[1, 0]
    ticker_issues = []
    for ticker in df['ticker'].unique()[:30]:
        ticker_data = df[df['ticker'] == ticker]
        inf_pct = np.isinf(ticker_data['revenue_growth']).sum() / len(ticker_data) * 100
        ticker_issues.append(inf_pct)
    
    ax3.bar(range(len(ticker_issues)), sorted(ticker_issues, reverse=True))
    ax3.set_xlabel('Ticker (sorted by issue severity)')
    ax3.set_ylabel('% Infinite Values')
    ax3.set_title('Data Quality by Ticker (sample of 30)')
    ax3.axhline(50, color='red', linestyle='--', label='50% threshold')
    ax3.legend()
    
    # 4. Feature correlation
    ax4 = axes[1, 1]
    feature_cols = ['revenue', 'fcf', 'operating_margin', 'fcf_margin']
    available_cols = [col for col in feature_cols if col in df.columns]
    if len(available_cols) > 0:
        corr_data = df[available_cols].corr()
        im = ax4.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(available_cols)))
        ax4.set_yticks(range(len(available_cols)))
        ax4.set_xticklabels(available_cols, rotation=45, ha='right')
        ax4.set_yticklabels(available_cols)
        ax4.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    output_path = PROCESSED_DATA_DIR / "data_quality_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualizations saved to: {output_path}")
    plt.close()


def main():
    """Run complete analysis"""
    df = analyze_data_quality()
    
    # Optional: Create visualizations
    try:
        visualize_issues(df)
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")


if __name__ == "__main__":
    main()
