"""
Training Data Cleaning Pipeline
Fixes data quality issues for LSTM-DCF training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class TrainingDataCleaner:
    """Clean and prepare training data for LSTM-DCF"""
    
    def __init__(self):
        self.logger = logger
        self.growth_cap_lower = -99  # Cap at -99% (can't lose more than everything)
        self.growth_cap_upper = 500  # Cap at 500% (extreme but possible)
        
    def load_data(self, path: Path) -> pd.DataFrame:
        """Load training data"""
        df = pd.read_csv(path)
        self.logger.info(f"📥 Loaded {len(df):,} rows from {path.name}")
        return df
    
    def remove_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN"""
        growth_cols = [col for col in df.columns if 'growth' in col.lower()]
        
        inf_counts = {}
        for col in growth_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        if inf_counts:
            self.logger.info(f"♾️  Removed infinite values:")
            for col, count in inf_counts.items():
                self.logger.info(f"   {col}: {count}")
        
        return df
    
    def cap_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme growth rates"""
        growth_cols = [col for col in df.columns if 'growth' in col.lower()]
        
        capped_counts = {}
        for col in growth_cols:
            # Count extremes before capping
            extreme_count = ((df[col] > self.growth_cap_upper) | (df[col] < self.growth_cap_lower)).sum()
            
            if extreme_count > 0:
                capped_counts[col] = extreme_count
                df[col] = df[col].clip(self.growth_cap_lower, self.growth_cap_upper)
        
        if capped_counts:
            self.logger.info(f"📊 Capped extreme growth rates ({self.growth_cap_lower}% to {self.growth_cap_upper}%):")
            for col, count in capped_counts.items():
                self.logger.info(f"   {col}: {count}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        initial_count = len(df)
        
        # Strategy 1: Forward fill within ticker groups
        df = df.sort_values(['ticker', 'date'])
        fill_cols = [col for col in df.columns if col not in ['ticker', 'date']]
        df[fill_cols] = df.groupby('ticker')[fill_cols].fillna(method='ffill')
        
        after_ffill = df.isnull().sum().sum()
        self.logger.info(f"📋 Forward filled missing values")
        
        # Strategy 2: Fill remaining with median
        for col in fill_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        after_median = df.isnull().sum().sum()
        self.logger.info(f"📋 Filled remaining with median")
        
        # Strategy 3: Drop rows with any remaining NaN
        df_clean = df.dropna()
        
        final_count = len(df_clean)
        removed = initial_count - final_count
        
        self.logger.info(f"❌ Removed {removed:,} rows with persistent missing values")
        self.logger.info(f"✅ Retained {final_count:,} rows ({final_count/initial_count*100:.1f}%)")
        
        return df_clean
    
    def remove_zero_base_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where base values are zero or near-zero"""
        initial_count = len(df)
        
        # Remove rows where revenue or fcf is too small
        df_clean = df[
            (df['revenue'].abs() >= 1e6) &  # At least $1M revenue
            (df['total_assets'].abs() >= 1e6)  # At least $1M assets
        ].copy()
        
        removed = initial_count - len(df_clean)
        self.logger.info(f"🔍 Removed {removed:,} rows with zero/near-zero base values")
        
        return df_clean
    
    def filter_short_sequences(self, df: pd.DataFrame, min_quarters: int = 20) -> pd.DataFrame:
        """Remove tickers with insufficient history"""
        initial_tickers = df['ticker'].nunique()
        
        # Count quarters per ticker
        ticker_counts = df.groupby('ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_quarters].index
        
        df_clean = df[df['ticker'].isin(valid_tickers)].copy()
        
        removed_tickers = initial_tickers - len(valid_tickers)
        self.logger.info(f"📅 Removed {removed_tickers} tickers with < {min_quarters} quarters")
        self.logger.info(f"✅ Retained {len(valid_tickers)} tickers with sufficient history")
        
        return df_clean
    
    def add_robust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add robust derived features"""
        # Rolling averages for smoother trends
        df = df.sort_values(['ticker', 'date'])
        
        for col in ['revenue_growth', 'fcf_growth']:
            if col in df.columns:
                df[f'{col}_ma4'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window=4, min_periods=1).mean()
                )
        
        self.logger.info(f"✨ Added robust features (4-quarter moving averages)")
        
        return df
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> bool:
        """Validate cleaned data quality"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ CLEANED DATA VALIDATION")
        self.logger.info(f"{'='*80}")
        
        # Check 1: No inf values
        inf_counts = {col: np.isinf(df[col]).sum() for col in df.columns if df[col].dtype in [np.float64, np.float32]}
        total_inf = sum(inf_counts.values())
        
        if total_inf == 0:
            self.logger.info(f"✅ No infinite values")
        else:
            self.logger.error(f"❌ Still have {total_inf} infinite values!")
            return False
        
        # Check 2: No missing values
        missing = df.isnull().sum().sum()
        if missing == 0:
            self.logger.info(f"✅ No missing values")
        else:
            self.logger.error(f"❌ Still have {missing} missing values!")
            return False
        
        # Check 3: Reasonable growth ranges
        for col in ['revenue_growth', 'fcf_growth']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                median_val = df[col].median()
                
                self.logger.info(f"📊 {col}: min={min_val:.1f}%, median={median_val:.1f}%, max={max_val:.1f}%")
                
                if max_val > 1000:
                    self.logger.error(f"❌ {col} still has extreme values > 1000%!")
                    return False
        
        # Check 4: Sufficient data
        self.logger.info(f"📊 Final dataset:")
        self.logger.info(f"   Rows: {len(df):,}")
        self.logger.info(f"   Tickers: {df['ticker'].nunique()}")
        self.logger.info(f"   Avg quarters/ticker: {len(df) / df['ticker'].nunique():.1f}")
        
        if len(df) < 1000:
            self.logger.error(f"❌ Insufficient data after cleaning (< 1000 rows)!")
            return False
        
        self.logger.info(f"\n✅ Data validation passed!")
        return True
    
    def clean_pipeline(self, input_path: Path, output_path: Path) -> pd.DataFrame:
        """Run complete cleaning pipeline"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🧹 TRAINING DATA CLEANING PIPELINE")
        self.logger.info(f"{'='*80}\n")
        
        # Load data
        df = self.load_data(input_path)
        initial_count = len(df)
        
        # Step 1: Remove infinite values
        self.logger.info(f"\n--- STEP 1: Remove Infinite Values ---")
        df = self.remove_infinite_values(df)
        
        # Step 2: Cap extreme growth rates
        self.logger.info(f"\n--- STEP 2: Cap Extreme Growth Rates ---")
        df = self.cap_growth_rates(df)
        
        # Step 3: Remove zero/near-zero base values
        self.logger.info(f"\n--- STEP 3: Remove Zero Base Values ---")
        df = self.remove_zero_base_values(df)
        
        # Step 4: Handle missing values
        self.logger.info(f"\n--- STEP 4: Handle Missing Values ---")
        df = self.handle_missing_values(df)
        
        # Step 5: Filter short sequences
        self.logger.info(f"\n--- STEP 5: Filter Short Sequences ---")
        df = self.filter_short_sequences(df, min_quarters=20)
        
        # Step 6: Add robust features
        self.logger.info(f"\n--- STEP 6: Add Robust Features ---")
        df = self.add_robust_features(df)
        
        # Validate
        if not self.validate_cleaned_data(df):
            raise ValueError("Data validation failed!")
        
        # Save
        df.to_csv(output_path, index=False)
        self.logger.info(f"\n💾 Saved cleaned data to: {output_path}")
        self.logger.info(f"   Original: {initial_count:,} rows")
        self.logger.info(f"   Cleaned: {len(df):,} rows")
        self.logger.info(f"   Retention: {len(df)/initial_count*100:.1f}%")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ CLEANING COMPLETE")
        self.logger.info(f"{'='*80}\n")
        
        return df


def main():
    """Run cleaning pipeline"""
    cleaner = TrainingDataCleaner()
    
    input_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_enhanced.csv"
    output_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
    
    # Run pipeline
    df_clean = cleaner.clean_pipeline(input_path, output_path)
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 CLEANED DATA SUMMARY")
    print("="*80)
    
    print(f"\n✅ Ready for training:")
    print(f"   File: {output_path}")
    print(f"   Rows: {len(df_clean):,}")
    print(f"   Tickers: {df_clean['ticker'].nunique()}")
    print(f"   Features: {len(df_clean.columns)}")
    
    print(f"\n📈 Growth Rate Statistics:")
    for col in ['revenue_growth', 'fcf_growth', 'ebitda_growth']:
        if col in df_clean.columns:
            print(f"   {col}:")
            print(f"      Mean: {df_clean[col].mean():.2f}%")
            print(f"      Median: {df_clean[col].median():.2f}%")
            print(f"      Std: {df_clean[col].std():.2f}%")
    
    print("\n" + "="*80)
    print("🚀 Next step: Retrain LSTM model with cleaned data")
    print("   Command: venv\\Scripts\\python.exe scripts\\train_lstm_dcf_enhanced.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
