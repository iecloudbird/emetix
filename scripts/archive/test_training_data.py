"""Quick test to see what's failing in training"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import PROCESSED_DATA_DIR

print("Loading training data...")
data_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "lstm_growth_training_data.csv"

print(f"File path: {data_file}")
print(f"File exists: {data_file.exists()}")

if data_file.exists():
    df = pd.read_csv(data_file)
    print(f"Data loaded: {len(df)} records")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst row:\n{df.iloc[0]}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
else:
    print("File not found!")
