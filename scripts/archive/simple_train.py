"""
Simple training script for debugging
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.models.deep_learning.lstm_growth_forecaster import LSTMGrowthForecaster
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR

print("="*80)
print("Simple LSTM Training")
print("="*80)

# Load data
data_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "lstm_growth_training_data.csv"
print(f"\n1. Loading data from {data_file.name}...")
df = pd.read_csv(data_file)
print(f"   Records: {len(df)}, Stocks: {df['ticker'].nunique()}")

# Create sequences
print(f"\n2. Creating sequences...")
def create_sequences(df, seq_len=20):
    X, y = [], []
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        if len(ticker_data) < seq_len + 1:
            continue
        values = ticker_data[['revenue_std', 'capex_std', 'da_std', 'nopat_std']].values
        
        for i in range(len(values) - seq_len):
            seq = values[i:i+seq_len]
            current = values[i+seq_len-1]
            next_val = values[i+seq_len]
            growth = np.zeros(4)
            for j in range(4):
                if abs(current[j]) > 1e-8:
                    growth[j] = (next_val[j] - current[j]) / abs(current[j])
            growth = np.clip(growth, -1, 1)
            X.append(seq)
            y.append(growth)
    return np.array(X), np.array(y)

X, y = create_sequences(df)
print(f"   Sequences: {len(X)}, Shape: {X.shape}")
print(f"   Has NaN in X: {np.isnan(X).any()}")
print(f"   Has NaN in y: {np.isnan(y).any()}")

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n3. Train/Val split: {len(X_train)}/{len(X_val)}")

# Datasets
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=32)

# Model
print(f"\n4. Initializing model...")
model = LSTMGrowthForecaster(input_size=4, hidden_size=64, num_layers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"   Device: {device}")

# Training
print(f"\n5. Training (10 epochs)...")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        if torch.isnan(loss):
            print(f"   ❌ NaN loss detected at epoch {epoch+1}!")
            print(f"      X_batch has NaN: {torch.isnan(X_batch).any().item()}")
            print(f"      y_batch has NaN: {torch.isnan(y_batch).any().item()}")
            print(f"      outputs has NaN: {torch.isnan(outputs).any().item()}")
            break
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    print(f"   Epoch {epoch+1:2d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

# Save
print(f"\n6. Saving model...")
save_path = MODELS_DIR / "lstm_growth_forecaster.pth"
model.save_model(str(save_path))
print(f"   Saved to: {save_path}")

# Test inference
print(f"\n7. Testing inference...")
test_input = torch.randn(1, 20, 4).to(device)
with torch.no_grad():
    test_output = model(test_input)
print(f"   Output: {test_output.squeeze().cpu().numpy()}")
print(f"   Has NaN: {torch.isnan(test_output).any().item()}")

print(f"\n✅ Training complete!")
