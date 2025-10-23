"""
Training Script for LSTM Growth Forecaster
Trains LSTM model to forecast DCF component growth rates

Based on: https://www.revocm.com/articles/lstm-networks-estimating-growth-rates-dcf-models

Usage:
    python scripts/train_lstm_growth_forecaster.py --epochs 50
    python scripts/train_lstm_growth_forecaster.py --quick-test  # Fast test run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.models.deep_learning.lstm_growth_forecaster import LSTMGrowthForecaster
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class GrowthRateDataset(Dataset):
    """Dataset for LSTM growth rate forecasting"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(df: pd.DataFrame, sequence_length: int = 20):
    """
    Create training sequences from financial data
    
    Args:
        df: DataFrame with columns [ticker, date, revenue_std, capex_std, da_std, nopat_std]
        sequence_length: Number of quarters in each sequence
    
    Returns:
        X: Array of sequences (num_sequences, sequence_length, 4)
        y: Array of targets (num_sequences, 4) - growth rates for next quarter
    """
    sequences = []
    targets = []
    
    # Group by ticker
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        # Check minimum length
        if len(ticker_data) < sequence_length + 1:
            continue
        
        # Extract standardized values
        values = ticker_data[['revenue_std', 'capex_std', 'da_std', 'nopat_std']].values
        
        # Create sequences
        for i in range(len(values) - sequence_length):
            # Sequence: 20 quarters of data
            seq = values[i:i+sequence_length]
            
            # Target: Growth rates for next quarter
            current = values[i+sequence_length-1]
            next_val = values[i+sequence_length]
            
            # Calculate growth rates
            growth_rates = np.zeros(4)
            for j in range(4):
                if current[j] != 0:
                    growth_rates[j] = (next_val[j] - current[j]) / abs(current[j])
                else:
                    growth_rates[j] = 0
            
            # Clip extreme growth rates
            growth_rates = np.clip(growth_rates, -1, 1)
            
            sequences.append(seq)
            targets.append(growth_rates)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    logger.info(f"Created {len(X)} sequences from {df['ticker'].nunique()} stocks")
    return X, y


def train_model(
    model: LSTMGrowthForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the LSTM growth forecaster"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    logger.info(f"\nTraining on device: {device}")
    logger.info(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
    logger.info("="*80)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:3d}/{epochs}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = MODELS_DIR / "lstm_growth_forecaster_best.pth"
            model.save_model(str(checkpoint_path))
            
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    logger.info("="*80)
    logger.info(f"✅ Training complete! Best validation loss: {best_val_loss:.6f}")
    
    return model, best_val_loss


def evaluate_model(model: LSTMGrowthForecaster, test_loader: DataLoader, device: str):
    """Evaluate model on test set"""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((preds - targets) ** 2, axis=0)
    mae = np.mean(np.abs(preds - targets), axis=0)
    
    # R-squared
    ss_res = np.sum((targets - preds) ** 2, axis=0)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / ss_tot)
    
    components = ['Revenue', 'CapEx', 'D&A', 'NOPAT']
    
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL EVALUATION (Test Set)")
    logger.info("="*80)
    
    for i, comp in enumerate(components):
        logger.info(f"\n{comp} Growth Rate:")
        logger.info(f"  MSE:  {mse[i]:.6f}")
        logger.info(f"  MAE:  {mae[i]:.6f}")
        logger.info(f"  R²:   {r2[i]:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info(f"Average Metrics:")
    logger.info(f"  MSE:  {mse.mean():.6f}")
    logger.info(f"  MAE:  {mae.mean():.6f}")
    logger.info(f"  R²:   {r2.mean():.4f}")
    logger.info("="*80)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'avg_mse': mse.mean(),
        'avg_mae': mae.mean(),
        'avg_r2': r2.mean()
    }


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Growth Forecaster')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=20, help='Sequence length (quarters)')
    parser.add_argument('--hidden-size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with 5 epochs')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        logger.info("⚡ Quick test mode: 5 epochs")
    
    logger.info("="*80)
    logger.info("LSTM GROWTH FORECASTER TRAINING")
    logger.info("="*80)
    
    # 1. Load data
    data_file = PROCESSED_DATA_DIR / "lstm_dcf_training" / "lstm_growth_training_data.csv"
    
    if not data_file.exists():
        logger.error(f"Training data not found: {data_file}")
        logger.info("Please run: python scripts/fetch_lstm_training_data.py")
        return
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"\n📊 Dataset loaded:")
    logger.info(f"  File: {data_file}")
    logger.info(f"  Records: {len(df)}")
    logger.info(f"  Stocks: {df['ticker'].nunique()}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 2. Create sequences
    logger.info(f"\n🔄 Creating sequences (length={args.sequence_length})...")
    X, y = create_sequences(df, sequence_length=args.sequence_length)
    
    logger.info(f"  Sequences: {len(X)}")
    logger.info(f"  Input shape: {X.shape}")
    logger.info(f"  Target shape: {y.shape}")
    
    # 3. Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"\n📈 Data split:")
    logger.info(f"  Train: {len(X_train)} sequences ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} sequences ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} sequences ({len(X_test)/len(X)*100:.1f}%)")
    
    # 4. Create data loaders
    train_dataset = GrowthRateDataset(X_train, y_train)
    val_dataset = GrowthRateDataset(X_val, y_val)
    test_dataset = GrowthRateDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Initialize model
    model = LSTMGrowthForecaster(
        input_size=4,  # Revenue, CapEx, D&A, NOPAT
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2,
        output_size=4  # Growth rates
    )
    
    logger.info(f"\n🧠 Model initialized:")
    logger.info(f"  Input size: 4 (Revenue, CapEx, D&A, NOPAT)")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Num layers: {args.num_layers}")
    logger.info(f"  Output size: 4 (growth rates)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  Device: {device}")
    
    # 6. Train model
    logger.info(f"\n🚀 Starting training...")
    model, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # 7. Load best model and evaluate
    logger.info(f"\n📊 Evaluating best model...")
    best_model = LSTMGrowthForecaster(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2,
        output_size=4
    )
    best_model.load_model(str(MODELS_DIR / "lstm_growth_forecaster_best.pth"))
    
    metrics = evaluate_model(best_model, test_loader, device)
    
    # 8. Save final model
    final_path = MODELS_DIR / "lstm_growth_forecaster.pth"
    best_model.save_model(str(final_path))
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"  Best model saved: {final_path}")
    logger.info(f"  Best validation loss: {best_val_loss:.6f}")
    logger.info(f"  Test R²: {metrics['avg_r2']:.4f}")
    logger.info(f"\n🎯 Ready for integration into analyze_stock.py!")


if __name__ == "__main__":
    main()
