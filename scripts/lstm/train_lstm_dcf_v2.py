"""
LSTM-DCF Training v2 - Fixed Training Pipeline
==============================================
Fixes mode collapse issue by:
1. Winsorizing extreme growth rates
2. Using Huber loss instead of MSE
3. Adding L2 regularization
4. Proper normalization of targets

Usage:
    python scripts/lstm/train_lstm_dcf_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class LSTMDCFModelV2(pl.LightningModule):
    """
    Improved LSTM-DCF with:
    - Huber loss for robustness to outliers
    - L2 regularization (weight decay)
    - Optional batch normalization
    - Gradient clipping
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.0005,
        output_size: int = 2,
        weight_decay: float = 0.01,
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.huber_delta = huber_delta
        
        # Batch normalization on input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Huber loss for robustness
        self.loss_fn = nn.HuberLoss(delta=huber_delta)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, features)
            
        Returns:
            predictions: (batch, output_size)
        """
        batch_size, seq_len, features = x.shape
        
        # Apply batch norm per time step
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Output
        predictions = self.fc_layers(last_hidden)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        
        # Also compute MAE for interpretability
        mae = torch.mean(torch.abs(y_pred - y))
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def winsorize(series: pd.Series, limits: tuple = (0.05, 0.95)) -> pd.Series:
    """
    Winsorize extreme values to percentile bounds.
    Uses 5th-95th percentile to allow wider range of predictions.
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower=lower, upper=upper)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int):
    """
    Create sliding window sequences.
    
    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples, n_targets)
        seq_length: Sequence length
        
    Returns:
        X_seq: (n_sequences, seq_length, n_features)
        y_seq: (n_sequences, n_targets)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])  # Predict the next quarter's growth
    
    return np.array(X_seq), np.array(y_seq)


def train_lstm_dcf_v2():
    """Train improved LSTM-DCF model."""
    
    logger.info("=" * 80)
    logger.info("LSTM-DCF v2 TRAINING - FIXED PIPELINE")
    logger.info("=" * 80)
    logger.info("\nImprovements over v1:")
    logger.info("  1. Winsorized growth rates (1st-99th percentile)")
    logger.info("  2. Huber loss (robust to outliers)")
    logger.info("  3. L2 regularization (weight decay)")
    logger.info("  4. RobustScaler for features")
    logger.info("  5. Normalized targets (StandardScaler)")
    
    # Configuration
    sequence_length = 8  # 8 quarters = 2 years
    batch_size = 32
    max_epochs = 100
    learning_rate = 0.0005
    weight_decay = 0.01
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Sequence length: {sequence_length} quarters")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max epochs: {max_epochs}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Weight decay (L2): {weight_decay}")
    logger.info(f"   Hidden size: {hidden_size}")
    logger.info(f"   LSTM layers: {num_layers}")
    logger.info(f"   Dropout: {dropout}")
    
    # 1. Load training data
    data_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
    
    if not data_path.exists():
        logger.error(f"❌ Training data not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"\n✅ Loaded {len(df):,} training samples")
    logger.info(f"   Stocks: {df['ticker'].nunique()}")
    
    # 2. Define features
    feature_cols = [
        'revenue', 'capex', 'da', 'fcf', 
        'operating_cf', 'ebitda', 'total_assets',
        'net_income', 'operating_income',
        'operating_margin', 'net_margin', 'fcf_margin',
        'ebitda_margin', 'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
    ]
    target_cols = ['revenue_growth', 'fcf_growth']
    
    # 3. Winsorize targets to remove extreme outliers
    logger.info(f"\n📊 Target distribution BEFORE winsorizing:")
    logger.info(f"   FCF Growth: mean={df['fcf_growth'].mean():.1f}, std={df['fcf_growth'].std():.1f}")
    logger.info(f"   Revenue Growth: mean={df['revenue_growth'].mean():.1f}, std={df['revenue_growth'].std():.1f}")
    
    # Winsorize at 5th and 95th percentile - allows wider range than before
    df['fcf_growth'] = winsorize(df['fcf_growth'], (0.05, 0.95))
    df['revenue_growth'] = winsorize(df['revenue_growth'], (0.05, 0.95))
    
    logger.info(f"\n📊 Target distribution AFTER winsorizing (5th-95th percentile):")
    logger.info(f"   FCF Growth: mean={df['fcf_growth'].mean():.1f}, std={df['fcf_growth'].std():.1f}")
    logger.info(f"   Revenue Growth: mean={df['revenue_growth'].mean():.1f}, std={df['revenue_growth'].std():.1f}")
    logger.info(f"   FCF Growth range: [{df['fcf_growth'].min():.1f}, {df['fcf_growth'].max():.1f}]")
    
    # 4. Create sequences by ticker
    all_X, all_y = [], []
    tickers_processed = 0
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        if len(ticker_data) < sequence_length + 1:
            continue
        
        X_ticker = ticker_data[feature_cols].values
        y_ticker = ticker_data[target_cols].values
        
        # Handle NaN
        X_ticker = np.nan_to_num(X_ticker, nan=0.0)
        y_ticker = np.nan_to_num(y_ticker, nan=0.0)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_ticker, y_ticker, sequence_length)
        
        if len(X_seq) > 0:
            all_X.append(X_seq)
            all_y.append(y_seq)
            tickers_processed += 1
    
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    
    logger.info(f"\n✅ Created {len(X):,} sequences from {tickers_processed} stocks")
    logger.info(f"   X shape: {X.shape}")
    logger.info(f"   y shape: {y.shape}")
    
    # 5. Scale features using RobustScaler (less sensitive to outliers)
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    feature_scaler = RobustScaler()  # More robust than StandardScaler
    X_scaled = feature_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
    
    # 6. Scale targets (IMPORTANT - normalize growth rates)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)
    
    logger.info(f"\n✅ Features scaled with RobustScaler")
    logger.info(f"✅ Targets scaled with StandardScaler")
    logger.info(f"   Target mean after scaling: {y_scaled.mean(axis=0)}")
    logger.info(f"   Target std after scaling: {y_scaled.std(axis=0)}")
    
    # 7. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"   Train: {len(X_train):,} sequences")
    logger.info(f"   Validation: {len(X_val):,} sequences")
    
    # 8. Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 9. Initialize model
    model = LSTMDCFModelV2(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        output_size=len(target_cols),
        weight_decay=weight_decay,
        huber_delta=1.0
    )
    
    logger.info(f"\n🤖 Model Architecture (v2):")
    logger.info(f"   Input size: {len(feature_cols)}")
    logger.info(f"   Hidden size: {hidden_size}")
    logger.info(f"   LSTM layers: {num_layers}")
    logger.info(f"   Output size: {len(target_cols)}")
    logger.info(f"   Dropout: {dropout}")
    logger.info(f"   Loss: HuberLoss (delta=1.0)")
    logger.info(f"   Optimizer: AdamW (weight_decay={weight_decay})")
    
    # 10. Setup callbacks
    checkpoint_dir = MODELS_DIR / "lstm_checkpoints_v2"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='lstm-dcf-v2-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )
    
    # 11. Train
    logger.info(f"\n🚀 Starting training...")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0  # Gradient clipping
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # 12. Save final model with scalers - overwrite lstm_dcf_enhanced.pth for compatibility
    # This ensures all existing code that references lstm_dcf_enhanced.pth uses the new model
    final_model_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': model.hparams,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'sequence_length': sequence_length,
        'model_version': 'v2'  # Mark as v2 model
    }, final_model_path)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING COMPLETE!")
    logger.info(f"{'=' * 80}")
    logger.info(f"\n Model saved to: {final_model_path}")
    logger.info(f"\n📊 Training Summary:")
    logger.info(f"   Best validation loss: {checkpoint_callback.best_model_score:.6f}")
    
    # 13. Quick validation test
    logger.info(f"\n🧪 Quick Validation Test:")
    model.eval()
    
    with torch.no_grad():
        # Test on a few validation samples
        sample_X = torch.FloatTensor(X_val[:10])
        sample_y = y_val[:10]
        
        predictions_scaled = model(sample_X).numpy()
        
        # Inverse transform to get actual growth rates
        predictions = target_scaler.inverse_transform(predictions_scaled)
        actuals = target_scaler.inverse_transform(sample_y)
        
        logger.info("\n   Sample Predictions vs Actuals:")
        logger.info("   Revenue Growth | FCF Growth")
        logger.info("   Pred vs Actual | Pred vs Actual")
        for i in range(5):
            logger.info(f"   {predictions[i, 0]:+6.1f} vs {actuals[i, 0]:+6.1f} | "
                       f"{predictions[i, 1]:+6.1f} vs {actuals[i, 1]:+6.1f}")
        
        # Check variance of predictions
        pred_std = predictions.std(axis=0)
        logger.info(f"\n   Prediction Std Dev: Revenue={pred_std[0]:.2f}, FCF={pred_std[1]:.2f}")
        
        if pred_std.mean() < 1.0:
            logger.warning("   ⚠️  Low prediction variance - may still have mode collapse")
        else:
            logger.info("   ✅ Good prediction variance - model differentiates inputs")
    
    logger.info(f"\n✅ Model ready for inference!")
    logger.info(f"   IMPORTANT: Use feature_scaler and target_scaler from checkpoint!")


if __name__ == '__main__':
    train_lstm_dcf_v2()
