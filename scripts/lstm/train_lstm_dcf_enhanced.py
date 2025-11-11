"""
Enhanced Training Script for LSTM-DCF Model
Trains on REAL financial statements with growth rate prediction
"""
import sys
from pathlib import Path
# Fix path since script is in lstm/ subfolder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
import torch
import yaml

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def load_config():
    """Load model configuration"""
    # Navigate up from scripts/lstm/ to project root
    config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('lstm_dcf', {})


def select_features(df: pd.DataFrame, feature_set: str = 'core') -> list:
    """
    Select features for LSTM training
    
    Args:
        df: DataFrame with all features
        feature_set: 'core' (16 features) or 'full' (all 29 features)
        
    Returns:
        List of feature column names
    """
    if feature_set == 'core':
        # 16 key features recommended for LSTM
        features = [
            # Core fundamentals (9)
            'revenue', 'capex', 'da', 'fcf', 'operating_cf', 'ebitda',
            'total_assets', 'net_income', 'operating_income',
            # Margins (4) - Quality signals
            'operating_margin', 'net_margin', 'fcf_margin', 'ebitda_margin',
            # Normalized (3) - Scale-independent
            'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
        ]
    else:
        # All features except date, ticker, and target variables
        exclude = ['date', 'ticker', 'revenue_growth', 'fcf_growth', 'ebitda_growth',
                   'capex_growth', 'da_growth', 'revenue_growth_yoy', 'fcf_growth_yoy']
        features = [col for col in df.columns if col not in exclude]
    
    return features


def create_sequences(data: np.ndarray, targets: np.ndarray, sequence_length: int = 8):
    """
    Create sequences for LSTM training
    
    Args:
        data: Feature array (samples, features)
        targets: Target array (samples, n_targets)
        sequence_length: Number of quarters to look back
        
    Returns:
        X: (n_sequences, sequence_length, features)
        y: (n_sequences, n_targets)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    
    return np.array(X), np.array(y)


def train_lstm_dcf_enhanced():
    """Train LSTM-DCF model on enhanced financial statement data"""
    
    logger.info("=" * 100)
    logger.info("ENHANCED LSTM-DCF MODEL TRAINING")
    logger.info("Using REAL financial statements (not price-based proxies)")
    logger.info("=" * 100)
    
    # Load configuration
    config = load_config()
    sequence_length = config.get('training', {}).get('sequence_length', 8)
    batch_size = config.get('training', {}).get('batch_size', 32)
    max_epochs = config.get('training', {}).get('max_epochs', 100)
    learning_rate = config.get('model', {}).get('learning_rate', 0.001)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Sequence length: {sequence_length} quarters")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max epochs: {max_epochs}")
    logger.info(f"   Learning rate: {learning_rate}")
    
    # 1. Load CLEANED training data (use cleaned version if available)
    cleaned_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
    original_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_enhanced.csv"
    
    if cleaned_path.exists():
        data_path = cleaned_path
        logger.info(f"✅ Using CLEANED training data")
    elif original_path.exists():
        data_path = original_path
        logger.warning(f"⚠️  Using original data (not cleaned). Run: python scripts/clean_training_data.py")
    else:
        logger.error(f"❌ Training data not found")
        logger.info("Please run: python scripts/build_enhanced_training_data.py")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"\n✅ Loaded {len(df):,} training samples from {data_path.name}")
    logger.info(f"   Stocks: {df['ticker'].nunique()}")
    logger.info(f"   Features: {df.shape[1]}")
    logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 2. Select features
    feature_cols = select_features(df, feature_set='core')
    target_cols = ['revenue_growth', 'fcf_growth']  # Predict 2 growth rates
    
    logger.info(f"\n📊 Feature Selection:")
    logger.info(f"   Input features: {len(feature_cols)}")
    logger.info(f"   Target variables: {target_cols}")
    logger.info(f"   Features: {feature_cols}")
    
    # 3. Prepare sequences by ticker
    all_X, all_y = [], []
    tickers_processed = 0
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        # Need at least sequence_length + 1 quarters
        if len(ticker_data) < sequence_length + 1:
            logger.warning(f"⚠️  {ticker}: Only {len(ticker_data)} quarters, need {sequence_length + 1}")
            continue
        
        # Extract features and targets
        X_ticker = ticker_data[feature_cols].values
        y_ticker = ticker_data[target_cols].values
        
        # Handle missing values (fill with 0 for growth rates)
        X_ticker = np.nan_to_num(X_ticker, nan=0.0)
        y_ticker = np.nan_to_num(y_ticker, nan=0.0)
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_ticker, y_ticker, sequence_length)
        
        if len(X_seq) > 0:
            all_X.append(X_seq)
            all_y.append(y_seq)
            tickers_processed += 1
    
    if not all_X:
        logger.error("❌ No sequences created. Check data quality.")
        return
    
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    
    logger.info(f"\n✅ Created {len(X):,} sequences from {tickers_processed} stocks")
    logger.info(f"   X shape: {X.shape} (samples, sequence_length, features)")
    logger.info(f"   y shape: {y.shape} (samples, n_targets)")
    
    # 4. Scale features (StandardScaler for financial ratios)
    # Reshape for scaling: (samples * seq_len, features) -> scale -> reshape back
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
    
    logger.info(f"\n✅ Features scaled with StandardScaler")
    
    # 5. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"   Train: {len(X_train):,} sequences")
    logger.info(f"   Validation: {len(X_val):,} sequences")
    
    # 6. Convert to PyTorch tensors and create DataLoaders
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
    
    # 7. Initialize model
    model = LSTMDCFModel(
        input_size=len(feature_cols),
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        learning_rate=learning_rate,
        output_size=len(target_cols)
    )
    
    logger.info(f"\n🤖 Model Architecture:")
    logger.info(f"   Input size: {len(feature_cols)}")
    logger.info(f"   Hidden size: 128")
    logger.info(f"   LSTM layers: 3")
    logger.info(f"   Output size: {len(target_cols)} (growth rates)")
    logger.info(f"   Dropout: 0.2")
    
    # 8. Setup callbacks
    checkpoint_dir = MODELS_DIR / "lstm_checkpoints_enhanced"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='lstm-dcf-{epoch:02d}-{val_loss:.4f}',
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
    
    # 9. Train model
    logger.info(f"\n🚀 Starting training...")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # 10. Save final model
    final_model_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': model.hparams,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'sequence_length': sequence_length
    }, final_model_path)
    
    logger.info(f"\n{'=' * 100}")
    logger.info(f"✅ TRAINING COMPLETE!")
    logger.info(f"{'=' * 100}")
    logger.info(f"\n💾 Model saved to:")
    logger.info(f"   {final_model_path}")
    logger.info(f"\n📊 Training Summary:")
    logger.info(f"   Best validation loss: {checkpoint_callback.best_model_score:.6f}")
    logger.info(f"   Checkpoints: {checkpoint_dir}")
    
    logger.info(f"\n🎯 Model predicts:")
    for i, target in enumerate(target_cols):
        logger.info(f"   Output {i}: {target}")
    
    logger.info(f"\n✅ Ready for inference!")
    logger.info(f"   Use: model.forward(sequence) -> [revenue_growth%, fcf_growth%]")


if __name__ == '__main__':
    train_lstm_dcf_enhanced()
