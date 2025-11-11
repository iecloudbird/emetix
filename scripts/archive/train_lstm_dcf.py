"""
Training Script for LSTM-DCF Model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import yaml

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.deep_learning.time_series_dataset import StockTimeSeriesDataset
from src.data.processors.time_series_processor import TimeSeriesProcessor
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def load_config():
    """Load model configuration"""
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['lstm_dcf']


def train_lstm_dcf():
    """Train LSTM-DCF model on historical data"""
    
    logger.info("=" * 80)
    logger.info("LSTM-DCF MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: {config}")
    
    # 1. Load training data
    training_dir = PROCESSED_DATA_DIR / "training"
    data_path = training_dir / "lstm_training_data.csv"
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.info("Please run: python scripts/fetch_historical_data.py")
        logger.info("This will fetch time-series data for training")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} training samples")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # 2. Prepare sequences
    processor = TimeSeriesProcessor(sequence_length=config['training']['sequence_length'])
    
    # Group by ticker and create sequences
    all_X, all_y = [], []
    tickers_processed = 0
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.drop('ticker', axis=1)
        
        if len(ticker_data) < config['training']['sequence_length']:
            logger.warning(f"Insufficient data for {ticker}, skipping")
            continue
        
        X, y = processor.create_sequences(ticker_data, target_col='close', fit_scaler=True)
        
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            tickers_processed += 1
    
    if not all_X:
        logger.error("No sequences created. Check data quality.")
        return
    
    X = np.vstack(all_X)
    y = np.hstack(all_y)
    
    logger.info(f"✓ Created {len(X)} sequences from {tickers_processed} tickers")
    logger.info(f"  Sequence shape: {X.shape}")
    
    # 3. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['training']['validation_split'], 
        random_state=42
    )
    
    logger.info(f"✓ Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # 4. Create datasets and dataloaders
    train_dataset = StockTimeSeriesDataset(X_train, y_train)
    val_dataset = StockTimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        num_workers=0
    )
    
    # 5. Initialize model
    model = LSTMDCFModel(
        input_size=X.shape[2],  # Number of features
        hidden_size=config['architecture']['hidden_size'],
        num_layers=config['architecture']['num_layers'],
        dropout=config['architecture']['dropout'],
        learning_rate=config['training']['learning_rate'],
        wacc=config['dcf_params']['wacc'],
        terminal_growth=config['dcf_params']['terminal_growth']
    )
    
    logger.info(f"✓ Model initialized with {X.shape[2]} input features")
    
    # 6. Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    
    checkpoint_dir = MODELS_DIR / "lstm_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='lstm_dcf_{epoch:02d}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    # 7. Train
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=[early_stop, checkpoint],
        accelerator='auto',  # GPU if available, CPU otherwise
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING...")
    logger.info("=" * 80)
    
    trainer.fit(model, train_loader, val_loader)
    
    # 8. Save final model and scaler
    final_path = MODELS_DIR / "lstm_dcf_final.pth"
    model.save_model(str(final_path))
    
    # Save the scaler used for normalization (critical for inference!)
    processor.save_scaler()
    
    # 9. Validation metrics
    val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✓ Final validation loss: {val_loss:.6f}")
    logger.info(f"✓ Model saved: {final_path}")
    logger.info(f"✓ Scaler saved: {processor.scaler_path}")
    logger.info(f"✓ Best checkpoint: {checkpoint.best_model_path}")
    
    return model, val_loss


if __name__ == "__main__":
    try:
        train_lstm_dcf()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
