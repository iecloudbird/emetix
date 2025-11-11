"""
Training Script for Random Forest Ensemble
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.data.fetchers import YFinanceFetcher
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def train_rf_ensemble():
    """Train RF ensemble on historical data"""
    
    logger.info("=" * 80)
    logger.info("RANDOM FOREST ENSEMBLE TRAINING")
    logger.info("=" * 80)
    
    # 1. Load or fetch stock data
    fetcher = YFinanceFetcher()
    
    # Sample tickers (extend with more for production)
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'UNH', 'HD', 'BAC',
        'DIS', 'NFLX', 'ADBE', 'CRM', 'PYPL'
    ]
    
    logger.info(f"Fetching data for {len(tickers)} tickers...")
    
    all_data = []
    successful = 0
    
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"  [{i}/{len(tickers)}] Fetching {ticker}...")
        data = fetcher.fetch_stock_data(ticker)
        
        if data is not None and not data.empty:
            all_data.append(data)
            successful += 1
        else:
            logger.warning(f"  Failed to fetch {ticker}")
    
    if not all_data:
        logger.error("No data fetched. Cannot train model.")
        return
    
    logger.info(f"✓ Successfully fetched {successful}/{len(tickers)} tickers")
    
    # 2. Create DataFrame
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✓ Created dataset with {len(df)} samples")
    logger.info(f"  Columns: {df.columns.tolist()}")
    
    # 3. Prepare features
    model = RFEnsembleModel(n_estimators=200, max_depth=15)
    
    # Prepare features for each stock individually, then combine
    all_features = []
    for idx, row in df.iterrows():
        stock_dict = row.to_dict()
        X_single = model.prepare_features(stock_dict, lstm_predictions=None)
        all_features.append(X_single)
    
    X = pd.concat(all_features, ignore_index=True)
    logger.info(f"✓ Prepared {len(X)} feature vectors")
    logger.info(f"  Features: {X.columns.tolist()}")
    
    # 4. Create synthetic targets (replace with actual historical returns in production)
    # For demonstration: use P/E ratio and revenue growth as proxies
    logger.info("Creating target variables...")
    
    # Regression target: estimate future returns based on fundamentals
    # Simple heuristic: low P/E + high growth = higher expected return
    pe_scores = 1 / (df['pe_ratio'].replace(0, np.nan).fillna(df['pe_ratio'].median()) + 1)
    growth_scores = df['revenue_growth'].fillna(0) / 100
    
    y_regression = (pe_scores + growth_scores).values
    y_regression = (y_regression - y_regression.min()) / (y_regression.max() - y_regression.min())  # Normalize
    
    # Classification target: undervalued if P/E < median and growth > 0
    median_pe = df['pe_ratio'].median()
    y_classification = (
        (df['pe_ratio'] < median_pe) & 
        (df['revenue_growth'] > 0)
    ).astype(int).values
    
    logger.info(f"✓ Created targets:")
    logger.info(f"  Regression range: [{y_regression.min():.3f}, {y_regression.max():.3f}]")
    logger.info(f"  Classification: {y_classification.sum()} undervalued, {len(y_classification) - y_classification.sum()} not")
    
    # 5. Train model
    logger.info("=" * 80)
    logger.info("TRAINING MODELS...")
    logger.info("=" * 80)
    
    metrics = model.train(X, y_regression, y_classification)
    
    logger.info("✓ Training complete!")
    logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  CV Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    if 'classification_accuracy' in metrics:
        logger.info(f"  Classification Accuracy: {metrics['classification_accuracy']:.4f}")
    
    # 6. Feature importance
    importance = model.get_feature_importance()
    logger.info("=" * 80)
    logger.info("TOP 10 FEATURE IMPORTANCES:")
    logger.info("=" * 80)
    for idx, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    # 7. Save model
    save_path = MODELS_DIR / "rf_ensemble.pkl"
    model.save(str(save_path))
    
    logger.info("=" * 80)
    logger.info(f"✓ Model saved: {save_path}")
    logger.info("=" * 80)
    
    # 8. Save feature importance
    importance_path = MODELS_DIR / "rf_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved: {importance_path}")
    
    return model, metrics


if __name__ == "__main__":
    try:
        train_rf_ensemble()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
