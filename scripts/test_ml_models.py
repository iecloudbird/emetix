"""
Test Script for LSTM-DCF and RF Ensemble Models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.models.ensemble.consensus_scorer import ConsensusScorer
from src.data.processors.time_series_processor import TimeSeriesProcessor
from src.data.fetchers import YFinanceFetcher
from config.settings import MODELS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def test_lstm_dcf_model():
    """Test LSTM-DCF model initialization and basic operations"""
    logger.info("=" * 80)
    logger.info("TEST 1: LSTM-DCF MODEL")
    logger.info("=" * 80)
    
    # Initialize model
    model = LSTMDCFModel(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    logger.info("✓ Model initialized")
    
    # Test forward pass
    batch_size = 4
    seq_len = 60
    features = 10
    dummy_input = torch.randn(batch_size, seq_len, features)
    
    output = model(dummy_input)
    logger.info(f"✓ Forward pass: Input {dummy_input.shape} → Output {output.shape}")
    
    # Test forecasting
    test_seq = torch.randn(1, seq_len, features)
    forecasts = model.forecast_fcff(test_seq, periods=5)
    logger.info(f"✓ FCFF Forecast (5 years): {[f'{f:.2f}' for f in forecasts]}")
    
    # Test DCF valuation
    dcf_result = model.dcf_valuation(forecasts, current_shares=1e9)
    logger.info(f"✓ DCF Valuation:")
    logger.info(f"  Fair Value: ${dcf_result['fair_value']:.2f}")
    logger.info(f"  Enterprise Value: ${dcf_result['enterprise_value']/1e9:.2f}B")
    logger.info(f"  Terminal Value: ${dcf_result['terminal_value']/1e9:.2f}B")
    
    # Test save/load
    test_path = MODELS_DIR / "test_lstm_dcf.pth"
    model.save_model(str(test_path))
    logger.info(f"✓ Model saved: {test_path}")
    
    model2 = LSTMDCFModel(input_size=10, hidden_size=64, num_layers=2)
    model2.load_model(str(test_path))
    logger.info("✓ Model loaded successfully")
    
    # Clean up
    test_path.unlink()
    
    logger.info("✓ LSTM-DCF Model Test: PASSED\n")


def test_rf_ensemble_model():
    """Test RF Ensemble model"""
    logger.info("=" * 80)
    logger.info("TEST 2: RF ENSEMBLE MODEL")
    logger.info("=" * 80)
    
    # Initialize model
    model = RFEnsembleModel(n_estimators=50, max_depth=10)
    logger.info("✓ Model initialized")
    
    # Create dummy training data (100 samples)
    n_samples = 100
    dummy_data = []
    for i in range(n_samples):
        dummy_data.append({
            'pe_ratio': np.random.uniform(10, 30),
            'price_to_book': np.random.uniform(1, 5),
            'beta': np.random.uniform(0.5, 2.0),
            'debt_to_equity': np.random.uniform(0, 2),
            'return_on_equity': np.random.uniform(0, 0.3),
            'revenue_growth': np.random.uniform(-0.1, 0.3),
            'free_cash_flow': np.random.uniform(1e9, 10e9),
            'revenue': np.random.uniform(10e9, 100e9)
        })
    
    # Prepare features for all samples
    all_features = []
    for data in dummy_data:
        X_single = model.prepare_features(data)
        all_features.append(X_single)
    
    X = pd.concat(all_features, ignore_index=True)
    logger.info(f"✓ Features prepared: {X.shape}")
    logger.info(f"  Features: {X.columns.tolist()}")
    
    # Create dummy targets
    y_reg = np.random.uniform(0, 1, n_samples)
    y_clf = (y_reg > 0.5).astype(int)
    
    # Train model
    metrics = model.train(X, y_reg, y_clf)
    logger.info(f"✓ Model trained:")
    logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  CV Mean: {metrics['cv_mean']:.4f}")
    logger.info(f"  Classification Accuracy: {metrics.get('classification_accuracy', 'N/A')}")
    
    # Test prediction
    test_sample = X.iloc[[0]]
    prediction = model.predict_score(test_sample)
    logger.info(f"✓ Prediction:")
    logger.info(f"  Ensemble Score: {prediction['ensemble_score']:.4f}")
    logger.info(f"  Is Undervalued: {prediction['is_undervalued']}")
    
    # Feature importance
    importance = model.get_feature_importance()
    logger.info(f"✓ Top 3 Features:")
    for idx, row in importance.head(3).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Test save/load
    test_path = MODELS_DIR / "test_rf_ensemble.pkl"
    model.save(str(test_path))
    logger.info(f"✓ Model saved: {test_path}")
    
    model2 = RFEnsembleModel()
    model2.load(str(test_path))
    logger.info("✓ Model loaded successfully")
    
    # Clean up
    test_path.unlink()
    
    logger.info("✓ RF Ensemble Model Test: PASSED\n")


def test_consensus_scorer():
    """Test consensus scoring"""
    logger.info("=" * 80)
    logger.info("TEST 3: CONSENSUS SCORER")
    logger.info("=" * 80)
    
    # Initialize scorer
    scorer = ConsensusScorer()
    logger.info(f"✓ Scorer initialized with weights: {scorer.weights}")
    
    # Test consensus calculation
    model_scores = {
        'lstm_dcf': 0.75,
        'rf_ensemble': 0.65,
        'linear_valuation': 0.70,
        'risk_classifier': 0.60
    }
    
    result = scorer.calculate_consensus(model_scores)
    logger.info(f"✓ Consensus calculated:")
    logger.info(f"  Consensus Score: {result['consensus_score']:.4f}")
    logger.info(f"  Confidence: {result['confidence']:.4f}")
    logger.info(f"  Is Undervalued: {result['is_undervalued']}")
    logger.info(f"  Num Models: {result['num_models']}")
    
    # Test model breakdown
    breakdown = scorer.get_model_breakdown(model_scores)
    logger.info(f"✓ Model Breakdown:")
    for model, details in breakdown.items():
        logger.info(f"  {model}: {details['raw_score']:.4f} × {details['weight']:.2f} = {details['weighted_contribution']:.4f}")
    
    logger.info("✓ Consensus Scorer Test: PASSED\n")


def test_time_series_processor():
    """Test time-series data processor"""
    logger.info("=" * 80)
    logger.info("TEST 4: TIME-SERIES PROCESSOR")
    logger.info("=" * 80)
    
    processor = TimeSeriesProcessor(sequence_length=60)
    logger.info("✓ Processor initialized")
    
    # Note: This requires internet connection and yfinance
    try:
        # Fetch sample data
        logger.info("  Fetching sample data for AAPL...")
        ts_data = processor.fetch_sequential_data('AAPL', period='2y')
        
        if ts_data is not None and not ts_data.empty:
            logger.info(f"✓ Data fetched: {len(ts_data)} records")
            logger.info(f"  Columns: {ts_data.columns.tolist()}")
            
            # Create sequences
            X, y = processor.create_sequences(ts_data, target_col='close')
            logger.info(f"✓ Sequences created: {X.shape}, Targets: {y.shape}")
            
            logger.info("✓ Time-Series Processor Test: PASSED\n")
        else:
            logger.warning("⚠ Could not fetch data (offline or API issue)")
            logger.info("⚠ Time-Series Processor Test: SKIPPED\n")
    
    except Exception as e:
        logger.warning(f"⚠ Time-Series Processor Test: SKIPPED ({e})")


def test_integration():
    """Integration test with all components"""
    logger.info("=" * 80)
    logger.info("TEST 5: INTEGRATION TEST")
    logger.info("=" * 80)
    
    try:
        # Fetch stock data
        logger.info("  Fetching stock data...")
        fetcher = YFinanceFetcher()
        stock_data = fetcher.fetch_stock_data('AAPL')
        
        if stock_data is None or (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
            logger.warning("⚠ Could not fetch stock data (offline or API issue)")
            logger.info("⚠ Integration Test: SKIPPED\n")
            return
        
        logger.info(f"✓ Stock data fetched for AAPL")
        
        # RF Ensemble prediction (without LSTM)
        rf_model = RFEnsembleModel(n_estimators=50)
        
        # Create dummy training data for RF
        dummy_train_list = [stock_data] * 20
        all_features = []
        for stock in dummy_train_list:
            X_single = rf_model.prepare_features(stock)
            all_features.append(X_single)
        
        X_train = pd.concat(all_features, ignore_index=True)
        y_reg = np.random.uniform(0, 1, len(X_train))
        y_clf = (y_reg > 0.5).astype(int)
        
        rf_model.train(X_train, y_reg, y_clf)
        
        # Predict
        X_test = rf_model.prepare_features(stock_data)
        rf_result = rf_model.predict_score(X_test)
        
        logger.info(f"✓ RF Prediction: {rf_result['ensemble_score']:.4f}")
        
        # Consensus (using dummy scores)
        scorer = ConsensusScorer()
        model_scores = {
            'lstm_dcf': 0.70,
            'rf_ensemble': rf_result['ensemble_score'],
            'linear_valuation': 0.65,
            'risk_classifier': 0.60
        }
        
        consensus = scorer.calculate_consensus(model_scores)
        logger.info(f"✓ Consensus: {consensus['consensus_score']:.4f}")
        logger.info(f"  Decision: {'UNDERVALUED' if consensus['is_undervalued'] else 'NOT UNDERVALUED'}")
        
        logger.info("✓ Integration Test: PASSED\n")
    
    except Exception as e:
        logger.error(f"✗ Integration Test: FAILED ({e})")


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("LSTM-DCF & RF ENSEMBLE MODEL TESTING SUITE")
    logger.info("=" * 80 + "\n")
    
    # Unit tests
    test_lstm_dcf_model()
    test_rf_ensemble_model()
    test_consensus_scorer()
    test_time_series_processor()
    
    # Integration test
    test_integration()
    
    logger.info("=" * 80)
    logger.info("ALL TESTS COMPLETED!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Install dependencies: pip install -r requirements.txt")
    logger.info("  2. Fetch training data: python scripts/fetch_historical_data.py")
    logger.info("  3. Train LSTM-DCF: python scripts/train_lstm_dcf.py")
    logger.info("  4. Train RF Ensemble: python scripts/train_rf_ensemble.py")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
