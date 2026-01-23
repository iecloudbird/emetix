"""
FCF Growth Prediction Evaluation
================================
Evaluates model predictions against actual FCF growth (not stock returns).
Uses 1-3 year horizons instead of 10-year for more relevant evaluation.

Key metrics:
1. Predicted FCF Growth vs Actual FCF Growth correlation
2. Direction accuracy (did we predict growth/decline correctly)
3. Relative ranking (do high-predicted stocks have higher actual growth)

Usage:
    python scripts/evaluation/evaluate_fcf_predictions.py
    python scripts/evaluation/evaluate_fcf_predictions.py --horizon 2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

from config.settings import MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


def load_v2_model():
    """Load the v2 LSTM model"""
    model_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
    
    from scripts.lstm.train_lstm_dcf_v2 import LSTMDCFModelV2
    hp = checkpoint['hyperparameters']
    
    model = LSTMDCFModelV2(
        input_size=hp['input_size'],
        hidden_size=hp['hidden_size'],
        num_layers=hp['num_layers'],
        dropout=hp.get('dropout', 0.3),
        output_size=hp.get('output_size', 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['feature_scaler'], checkpoint['target_scaler'], checkpoint['feature_cols']


def load_quarterly_data():
    """Load quarterly financial data for evaluation"""
    data_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def evaluate_fcf_predictions(horizon_years: int = 1):
    """
    Evaluate FCF growth predictions against actual FCF growth.
    
    Args:
        horizon_years: Number of years to look ahead for actual growth (1-3)
    """
    print("=" * 70)
    print(f"FCF GROWTH PREDICTION EVALUATION ({horizon_years}-YEAR HORIZON)")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading model...")
    model, feature_scaler, target_scaler, feature_cols = load_v2_model()
    print(f"   Model loaded: {len(feature_cols)} features")
    
    # Load data
    print("\n2. Loading quarterly data...")
    df = load_quarterly_data()
    if df is None:
        return
    
    print(f"   Loaded {len(df):,} samples from {df['ticker'].nunique()} stocks")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    
    # Configuration
    seq_length = 8  # 8 quarters = 2 years
    horizon_quarters = horizon_years * 4
    
    # Prepare evaluation data
    print(f"\n3. Preparing evaluation data (horizon = {horizon_quarters} quarters)...")
    
    results = []
    tickers_processed = 0
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        
        if len(ticker_data) < seq_length + horizon_quarters:
            continue
        
        # Get features and actual future FCF growth
        for i in range(len(ticker_data) - seq_length - horizon_quarters):
            # Input sequence
            seq_data = ticker_data.iloc[i:i+seq_length]
            
            # Future FCF (horizon_quarters ahead)
            future_idx = i + seq_length + horizon_quarters - 1
            if future_idx >= len(ticker_data):
                continue
            
            current_fcf = ticker_data.iloc[i + seq_length - 1]['fcf']
            future_fcf = ticker_data.iloc[future_idx]['fcf']
            
            # Skip if FCF is zero or negative (can't compute meaningful growth)
            if current_fcf <= 0 or future_fcf <= 0:
                continue
            
            # Calculate actual FCF CAGR
            actual_cagr = (future_fcf / current_fcf) ** (1 / horizon_years) - 1
            
            # Skip extreme values
            if abs(actual_cagr) > 2.0:  # More than 200% CAGR is noise
                continue
            
            # Prepare features for prediction
            X = seq_data[feature_cols].values
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            X_scaled = feature_scaler.transform(X.reshape(-1, len(feature_cols)))
            X_scaled = X_scaled.reshape(1, seq_length, len(feature_cols))
            
            # Predict
            with torch.no_grad():
                output = model(torch.FloatTensor(X_scaled))
                pred_scaled = output.numpy()
            
            # Inverse transform prediction
            pred = target_scaler.inverse_transform(pred_scaled)[0]
            pred_fcf_growth = pred[1] / 100  # Convert from percentage
            
            results.append({
                'ticker': ticker,
                'date': seq_data.iloc[-1]['date'],
                'predicted_fcf_growth': pred_fcf_growth,
                'actual_fcf_cagr': actual_cagr,
                'current_fcf': current_fcf,
                'future_fcf': future_fcf
            })
        
        tickers_processed += 1
    
    results_df = pd.DataFrame(results)
    print(f"   Generated {len(results_df):,} predictions from {tickers_processed} stocks")
    
    # Analysis
    print(f"\n4. Evaluation Results:")
    print("-" * 50)
    
    # Correlation
    correlation, p_value = stats.pearsonr(
        results_df['predicted_fcf_growth'], 
        results_df['actual_fcf_cagr']
    )
    print(f"\n   CORRELATION:")
    print(f"   Predicted vs Actual FCF Growth: r = {correlation:.3f} (p = {p_value:.4f})")
    
    # Direction accuracy
    results_df['predicted_direction'] = results_df['predicted_fcf_growth'] > 0
    results_df['actual_direction'] = results_df['actual_fcf_cagr'] > 0
    direction_accuracy = (results_df['predicted_direction'] == results_df['actual_direction']).mean()
    
    print(f"\n   DIRECTION ACCURACY:")
    print(f"   Correctly predicted growth/decline: {direction_accuracy:.1%}")
    
    # MAE and RMSE
    mae = np.abs(results_df['predicted_fcf_growth'] - results_df['actual_fcf_cagr']).mean()
    rmse = np.sqrt(((results_df['predicted_fcf_growth'] - results_df['actual_fcf_cagr'])**2).mean())
    
    print(f"\n   ERROR METRICS:")
    print(f"   Mean Absolute Error: {mae:.1%}")
    print(f"   RMSE: {rmse:.1%}")
    
    # Relative ranking - do high predictions lead to higher actual growth?
    print(f"\n   RELATIVE RANKING (by prediction quintile):")
    results_df['pred_quintile'] = pd.qcut(
        results_df['predicted_fcf_growth'], 5, 
        labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'],
        duplicates='drop'
    )
    
    quintile_stats = results_df.groupby('pred_quintile').agg({
        'predicted_fcf_growth': 'mean',
        'actual_fcf_cagr': 'mean',
        'ticker': 'count'
    }).rename(columns={'ticker': 'count'})
    
    print(f"\n   {'Quintile':<10} {'Pred Growth':>12} {'Actual Growth':>14} {'Count':>8}")
    print(f"   {'-'*10} {'-'*12} {'-'*14} {'-'*8}")
    for idx, row in quintile_stats.iterrows():
        print(f"   {idx:<10} {row['predicted_fcf_growth']:>11.1%} {row['actual_fcf_cagr']:>13.1%} {int(row['count']):>8}")
    
    # Check if higher predictions lead to higher actual growth
    quintile_means = quintile_stats['actual_fcf_cagr'].values
    is_monotonic = all(quintile_means[i] <= quintile_means[i+1] for i in range(len(quintile_means)-1))
    
    print(f"\n   MONOTONICITY CHECK:")
    if is_monotonic:
        print(f"   Higher predictions lead to higher actual growth (ideal)")
    else:
        # Calculate Spearman rank correlation
        rank_corr, _ = stats.spearmanr(
            results_df['predicted_fcf_growth'],
            results_df['actual_fcf_cagr']
        )
        print(f"   Spearman rank correlation: {rank_corr:.3f}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n   Model Version: v2 (no clipping)")
    print(f"   Horizon: {horizon_years} year(s)")
    print(f"   Samples: {len(results_df):,}")
    print(f"\n   Key Metrics:")
    print(f"   - Correlation: {correlation:.3f}")
    print(f"   - Direction Accuracy: {direction_accuracy:.1%}")
    print(f"   - MAE: {mae:.1%}")
    
    # Interpretation
    print(f"\n   Interpretation:")
    if correlation > 0.3:
        print(f"   + Good correlation - model has predictive signal")
    elif correlation > 0.1:
        print(f"   ~ Weak correlation - limited but present signal")
    else:
        print(f"   - No meaningful correlation")
    
    if direction_accuracy > 0.55:
        print(f"   + Good direction prediction (>{55}%)")
    elif direction_accuracy > 0.50:
        print(f"   ~ Slight edge in direction prediction")
    else:
        print(f"   - Direction prediction worse than random")
    
    # Save results
    output_path = PROCESSED_DATA_DIR / "evaluation" / f"fcf_predictions_{horizon_years}yr.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n   Results saved to: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate FCF growth predictions")
    parser.add_argument('--horizon', type=int, default=1, choices=[1, 2, 3, 5, 10],
                        help="Prediction horizon in years (default: 1)")
    args = parser.parse_args()
    
    evaluate_fcf_predictions(horizon_years=args.horizon)


if __name__ == '__main__':
    main()
