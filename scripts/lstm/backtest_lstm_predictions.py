"""
Backtest LSTM Model Against Historical Growth Rates
===================================================

Tests model predictions against actual historical growth rates to measure accuracy.
Uses rolling windows: features from quarters 1-60 predict quarter 61, then 2-61 predict 62, etc.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class LSTMBacktester:
    """Backtest LSTM predictions against historical actual growth rates"""
    
    def __init__(self, model_path: str = None):
        """Initialize with trained model"""
        self.model_path = model_path or str(MODELS_DIR / "lstm_dcf_enhanced.pth")
        self.model = None
        self._load_model()
        
        # Feature columns (must match training)
        self.feature_cols = [
            'revenue', 'capex', 'da', 'fcf', 
            'operating_cf', 'ebitda', 'total_assets',
            'net_income', 'operating_income',
            'operating_margin', 'net_margin', 'fcf_margin',
            'ebitda_margin', 'revenue_per_asset', 'fcf_per_asset', 'ebitda_per_asset'
        ]
        
        # Target columns (what we're predicting)
        self.target_cols = ['revenue_growth', 'fcf_growth']
    
    def _load_model(self):
        """Load trained LSTM model"""
        try:
            checkpoint = torch.load(self.model_path, weights_only=False, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
                input_size = checkpoint['hyperparameters']['input_size']
                hidden_size = checkpoint['hyperparameters']['hidden_size']
                num_layers = checkpoint['hyperparameters']['num_layers']
                dropout = checkpoint['hyperparameters'].get('dropout', 0.2)
                output_size = checkpoint['hyperparameters'].get('output_size', 2)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint['state_dict']
            else:
                raise ValueError("Cannot load model without hyperparameters")
            
            self.model = LSTMDCFModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_size=output_size
            )
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            logger.info(f"✅ Model loaded: {input_size} inputs, {output_size} outputs")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load cleaned training data with historical growth rates"""
        data_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"✅ Loaded {len(df)} historical records from {len(df['ticker'].unique())} tickers")
        return df
    
    def create_rolling_windows(self, ticker_data: pd.DataFrame, sequence_length: int = 60) -> List[Dict]:
        """
        Create rolling windows for backtesting
        
        Args:
            ticker_data: Historical data for one ticker (sorted by date)
            sequence_length: Number of quarters to use as features
            
        Returns:
            List of dicts with features (X) and actual targets (y_actual)
        """
        windows = []
        
        # Need at least sequence_length + 1 quarters (60 for features, 1 for target)
        if len(ticker_data) < sequence_length + 1:
            return windows
        
        # Rolling window: use quarters 0-59 to predict 60, then 1-60 to predict 61, etc.
        for i in range(len(ticker_data) - sequence_length):
            # Features: 60 quarters
            feature_window = ticker_data.iloc[i:i+sequence_length][self.feature_cols].values
            
            # Actual growth in next quarter (what we're trying to predict)
            actual_growth = ticker_data.iloc[i+sequence_length][self.target_cols].values
            
            # Metadata
            ticker = ticker_data.iloc[i]['ticker']
            prediction_date = ticker_data.iloc[i+sequence_length]['date']
            
            windows.append({
                'ticker': ticker,
                'date': prediction_date,
                'features': feature_window,
                'actual_growth': actual_growth  # [revenue_growth, fcf_growth]
            })
        
        return windows
    
    def predict_on_window(self, features: np.ndarray) -> np.ndarray:
        """
        Make prediction for one window
        
        Args:
            features: (60, 16) array of features
            
        Returns:
            (2,) array of predictions [revenue_growth, fcf_growth]
        """
        # Normalize
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std[std == 0] = 1
        features_normalized = (features - mean) / std
        
        # Convert to tensor
        X = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)  # (1, 60, 16)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(X)
            return prediction[0].cpu().numpy()  # (2,) [revenue_growth, fcf_growth]
    
    def backtest_ticker(self, ticker: str, df: pd.DataFrame) -> Dict:
        """
        Backtest model on one ticker
        
        Args:
            ticker: Stock ticker
            df: Full historical dataframe
            
        Returns:
            Dict with predictions and actuals
        """
        ticker_data = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        
        windows = self.create_rolling_windows(ticker_data)
        
        if len(windows) == 0:
            logger.warning(f"⚠️  Insufficient data for {ticker}")
            return None
        
        predictions = []
        actuals = []
        dates = []
        
        for window in windows:
            pred = self.predict_on_window(window['features'])
            predictions.append(pred)
            actuals.append(window['actual_growth'])
            dates.append(window['date'])
        
        return {
            'ticker': ticker,
            'predictions': np.array(predictions),  # (n_windows, 2)
            'actuals': np.array(actuals),          # (n_windows, 2)
            'dates': dates
        }
    
    def calculate_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Calculate accuracy metrics
        
        Args:
            predictions: (n, 2) array of predictions
            actuals: (n, 2) array of actual values
            
        Returns:
            Dict with metrics for revenue and FCF growth
        """
        # Ensure float type and remove NaN/inf values
        predictions = predictions.astype(float)
        actuals = actuals.astype(float)
        
        mask = ~(np.isnan(actuals).any(axis=1) | np.isinf(actuals).any(axis=1) | 
                 np.isnan(predictions).any(axis=1) | np.isinf(predictions).any(axis=1))
        predictions = predictions[mask]
        actuals = actuals[mask]
        
        if len(predictions) == 0:
            return None
        
        metrics = {}
        
        for i, target in enumerate(['revenue_growth', 'fcf_growth']):
            pred = predictions[:, i]
            actual = actuals[:, i]
            
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            
            # Direction accuracy (did we predict up/down correctly?)
            pred_direction = (pred > 0).astype(int)
            actual_direction = (actual > 0).astype(int)
            direction_accuracy = (pred_direction == actual_direction).mean() * 100
            
            metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'mean_pred': pred.mean(),
                'mean_actual': actual.mean(),
                'std_pred': pred.std(),
                'std_actual': actual.std()
            }
        
        return metrics
    
    def backtest_all(self, max_tickers: int = None) -> Dict:
        """
        Backtest on all tickers
        
        Args:
            max_tickers: Maximum number of tickers to test (None = all)
            
        Returns:
            Aggregated results
        """
        df = self.load_historical_data()
        tickers = df['ticker'].unique()
        
        if max_tickers:
            tickers = tickers[:max_tickers]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 BACKTESTING ON {len(tickers)} TICKERS")
        logger.info(f"{'='*80}\n")
        
        all_predictions = []
        all_actuals = []
        ticker_results = []
        
        for i, ticker in enumerate(tickers, 1):
            result = self.backtest_ticker(ticker, df)
            
            if result is None:
                continue
            
            all_predictions.append(result['predictions'])
            all_actuals.append(result['actuals'])
            
            # Calculate metrics for this ticker
            metrics = self.calculate_metrics(result['predictions'], result['actuals'])
            
            if metrics:
                ticker_results.append({
                    'ticker': ticker,
                    'n_predictions': len(result['predictions']),
                    **{f"{k}_{m}": v for k, d in metrics.items() for m, v in d.items()}
                })
                
                if i <= 5:  # Show first 5 tickers
                    logger.info(f"✅ {ticker}: {len(result['predictions'])} predictions")
                    logger.info(f"   Revenue MAE: {metrics['revenue_growth']['mae']:.2f}%")
                    logger.info(f"   FCF MAE: {metrics['fcf_growth']['mae']:.2f}%")
                    logger.info(f"   Revenue Direction Accuracy: {metrics['revenue_growth']['direction_accuracy']:.1f}%")
        
        # Aggregate all predictions
        all_predictions = np.vstack(all_predictions)
        all_actuals = np.vstack(all_actuals)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(all_predictions, all_actuals)
        
        return {
            'overall_metrics': overall_metrics,
            'ticker_results': pd.DataFrame(ticker_results),
            'all_predictions': all_predictions,
            'all_actuals': all_actuals
        }
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Create visualization of backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        predictions = results['all_predictions']
        actuals = results['all_actuals']
        metrics = results['overall_metrics']
        
        # Revenue Growth: Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5, s=10)
        ax.plot([-100, 100], [-100, 100], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual Revenue Growth (%)')
        ax.set_ylabel('Predicted Revenue Growth (%)')
        ax.set_title(f"Revenue Growth: R² = {metrics['revenue_growth']['r2']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # FCF Growth: Predicted vs Actual
        ax = axes[0, 1]
        ax.scatter(actuals[:, 1], predictions[:, 1], alpha=0.5, s=10)
        ax.plot([-100, 500], [-100, 500], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual FCF Growth (%)')
        ax.set_ylabel('Predicted FCF Growth (%)')
        ax.set_title(f"FCF Growth: R² = {metrics['fcf_growth']['r2']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Prediction Error Distribution - Revenue
        ax = axes[1, 0]
        errors = predictions[:, 0] - actuals[:, 0]
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Revenue Growth Error Distribution (MAE: {metrics['revenue_growth']['mae']:.2f}%)")
        ax.grid(True, alpha=0.3)
        
        # Prediction Error Distribution - FCF
        ax = axes[1, 1]
        errors = predictions[:, 1] - actuals[:, 1]
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"FCF Growth Error Distribution (MAE: {metrics['fcf_growth']['mae']:.2f}%)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Plot saved to {save_path}")
        
        plt.close()


def main():
    """Run backtest"""
    backtester = LSTMBacktester()
    
    # Backtest on all tickers (or limit for speed)
    results = backtester.backtest_all(max_tickers=20)  # Test on 20 tickers for speed
    
    # Print overall results
    print("\n" + "="*80)
    print("📊 OVERALL BACKTEST RESULTS")
    print("="*80 + "\n")
    
    metrics = results['overall_metrics']
    
    print("REVENUE GROWTH:")
    print(f"  MAE (Mean Absolute Error):    {metrics['revenue_growth']['mae']:.2f}%")
    print(f"  RMSE (Root Mean Squared):     {metrics['revenue_growth']['rmse']:.2f}%")
    print(f"  R² Score:                     {metrics['revenue_growth']['r2']:.3f}")
    print(f"  Direction Accuracy:           {metrics['revenue_growth']['direction_accuracy']:.1f}%")
    print(f"  Mean Predicted:               {metrics['revenue_growth']['mean_pred']:.2f}%")
    print(f"  Mean Actual:                  {metrics['revenue_growth']['mean_actual']:.2f}%")
    print()
    
    print("FCF GROWTH:")
    print(f"  MAE (Mean Absolute Error):    {metrics['fcf_growth']['mae']:.2f}%")
    print(f"  RMSE (Root Mean Squared):     {metrics['fcf_growth']['rmse']:.2f}%")
    print(f"  R² Score:                     {metrics['fcf_growth']['r2']:.3f}")
    print(f"  Direction Accuracy:           {metrics['fcf_growth']['direction_accuracy']:.1f}%")
    print(f"  Mean Predicted:               {metrics['fcf_growth']['mean_pred']:.2f}%")
    print(f"  Mean Actual:                  {metrics['fcf_growth']['mean_actual']:.2f}%")
    print()
    
    print(f"Total Predictions Tested:       {len(results['all_predictions'])}")
    print(f"Tickers Tested:                 {len(results['ticker_results'])}")
    
    # Save detailed results
    ticker_results = results['ticker_results']
    output_path = PROCESSED_DATA_DIR / "lstm_backtest_results.csv"
    ticker_results.to_csv(output_path, index=False)
    print(f"\n✅ Detailed results saved to: {output_path}")
    
    # Create visualization
    plot_path = PROCESSED_DATA_DIR / "lstm_backtest_visualization.png"
    backtester.plot_results(results, save_path=str(plot_path))
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
