"""
Random Forest Risk + Sentiment Trainer
=======================================
Trains RF classifier with 14 features including 30-day news sentiment.

Features:
- 6 Risk Metrics: beta, volatility, debt/equity, volume z-score, short %, RSI
- 4 Sentiment Metrics: sentiment mean/std, news volume, relevance
- 4 Valuation Metrics: P/E, P/B, profit margin, ROE

Target: Risk Class (Low / Medium / High) based on 6-month max drawdown

Usage:
    python scripts/rf/train_rf_risk_sentiment.py
    python scripts/rf/train_rf_risk_sentiment.py --test-size 0.3 --cv-folds 10
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class RiskSentimentTrainer:
    """
    Trains Random Forest for risk classification with sentiment
    """
    
    def __init__(self, test_size: float = 0.2, cv_folds: int = 5, random_state: int = 42):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown over period"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return abs(drawdown.min())
    
    def get_news_sentiment_features(self, ticker: str) -> Dict[str, float]:
        """
        Get 30-day news sentiment from cache or return neutral defaults
        
        TODO: Implement Alpha Vantage NEWS_SENTIMENT integration
        For now, returns neutral values
        """
        # Check cache
        # cache_path = RAW_DATA_DIR / 'news_sentiment' / f'{ticker}_sentiment.csv'
        
        # Default neutral sentiment (when no news available)
        return {
            'sentiment_mean': 0.0,      # Neutral
            'sentiment_std': 0.2,       # Low volatility
            'news_volume': 10,          # Moderate coverage
            'relevance_mean': 0.5       # Medium relevance
        }
    
    def extract_features(self, ticker: str) -> Dict[str, float]:
        """
        Extract 14 features for a single stock
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='6mo')
            
            if hist.empty:
                logger.warning(f"{ticker} - No price history available")
                return None
            
            # 1. Risk Metrics (6)
            beta = info.get('beta', 1.0)
            
            # 30-day volatility (annualized)
            recent_prices = hist['Close'].tail(30)
            volatility_30d = recent_prices.pct_change().std() * np.sqrt(252) if len(recent_prices) > 1 else 0.3
            
            debt_to_equity = info.get('debtToEquity', 0) / 100  # Convert to ratio
            
            # Volume z-score (last 5 days vs 3-month avg)
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].tail(5).mean()
            volume_zscore = (recent_volume - avg_volume) / (hist['Volume'].std() + 1e-8)
            
            short_pct = info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0
            
            rsi_14 = self.calculate_rsi(hist['Close'], period=14)
            
            # 2. Sentiment Metrics (4)
            sentiment_features = self.get_news_sentiment_features(ticker)
            
            # 3. Valuation Metrics (4)
            pe_ratio = info.get('trailingPE', 20)
            if pe_ratio is None or pe_ratio < 0:
                pe_ratio = 20  # Default for negative earnings
            
            price_to_book = info.get('priceToBook', 2)
            profit_margin = info.get('profitMargins', 0.1) * 100 if info.get('profitMargins') else 10
            roe = info.get('returnOnEquity', 0.15) * 100 if info.get('returnOnEquity') else 15
            
            features = {
                # Risk (6)
                'beta_5y': beta,
                'volatility_30d': volatility_30d,
                'debt_to_equity': debt_to_equity,
                'volume_zscore': volume_zscore,
                'short_pct_float': short_pct,
                'rsi_14': rsi_14,
                
                # Sentiment (4)
                'sentiment_mean': sentiment_features['sentiment_mean'],
                'sentiment_std': sentiment_features['sentiment_std'],
                'news_volume': sentiment_features['news_volume'],
                'relevance_mean': sentiment_features['relevance_mean'],
                
                # Valuation (4)
                'pe_ratio': pe_ratio,
                'price_to_book': price_to_book,
                'profit_margin': profit_margin,
                'roe': roe
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {ticker}: {e}")
            return None
    
    def assign_risk_label(self, ticker: str) -> str:
        """
        Assign risk label based on 6-month max drawdown
        
        Low:    < 15% drawdown
        Medium: 15-30% drawdown
        High:   > 30% drawdown
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return None
            
            max_dd = self.calculate_max_drawdown(hist['Close'])
            
            if max_dd < 0.15:
                return 'Low'
            elif max_dd < 0.30:
                return 'Medium'
            else:
                return 'High'
                
        except Exception as e:
            logger.error(f"Risk labeling failed for {ticker}: {e}")
            return None
    
    def build_training_dataset(self, tickers: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset from list of tickers
        """
        logger.info(f"Building training dataset from {len(tickers)} tickers...")
        
        data = []
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{len(tickers)}...")
            
            # Extract features
            features = self.extract_features(ticker)
            if features is None:
                continue
            
            # Assign risk label
            risk_label = self.assign_risk_label(ticker)
            if risk_label is None:
                continue
            
            # Combine
            features['ticker'] = ticker
            features['risk_label'] = risk_label
            data.append(features)
        
        df = pd.DataFrame(data)
        
        logger.info(f"✅ Dataset built: {len(df)} stocks")
        logger.info(f"   Risk distribution:")
        logger.info(f"   {df['risk_label'].value_counts().to_dict()}")
        
        # Save dataset
        output_path = PROCESSED_DATA_DIR / 'training' / 'rf_risk_sentiment_data.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"   Saved to {output_path}")
        
        # Split features and target
        X = df.drop(['ticker', 'risk_label'], axis=1)
        y = df['risk_label']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train Random Forest classifier
        """
        logger.info(f"{'='*80}")
        logger.info(f"Training Random Forest Risk + Sentiment Classifier")
        logger.info(f"{'='*80}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"\nDataset split:")
        logger.info(f"   Training:   {len(X_train)} samples")
        logger.info(f"   Testing:    {len(X_test)} samples")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info(f"\nTraining Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        logger.info(f"\nPerforming {self.cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model,
            X_train_scaled,
            y_train,
            cv=self.cv_folds,
            n_jobs=-1
        )
        
        logger.info(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Test set evaluation
        y_pred = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        
        logger.info(f"\nTest Set Performance:")
        logger.info(f"   Accuracy: {test_acc:.3f}")
        
        print("\n" + "="*80)
        print("Classification Report:")
        print("="*80)
        print(classification_report(y_test, y_pred))
        
        print("\n" + "="*80)
        print("Confusion Matrix:")
        print("="*80)
        cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
        cm_df = pd.DataFrame(
            cm,
            index=['True Low', 'True Medium', 'True High'],
            columns=['Pred Low', 'Pred Medium', 'Pred High']
        )
        print(cm_df)
        
        # Feature importance
        logger.info(f"\n{'='*80}")
        logger.info(f"Feature Importance:")
        logger.info(f"{'='*80}")
        
        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(feature_imp.to_string(index=False))
        
        # Save feature importance
        imp_path = MODELS_DIR / 'rf_risk_sentiment_feature_importance.csv'
        feature_imp.to_csv(imp_path, index=False)
        logger.info(f"\n✅ Feature importance saved to {imp_path}")
        
        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_acc,
            'feature_importance': feature_imp
        }
    
    def save_model(self, metrics: Dict):
        """Save trained model and scaler"""
        model_path = MODELS_DIR / 'rf_risk_sentiment.pkl'
        
        model_bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': metrics,
            'training_date': datetime.now().isoformat(),
            'n_estimators': 200,
            'max_depth': 12,
            'test_size': self.test_size,
            'cv_folds': self.cv_folds
        }
        
        joblib.dump(model_bundle, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        
        return model_path


def get_sp500_tickers() -> List[str]:
    """Get S&P 500 tickers (sample for training)"""
    # Using a hardcoded list of 100 liquid S&P 500 stocks
    # In production, fetch from Wikipedia or use stored list
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
        'WMT', 'JPM', 'PG', 'XOM', 'UNH', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
        'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'DHR',
        'WFC', 'DIS', 'ADBE', 'VZ', 'CRM', 'NEE', 'CMCSA', 'TXN', 'PM', 'NKE',
        'UPS', 'RTX', 'HON', 'ORCL', 'INTC', 'QCOM', 'BMY', 'LIN', 'AMD', 'UNP',
        'LOW', 'COP', 'IBM', 'BA', 'AMGN', 'SBUX', 'GE', 'ELV', 'SPGI', 'DE',
        'CAT', 'GS', 'AXP', 'BLK', 'MMM', 'MDT', 'LMT', 'CVS', 'PLD', 'GILD',
        'AMT', 'BKNG', 'C', 'ADI', 'SYK', 'ZTS', 'TJX', 'MDLZ', 'ADP', 'CI',
        'TMUS', 'SO', 'REGN', 'ISRG', 'MMC', 'CB', 'PGR', 'DUK', 'MO', 'SLB',
        'EOG', 'USB', 'CSX', 'PNC', 'BDX', 'NSC', 'ITW', 'CL', 'BSX', 'NOC'
    ]
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Train RF Risk + Sentiment Classifier")
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--tickers-file',
        type=str,
        default=None,
        help='Optional CSV file with tickers (column: ticker)'
    )
    parser.add_argument(
        '--use-existing-data',
        action='store_true',
        help='Use existing training data if available'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RiskSentimentTrainer(
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )
    
    # Load or build dataset
    data_path = PROCESSED_DATA_DIR / 'training' / 'rf_risk_sentiment_data.csv'
    
    if args.use_existing_data and data_path.exists():
        logger.info(f"Loading existing training data from {data_path}")
        df = pd.read_csv(data_path)
        X = df.drop(['ticker', 'risk_label'], axis=1)
        y = df['risk_label']
        trainer.feature_names = X.columns.tolist()
    else:
        # Get tickers
        if args.tickers_file:
            tickers_df = pd.read_csv(args.tickers_file)
            tickers = tickers_df['ticker'].tolist()
        else:
            tickers = get_sp500_tickers()
        
        # Build dataset
        X, y = trainer.build_training_dataset(tickers)
    
    # Train model
    metrics = trainer.train(X, y)
    
    # Save model
    model_path = trainer.save_model(metrics)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"{'='*80}")
    print(f"\nModel saved to: {model_path}")
    print(f"CV Accuracy: {metrics['cv_accuracy']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"\nTop 5 Important Features:")
    print(metrics['feature_importance'].head().to_string(index=False))
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
