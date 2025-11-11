"""
Random Forest Ensemble for Multi-Metric Stock Evaluation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from typing import Dict, List, Optional
from pathlib import Path

from config.settings import MODELS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class RFEnsembleModel:
    """
    Random Forest Ensemble for stock evaluation
    
    Features:
    - LSTM fair value gap
    - Traditional valuation metrics (P/E, P/B, PEG)
    - Risk metrics (Beta, Debt/Equity)
    - Profitability metrics (ROE, FCF)
    
    Pattern: Follows project's scikit-learn model conventions
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        random_state: int = 42
    ):
        """
        Initialize RF Ensemble
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
        """
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        lstm_predictions: Optional[Dict[str, float]] = None,
        enhanced_features: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Prepare enhanced 14-feature matrix for RF Risk + Sentiment Pipeline
        
        Args:
            df: Stock data from YFinanceFetcher (dict or DataFrame)
            lstm_predictions: Optional LSTM fair value predictions
            enhanced_features: Optional technical/sentiment features
            
        Returns:
            Feature DataFrame with 14 features
        """
        # Convert dict to DataFrame if needed
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        elif not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected dict or DataFrame, got {type(df)}")
        
        # Helper function to safely extract values
        def safe_get(col_name, default=0):
            val = df.get(col_name, default)
            if isinstance(val, pd.Series):
                return val.iloc[0] if not val.empty else default
            return val
        
        # Build ENHANCED 14-feature set for Risk + Sentiment Pipeline
        features = pd.DataFrame()
        
        if enhanced_features:
            # Use enhanced technical/sentiment features (14 features)
            
            # 1-3: Core Risk Metrics
            features['beta'] = [enhanced_features.get('beta', 1.0)]
            features['debt_to_equity'] = [enhanced_features.get('debt_to_equity', 0)]
            features['30d_volatility'] = [enhanced_features.get('30d_volatility', 0.2)]
            
            # 4-5: Volume and Short Interest
            features['volume_zscore'] = [enhanced_features.get('volume_zscore', 0)]
            features['short_percent'] = [enhanced_features.get('short_percent', 0)]
            
            # 6: Technical Indicator
            features['rsi_14'] = [enhanced_features.get('rsi_14', 50)]
            
            # 7-10: News Sentiment Features
            features['sentiment_mean'] = [enhanced_features.get('sentiment_mean', 0)]
            features['sentiment_std'] = [enhanced_features.get('sentiment_std', 0.3)]
            features['news_volume'] = [enhanced_features.get('news_volume', 20)]
            features['relevance_mean'] = [enhanced_features.get('relevance_mean', 0.5)]
            
            # 11-14: Fundamental Features
            features['pe_ratio'] = [enhanced_features.get('pe_ratio', 0)]
            features['revenue_growth'] = [enhanced_features.get('revenue_growth', 0)]
            features['current_ratio'] = [enhanced_features.get('current_ratio', 1.0)]
            features['return_on_equity'] = [enhanced_features.get('return_on_equity', 0)]
            
        else:
            # Fallback to basic features (backward compatibility)
            features['beta'] = [safe_get('beta', 1.0)]
            features['debt_to_equity'] = [safe_get('debt_to_equity', 0)]
            features['30d_volatility'] = [0.2]  # Default volatility
            features['volume_zscore'] = [0]
            features['short_percent'] = [0]
            features['rsi_14'] = [50]  # Neutral RSI
            features['sentiment_mean'] = [0]  # Neutral sentiment
            features['sentiment_std'] = [0.3]  # Default volatility
            features['news_volume'] = [20]  # Default news volume
            features['relevance_mean'] = [0.5]  # Default relevance
            features['pe_ratio'] = [safe_get('pe_ratio', 0)]
            features['revenue_growth'] = [safe_get('revenue_growth', 0)]
            features['current_ratio'] = [safe_get('current_ratio', 1.0)]
            features['return_on_equity'] = [safe_get('return_on_equity', 0)]
        
        # Fill NaN with defaults
        features = features.fillna({
            'beta': 1.0, 'debt_to_equity': 0, '30d_volatility': 0.2,
            'volume_zscore': 0, 'short_percent': 0, 'rsi_14': 50,
            'sentiment_mean': 0, 'sentiment_std': 0.3, 'news_volume': 20, 'relevance_mean': 0.5,
            'pe_ratio': 0, 'revenue_growth': 0, 'current_ratio': 1.0, 'return_on_equity': 0
        })
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train(
        self, 
        X: pd.DataFrame, 
        y_regression: np.ndarray,
        y_classification: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Train RF ensemble
        
        Args:
            X: Feature matrix
            y_regression: Continuous target (e.g., forward returns)
            y_classification: Binary target (e.g., outperform vs underperform)
            cv: Cross-validation folds
            
        Returns:
            Training metrics
        """
        # Store feature names BEFORE training
        self.feature_names = X.columns.tolist()
        
        # Train regressor
        self.regressor.fit(X, y_regression)
        reg_scores = cross_val_score(self.regressor, X, y_regression, cv=cv, scoring='r2')
        
        # Train classifier  
        self.classifier.fit(X, y_classification)
        clf_scores = cross_val_score(self.classifier, X, y_classification, cv=cv, scoring='accuracy')
        
        self.is_trained = True
        
        # Return metrics
        return {
            'r2_score': float(reg_scores.mean()),
            'cv_mean': float(reg_scores.mean()),
            'cv_std': float(reg_scores.std()),
            'classification_accuracy': float(clf_scores.mean()),
            'classification_std': float(clf_scores.std())
        }
    
    def predict_score(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Predict ensemble score
        
        Args:
            X: Feature DataFrame (single row or multiple rows)
            
        Returns:
            Dict with regression_score, classification_prob, ensemble_score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        reg_pred = self.regressor.predict(X)[0]
        clf_prob = self.classifier.predict_proba(X)[0][1] if hasattr(self.classifier, 'classes_') else 0.5
        
        # Weighted ensemble (configurable)
        ensemble_score = 0.6 * reg_pred + 0.4 * clf_prob
        
        return {
            'regression_score': float(reg_pred),
            'classification_prob': float(clf_prob),
            'ensemble_score': float(ensemble_score),
            'is_undervalued': bool(ensemble_score > 0.5)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings"""
        if not self.is_trained:
            return pd.DataFrame()
        
        # Get importances from regressor
        importances = self.regressor.feature_importances_
        
        # Ensure lengths match
        if len(self.feature_names) != len(importances):
            logger.warning(f"Feature name count ({len(self.feature_names)}) != importance count ({len(importances)})")
            # Use indices if mismatch
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, path: str):
        """Save model (joblib for scikit-learn compatibility)"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'regressor': self.regressor,
            'classifier': self.classifier,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }, save_path)
        logger.info(f"RF Ensemble saved: {save_path}")
    
    def load(self, path: str):
        """Load model"""
        data = joblib.load(path)
        self.regressor = data['regressor']
        self.classifier = data['classifier']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        logger.info(f"RF Ensemble loaded: {path}")
