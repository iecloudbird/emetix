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
        lstm_predictions: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for RF
        
        Args:
            df: Stock data from YFinanceFetcher (dict or DataFrame)
            lstm_predictions: Optional LSTM fair value predictions
            
        Returns:
            Feature DataFrame
        """
        # Convert dict to DataFrame if needed
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        elif not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected dict or DataFrame, got {type(df)}")
        
        features = pd.DataFrame()
        
        # LSTM features (if available)
        if lstm_predictions:
            features['lstm_fair_value_gap'] = [lstm_predictions.get('fair_value_gap', 0)]
            features['lstm_terminal_value'] = [lstm_predictions.get('terminal_value', 0)]
        else:
            features['lstm_fair_value_gap'] = [0]
            features['lstm_terminal_value'] = [0]
        
        # Helper function to safely extract values
        def safe_get(col_name, default=0):
            val = df.get(col_name, default)
            if isinstance(val, pd.Series):
                return val.iloc[0] if not val.empty else default
            return val
        
        # Valuation metrics
        features['pe_ratio'] = [safe_get('pe_ratio', np.nan)]
        features['pb_ratio'] = [safe_get('price_to_book', np.nan)]
        features['peg_ratio'] = [safe_get('peg_ratio', np.nan)]
        features['ev_ebitda'] = [safe_get('enterprise_to_ebitda', np.nan)]
        
        # Risk metrics
        features['beta'] = [safe_get('beta', 1.0)]
        features['debt_equity'] = [safe_get('debt_to_equity', 0)]
        features['current_ratio'] = [safe_get('current_ratio', 1.0)]
        
        # Profitability
        features['roe'] = [safe_get('return_on_equity', 0)]
        features['revenue_growth'] = [safe_get('revenue_growth', 0)]
        
        # FCF margin (calculate if data available)
        fcf = safe_get('free_cash_flow', 0)
        revenue = safe_get('revenue', 1)
        features['fcf_margin'] = [fcf / (revenue + 1e-6)]  # Avoid division by zero
        
        # Fill NaN with median or default values
        features = features.fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train(
        self, 
        X: pd.DataFrame, 
        y_regression: np.ndarray, 
        y_classification: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train ensemble models
        
        Args:
            X: Feature matrix
            y_regression: Target (e.g., future returns)
            y_classification: Optional binary labels (undervalued=1)
            
        Returns:
            Training metrics
        """
        # Train regressor
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )
        
        self.regressor.fit(X_train, y_train)
        reg_score = self.regressor.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.regressor, X, y_regression, cv=5, scoring='r2'
        )
        
        metrics = {
            'r2_score': float(reg_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        }
        
        # Train classifier if labels provided
        if y_classification is not None:
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X, y_classification, test_size=0.2, random_state=42
            )
            self.classifier.fit(X_train_clf, y_train_clf)
            metrics['classification_accuracy'] = float(self.classifier.score(X_test_clf, y_test_clf))
        
        self.is_trained = True
        logger.info(f"RF Ensemble trained - R²: {reg_score:.4f}, CV: {cv_scores.mean():.4f}")
        
        return metrics
    
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
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.regressor.feature_importances_
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
