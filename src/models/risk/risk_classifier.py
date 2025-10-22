"""
Random Forest classifier for risk assessment
Based on Phase 2 draft code
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
from config.logging_config import get_logger

logger = get_logger(__name__)


class RiskClassifier:
    """
    Random Forest model for classifying stocks into risk categories
    Labels: 0 = Low Risk, 1 = Medium Risk, 2 = High Risk
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.logger = logger
        self.is_trained = False
        self.feature_importance = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train the risk classification model
        
        Args:
            X: Features (beta, volatility, debt/equity, etc.)
            y: Risk labels (0=Low, 1=Medium, 2=High)
            test_size: Proportion of test set
        
        Returns:
            Dictionary with training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Train
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.is_trained = True
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            self.logger.info(f"Risk model trained - Test Accuracy: {test_score:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training risk model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk category"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability for each risk class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.model.predict_proba(X)
    
    def assess_risk(self, features: pd.DataFrame) -> dict:
        """
        Comprehensive risk assessment
        
        Args:
            features: Stock features DataFrame
        
        Returns:
            Dictionary with risk assessment
        """
        risk_label = self.predict(features)[0]
        risk_proba = self.predict_proba(features)[0]
        
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
        
        return {
            'risk_category': risk_names[risk_label],
            'risk_score': risk_label,
            'confidence': max(risk_proba) * 100,
            'probabilities': {
                'low_risk': risk_proba[0],
                'medium_risk': risk_proba[1],
                'high_risk': risk_proba[2]
            },
            'recommendation': self._get_risk_recommendation(risk_label, max(risk_proba))
        }
    
    def _get_risk_recommendation(self, risk_label: int, confidence: float) -> str:
        """Get investment recommendation based on risk"""
        if risk_label == 0:  # Low risk
            return "SUITABLE FOR CONSERVATIVE INVESTORS"
        elif risk_label == 1:  # Medium risk
            return "MODERATE RISK - DIVERSIFY PORTFOLIO"
        else:  # High risk
            return "HIGH RISK - CAUTION ADVISED"
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings"""
        if self.feature_importance is None:
            return None
        return self.feature_importance
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Risk model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"Risk model loaded from {filepath}")


def label_risk_from_beta(beta: float) -> int:
    """
    Helper function to label risk based on beta
    Beta < 0.8: Low Risk (0)
    0.8 <= Beta <= 1.2: Medium Risk (1)
    Beta > 1.2: High Risk (2)
    """
    if beta < 0.8:
        return 0
    elif beta <= 1.2:
        return 1
    else:
        return 2


# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'beta': np.random.uniform(0.3, 2.5, n_samples),
        'volatility': np.random.uniform(10, 80, n_samples),
        'debt_equity': np.random.uniform(0.1, 3.0, n_samples),
        'current_ratio': np.random.uniform(0.5, 3.0, n_samples)
    })
    
    # Generate labels based on beta
    y = X['beta'].apply(label_risk_from_beta)
    
    # Train
    risk_model = RiskClassifier()
    metrics = risk_model.train(X, y)
    print("Training metrics:", metrics)
    
    # Feature importance
    print("\nFeature Importance:")
    print(risk_model.get_feature_importance())
    
    # Predict
    test_stock = pd.DataFrame({
        'beta': [0.7],
        'volatility': [25],
        'debt_equity': [0.5],
        'current_ratio': [2.0]
    })
    
    result = risk_model.assess_risk(test_stock)
    print("\nRisk Assessment:")
    print(result)
