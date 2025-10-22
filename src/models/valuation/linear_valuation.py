"""
Linear Regression model for stock valuation
Based on Phase 2 draft code
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import numpy as np
import pandas as pd
from config.logging_config import get_logger

logger = get_logger(__name__)


class LinearValuationModel:
    """
    Linear regression model for fair value estimation
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.logger = logger
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Train the valuation model
        
        Args:
            X: Features (P/E, debt/equity, growth rate, etc.)
            y: Target (actual stock prices)
            test_size: Proportion of test set
            random_state: Random seed
        
        Returns:
            Dictionary with training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
            
            self.is_trained = True
            
            metrics = {
                'train_r2': train_score,
                'test_r2': test_score,
                'cv_mae': -cv_scores.mean(),
                'cv_mae_std': cv_scores.std()
            }
            
            self.logger.info(f"Model trained - Test R²: {test_score:.4f}, CV MAE: {metrics['cv_mae']:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fair values
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predicted prices
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def calculate_fair_value_margin(self, features: pd.DataFrame, current_price: float) -> dict:
        """
        Calculate how undervalued/overvalued a stock is
        
        Args:
            features: Stock features
            current_price: Current market price
        
        Returns:
            Dictionary with fair value and margin
        """
        fair_value = self.predict(features)[0]
        margin = ((fair_value - current_price) / current_price) * 100
        
        return {
            'fair_value': fair_value,
            'current_price': current_price,
            'margin_pct': margin,
            'is_undervalued': margin > 0,
            'recommendation': self._get_recommendation(margin)
        }
    
    def _get_recommendation(self, margin_pct: float) -> str:
        """Get buy/hold/sell recommendation"""
        if margin_pct > 20:
            return "STRONG BUY"
        elif margin_pct > 10:
            return "BUY"
        elif margin_pct > -10:
            return "HOLD"
        elif margin_pct > -20:
            return "SELL"
        else:
            return "STRONG SELL"
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Sample data (replace with real data)
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'pe_ratio': np.random.uniform(5, 30, n_samples),
        'debt_equity': np.random.uniform(0.1, 2.0, n_samples),
        'revenue_growth': np.random.uniform(-0.1, 0.3, n_samples),
        'beta': np.random.uniform(0.5, 2.0, n_samples)
    })
    
    # Synthetic target (actual prices)
    y = 50 + 2*X['pe_ratio'] - 5*X['debt_equity'] + 100*X['revenue_growth'] + np.random.normal(0, 10, n_samples)
    
    # Train
    model = LinearValuationModel()
    metrics = model.train(X, y)
    print("Training metrics:", metrics)
    
    # Predict
    test_features = pd.DataFrame({
        'pe_ratio': [15],
        'debt_equity': [0.5],
        'revenue_growth': [0.15],
        'beta': [1.2]
    })
    
    result = model.calculate_fair_value_margin(test_features, current_price=100)
    print("\nValuation result:", result)
