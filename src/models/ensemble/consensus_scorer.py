"""
Consensus Scoring System
Weighted voting across LSTM-DCF, RF, and existing models
"""
from typing import Dict
import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


class ConsensusScorer:
    """
    Multi-model consensus scoring
    
    Default Weights:
    - LSTM-DCF: 40%
    - RF Ensemble: 30%
    - Linear Valuation: 20%
    - Risk Classifier: 10%
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        """
        Initialize consensus scorer
        
        Args:
            weights: Custom weights for each model (optional)
        """
        self.weights = weights or {
            'lstm_dcf': 0.40,
            'rf_ensemble': 0.30,
            'linear_valuation': 0.20,
            'risk_classifier': 0.10
        }
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        else:
            logger.warning("Weight sum is zero, using equal weights")
            n_models = len(self.weights)
            self.weights = {k: 1.0/n_models for k in self.weights.keys()}
    
    def calculate_consensus(
        self,
        model_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate weighted consensus score
        
        Args:
            model_scores: Dict with keys matching weight keys
                         Values should be normalized 0-1 scores
            
        Returns:
            Consensus metrics including:
            - consensus_score: Weighted average score
            - confidence: Agreement measure (1 - std deviation)
            - is_undervalued: Boolean decision
            - model_scores: Individual model scores
            - weights_used: Weights applied
        """
        # Filter to only use scores from models we have weights for
        valid_scores = {k: v for k, v in model_scores.items() if k in self.weights}
        
        if not valid_scores:
            logger.warning("No valid model scores provided")
            return {
                'consensus_score': 0.5,
                'confidence': 0.0,
                'is_undervalued': False,
                'model_scores': {},
                'weights_used': self.weights,
                'error': 'No valid scores'
            }
        
        # Calculate weighted sum
        consensus_score = sum(
            self.weights.get(model, 0) * score
            for model, score in valid_scores.items()
        )
        
        # Calculate confidence based on agreement
        # Confidence = 1 - (coefficient of variation)
        scores_list = list(valid_scores.values())
        if len(scores_list) > 1:
            mean_score = np.mean(scores_list)
            std_score = np.std(scores_list)
            # Normalize standard deviation by mean (coefficient of variation)
            cv = std_score / (mean_score + 1e-6)
            confidence = max(0.0, 1.0 - cv)
        else:
            # Only one model, confidence is moderate
            confidence = 0.5
        
        # Final decision: undervalued if consensus > 0.5 and confidence > 0.6
        is_undervalued = consensus_score > 0.5 and confidence > 0.6
        
        return {
            'consensus_score': float(consensus_score),
            'confidence': float(np.clip(confidence, 0, 1)),
            'is_undervalued': is_undervalued,
            'model_scores': valid_scores,
            'weights_used': self.weights,
            'num_models': len(valid_scores)
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights
        
        Args:
            new_weights: New weight dictionary
        """
        self.weights = new_weights
        
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Updated weights: {self.weights}")
    
    def get_model_breakdown(self, model_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Get detailed breakdown of each model's contribution
        
        Args:
            model_scores: Individual model scores
            
        Returns:
            Breakdown showing raw score, weight, and weighted contribution
        """
        breakdown = {}
        
        for model, score in model_scores.items():
            weight = self.weights.get(model, 0)
            contribution = weight * score
            
            breakdown[model] = {
                'raw_score': float(score),
                'weight': float(weight),
                'weighted_contribution': float(contribution),
                'weight_percentage': float(weight * 100)
            }
        
        return breakdown
