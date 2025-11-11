"""
Enhanced Consensus Scoring System
Integrates LSTM-DCF + RF Risk+Sentiment + P/E Sanity Check

Weighting:
- LSTM-DCF: 70% (Truth: Long-term fair value)
- RF Risk+Sentiment: 20% (Brake: Short-term risk + market mood)
- P/E Sanity Check: 10% (Anchor: Market reality check)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional
from config.logging_config import get_logger
from config.settings import MODELS_DIR

# Model imports
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.models.ensemble.consensus_scorer import ConsensusScorer
from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.fetchers.technical_sentiment_fetcher import TechnicalSentimentFetcher
from src.data.processors.time_series_processor import TimeSeriesProcessor

logger = get_logger(__name__)


class EnhancedConsensusScorer:
    """
    Enhanced consensus scorer with 70-20-10 weighting:
    - LSTM-DCF: 70% (long-term fair value)
    - RF Risk+Sentiment: 20% (short-term risk brake)
    - P/E Sanity: 10% (market reality anchor)
    """
    
    def __init__(self):
        """Initialize all models and components"""
        self.logger = get_logger(__name__)
        
        # Initialize fetchers
        self.yf_fetcher = YFinanceFetcher()
        self.tech_sentiment_fetcher = TechnicalSentimentFetcher()
        self.ts_processor = TimeSeriesProcessor()
        
        # Initialize models (will be loaded on first use)
        self.lstm_model = None
        self.rf_model = None
        
        # Consensus weights (70-20-10)
        self.weights = {
            'lstm_dcf': 0.70,           # Truth: Fair value
            'rf_risk_sentiment': 0.20,  # Brake: Risk + sentiment
            'pe_sanity_score': 0.10     # Anchor: Market reality
        }
        
        self.consensus_scorer = ConsensusScorer(weights=self.weights)
    
    def load_models(self):
        """Load trained models on demand"""
        if self.lstm_model is None:
            try:
                self.lstm_model = LSTMDCFModel(input_size=16, hidden_size=128, num_layers=3)
                # Try different model configurations to match saved checkpoints
                enhanced_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
                final_path = MODELS_DIR / "lstm_dcf_final.pth"
                
                # First try the current 16-input, 2-output model with enhanced checkpoint
                if enhanced_path.exists():
                    try:
                        # Load with weights_only=False for compatibility
                        import torch
                        checkpoint = torch.load(str(enhanced_path), map_location='cpu', weights_only=False)
                        self.lstm_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                        self.lstm_model.eval()
                        self.logger.info("✅ LSTM-DCF enhanced model loaded (16 inputs)")
                    except Exception as e:
                        self.logger.warning(f"Enhanced model failed: {e}")
                        self.lstm_model = None
                
                # If enhanced failed, try 12-input, 1-output model with final checkpoint  
                if self.lstm_model is None and final_path.exists():
                    try:
                        # Create model matching the saved checkpoint architecture
                        self.lstm_model = LSTMDCFModel(input_size=12, hidden_size=128, num_layers=3, output_size=1)
                        import torch
                        checkpoint = torch.load(str(final_path), map_location='cpu', weights_only=False)
                        self.lstm_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                        self.lstm_model.eval()
                        self.logger.info("✅ LSTM-DCF final model loaded (12 inputs, 1 output)")
                    except Exception as e:
                        self.logger.warning(f"Final model failed: {e}")
                        self.lstm_model = None
                
                if self.lstm_model is None:
                    self.logger.warning("❌ No LSTM model could be loaded")
                    self.lstm_model = None
            except Exception as e:
                self.logger.error(f"❌ Error loading LSTM model: {e}")
                self.lstm_model = None
        
        if self.rf_model is None:
            try:
                self.rf_model = RFEnsembleModel()
                rf_path = MODELS_DIR / "rf_ensemble.pkl"
                if rf_path.exists():
                    self.rf_model.load(str(rf_path))
                    self.logger.info("✅ RF Risk+Sentiment model loaded")
                else:
                    self.logger.warning(f"❌ RF model not found: {rf_path}")
                    self.rf_model = None
            except Exception as e:
                self.logger.error(f"❌ Error loading RF model: {e}")
                self.rf_model = None
    
    def get_lstm_dcf_score(self, ticker: str) -> Optional[float]:
        """
        Get LSTM-DCF valuation score (0-1 normalized)
        For now, use a simplified fundamental-based approach until LSTM integration is complete
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Normalized score 0-1 (1 = undervalued, 0 = overvalued)
        """
        # Simplified LSTM scoring based on fundamentals for now
        # This will be replaced with actual LSTM predictions once model loading is fixed
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get key metrics
            pe_ratio = info.get('forwardPE', info.get('trailingPE', 20))
            pb_ratio = info.get('priceToBook', 2)
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            
            # Simplified DCF-like scoring
            # Good value: Low P/E, Low P/B, High growth, High ROE
            scores = []
            
            # P/E component (inverse - lower is better)
            if pe_ratio > 0:
                pe_score = np.clip(1 - (pe_ratio - 10) / 30, 0, 1)  # Ideal P/E around 10-15
                scores.append(pe_score)
            
            # P/B component (inverse - lower is better up to a point)
            if pb_ratio > 0:
                pb_score = np.clip(1 - (pb_ratio - 1) / 4, 0, 1)  # Ideal P/B around 1-2
                scores.append(pb_score)
            
            # Growth component (higher is better)
            growth_score = np.clip(revenue_growth / 20, 0, 1)  # 20% growth = max score
            scores.append(growth_score)
            
            # ROE component (higher is better)
            roe_score = np.clip(roe / 30, 0, 1)  # 30% ROE = max score
            scores.append(roe_score)
            
            if scores:
                # Average the scores
                final_score = np.mean(scores)
                return float(final_score)
            
            return 0.5  # Neutral if no data
            
        except Exception as e:
            self.logger.debug(f"Error in LSTM-DCF scoring for {ticker}: {e}")
            return 0.5
    
    def get_rf_risk_sentiment_score(self, ticker: str) -> Optional[float]:
        """
        Get RF Risk+Sentiment score (0-1 normalized)
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Normalized score 0-1 (1 = low risk + positive sentiment, 0 = high risk + negative sentiment)
        """
        if self.rf_model is None:
            return None
        
        try:
            # Get enhanced technical/sentiment features
            enhanced_features = self.tech_sentiment_fetcher.fetch_enhanced_features(ticker)
            if enhanced_features is None:
                return None
            
            # Get basic stock data for RF model
            stock_data = self.yf_fetcher.fetch_stock_data(ticker)
            if stock_data is None:
                return None
            
            # Prepare features
            features_df = self.rf_model.prepare_features(
                stock_data, 
                lstm_predictions=None, 
                enhanced_features=enhanced_features
            )
            
            # Get RF prediction
            prediction = self.rf_model.predict_score(features_df)
            
            # Extract risk and sentiment signals
            predicted_return = prediction['regression_score']  # % return
            outperform_prob = prediction['classification_prob']  # 0-1 probability
            
            # Convert to risk score (normalize predicted return to 0-1)
            # Assume returns from -50% to +100% map to 0-1 risk score
            return_score = np.clip((predicted_return + 50) / 150, 0, 1)
            
            # Combine return prediction and probability
            # 60% return prediction, 40% outperform probability
            rf_score = 0.6 * return_score + 0.4 * outperform_prob
            
            return float(np.clip(rf_score, 0, 1))
            
        except Exception as e:
            self.logger.debug(f"Error in RF risk+sentiment scoring for {ticker}: {e}")
            return None
    
    def get_pe_sanity_score(self, ticker: str) -> Optional[float]:
        """
        Get P/E sanity check score (0-1 normalized)
        Simple market reality anchor based on P/E relative to sector
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Normalized score 0-1 (1 = reasonable P/E, 0 = extreme P/E)
        """
        try:
            # Get basic stock info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
            if not pe_ratio or pe_ratio <= 0:
                return 0.5  # Neutral if no P/E
            
            # Simple P/E sanity check
            # Reasonable P/E range: 10-25 for most stocks
            if pe_ratio < 5:
                return 0.2  # Too low, might be distressed
            elif pe_ratio <= 15:
                return 0.8  # Good value
            elif pe_ratio <= 25:
                return 0.6  # Reasonable
            elif pe_ratio <= 40:
                return 0.4  # Expensive
            else:
                return 0.2  # Very expensive
                
        except Exception as e:
            self.logger.debug(f"Error in P/E sanity check for {ticker}: {e}")
            return 0.5  # Neutral default
    
    def get_consensus_score(self, ticker: str) -> Dict:
        """
        Get comprehensive consensus score for ticker
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with consensus results and breakdown
        """
        self.load_models()  # Load models if not already loaded
        
        # Get individual model scores
        lstm_score = self.get_lstm_dcf_score(ticker)
        rf_score = self.get_rf_risk_sentiment_score(ticker)
        pe_score = self.get_pe_sanity_score(ticker)
        
        # Prepare model scores dict
        model_scores = {}
        if lstm_score is not None:
            model_scores['lstm_dcf'] = lstm_score
        if rf_score is not None:
            model_scores['rf_risk_sentiment'] = rf_score
        if pe_score is not None:
            model_scores['pe_sanity_score'] = pe_score
        
        # Calculate consensus
        consensus_result = self.consensus_scorer.calculate_consensus(model_scores)
        
        # Add individual scores for debugging
        consensus_result['individual_scores'] = {
            'lstm_dcf': lstm_score,
            'rf_risk_sentiment': rf_score,
            'pe_sanity_score': pe_score
        }
        
        # Add weighting info
        consensus_result['weighting_scheme'] = self.weights
        
        return consensus_result
    
    def analyze_stock_comprehensive(self, ticker: str) -> Dict:
        """
        Comprehensive stock analysis with consensus scoring
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Complete analysis dict
        """
        # Get current price
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        except:
            current_price = 0
        
        # Get consensus score
        consensus = self.get_consensus_score(ticker)
        
        # Generate recommendation
        score = consensus['consensus_score']
        confidence = consensus['confidence']
        
        if score > 0.7 and confidence > 0.7:
            recommendation = "🚀 STRONG BUY"
        elif score > 0.6 and confidence > 0.6:
            recommendation = "✅ BUY"
        elif score > 0.4:
            recommendation = "⚖️ HOLD"
        elif score > 0.3:
            recommendation = "⚠️ SELL"
        else:
            recommendation = "🔴 STRONG SELL"
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'consensus_score': score,
            'confidence': confidence,
            'recommendation': recommendation,
            'individual_scores': consensus['individual_scores'],
            'weighting': self.weights,
            'model_breakdown': self.consensus_scorer.get_model_breakdown(consensus['model_scores']),
            'is_undervalued': consensus['is_undervalued']
        }


def test_enhanced_consensus():
    """Test the enhanced consensus scorer"""
    scorer = EnhancedConsensusScorer()
    
    test_tickers = ['AAPL', 'TSLA', 'JNJ', 'AMD', 'MSFT']
    
    print("=" * 100)
    print("🎯 ENHANCED CONSENSUS SCORING TEST (70% LSTM + 20% RF + 10% P/E)")
    print("=" * 100)
    
    for ticker in test_tickers:
        print(f"\n📊 {ticker} Analysis:")
        print("-" * 50)
        
        try:
            analysis = scorer.analyze_stock_comprehensive(ticker)
            
            print(f"Current Price: ${analysis['current_price']:.2f}")
            print(f"Consensus Score: {analysis['consensus_score']:.3f} (Confidence: {analysis['confidence']:.3f})")
            print(f"Recommendation: {analysis['recommendation']}")
            print(f"Undervalued: {analysis['is_undervalued']}")
            
            print("\nModel Breakdown:")
            individual = analysis['individual_scores']
            for model, score in individual.items():
                weight = analysis['weighting'][model]
                if score is not None:
                    contribution = weight * score
                    print(f"  {model:20s}: {score:.3f} (weight: {weight:.0%}) = {contribution:.3f}")
                else:
                    print(f"  {model:20s}: N/A")
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")


if __name__ == "__main__":
    # Add torch import here since it's needed for LSTM
    try:
        import torch
    except ImportError:
        print("PyTorch not available, LSTM scoring will be disabled")
    
    test_enhanced_consensus()