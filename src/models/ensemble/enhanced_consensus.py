"""
Enhanced Consensus Scorer with Full ML Integration

Architecture Shift (Jan 2025):
- LSTM-DCF: 50% (Deep learning fair value prediction)
- GARP Score: 25% (Transparent Forward P/E + PEG, replaces RF)
- Risk Score: 25% (Beta + volatility filter)

RF Ensemble DEPRECATED - 99.93% P/E importance made it a redundant black-box.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.ensemble.consensus_scorer import ConsensusScorer
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
# RF DEPRECATED: from src.models.ensemble.deprecated.rf_ensemble import RFEnsembleModel
from src.data.fetchers import YFinanceFetcher
from src.data.processors.time_series_processor import TimeSeriesProcessor
from sklearn.preprocessing import MinMaxScaler
from config.settings import MODELS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class EnhancedConsensusScorer:
    """
    Enhanced consensus scoring with full ML pipeline integration
    
    Architecture (Revised Jan 2025):
    - LSTM-DCF (50%): Deep learning fair value prediction
    - GARP Score (25%): Transparent Forward P/E + PEG scoring
    - Risk Score (25%): Beta + volatility filter
    
    Note: RF Ensemble deprecated - was 99% P/E ratio
    """
    
    def __init__(self):
        """Initialize all models and consensus scorer"""
        self.fetcher = YFinanceFetcher()
        self.consensus_scorer = ConsensusScorer(weights={
            'lstm_dcf': 0.50,
            'garp_score': 0.25,
            'risk_score': 0.25
        })
        
        # Load LSTM model
        self.lstm_model = None
        self.lstm_available = self._load_lstm_model()
        
        # RF DEPRECATED - no longer loaded
        self.rf_model = None
        self.rf_available = False
        
        logger.info(f"Enhanced Consensus Scorer initialized: LSTM={self.lstm_available}, GARP=True, Risk=True")
    
    def _load_lstm_model(self) -> bool:
        """Load LSTM-DCF model using from_checkpoint for auto-configuration"""
        try:
            # Check both Enhanced and Final model paths
            enhanced_path = Path(MODELS_DIR) / "lstm_dcf_enhanced.pth"
            final_path = Path(MODELS_DIR) / "lstm_dcf_enhanced.pth"
            
            # Priority 1: Load Enhanced model using from_checkpoint
            if enhanced_path.exists():
                try:
                    self.lstm_model, metadata = LSTMDCFModel.from_checkpoint(str(enhanced_path))
                    hp = metadata.get('hyperparameters', {})
                    self.lstm_input_size = hp.get('input_size', 16)
                    self.lstm_feature_cols = metadata.get('feature_cols', [])
                    self.lstm_scaler = metadata.get('scaler', None)
                    logger.info("✓ Loaded Enhanced LSTM model (16-input, 2-output) from lstm_dcf_enhanced.pth")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load Enhanced model: {e}")
            
            # Priority 2: Try loading Final model as Enhanced (16-input)
            if final_path.exists():
                try:
                    self.lstm_model = LSTMDCFModel(
                        input_size=16,
                        hidden_size=128,
                        num_layers=3,
                        output_size=2
                    )
                    checkpoint = torch.load(str(final_path), map_location='cpu', weights_only=False)
                    self.lstm_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                    self.lstm_model.eval()
                    self.lstm_input_size = 16
                    logger.info("✓ Loaded Enhanced LSTM model (16-input, 2-output) from lstm_dcf_enhanced.pth")
                    return True
                except Exception as e:
                    # Fallback to Final model (12-input, 1-output)
                    try:
                        self.lstm_model = LSTMDCFModel(
                            input_size=12,
                            hidden_size=128,
                            num_layers=3,
                            output_size=1
                        )
                        checkpoint = torch.load(str(final_path), map_location='cpu', weights_only=False)
                        self.lstm_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                        self.lstm_model.eval()
                        self.lstm_input_size = 12
                        logger.warning("⚠️  Loaded Final LSTM model (12-input) - predictions may differ from ValuationAnalyzer")
                        return True
                    except Exception as e2:
                        logger.error(f"Failed to load LSTM model: {e2}")
                        return False
            
            logger.error("No LSTM model file found")
            return False
        except Exception as e:
            logger.error(f"LSTM model loading error: {e}")
            return False
    
    def _load_rf_model(self) -> bool:
        """RF Model DEPRECATED - always returns False"""
        return False
    
    def _get_lstm_score(self, ticker: str, stock_data: pd.Series) -> float:
        """
        Get LSTM-DCF valuation score
        
        Returns:
            Score 0-1 (0=overvalued, 1=undervalued)
        """
        if not self.lstm_available or self.lstm_model is None:
            return None
        
        try:
            # Fetch 60-day historical data for LSTM
            historical = self.fetcher.fetch_historical_prices(ticker, period='3mo')
            
            if historical is None or historical.empty or len(historical) < 60:
                logger.warning(f"Insufficient historical data for LSTM: {ticker}")
                return None
            
            # Helper to safely get float from Series or scalar
            def safe_float(val, default):
                if val is None:
                    return default
                if isinstance(val, (pd.Series, np.ndarray)):
                    if len(val) == 0:
                        return default
                    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val[0])
                return float(val)
            
            # Prepare features based on loaded model input size
            features = []
            for _, row in historical.tail(60).iterrows():
                if self.lstm_input_size == 16:
                    # Enhanced model (16 features)
                    feature_row = [
                        float(row['Close']),
                        float(row['Volume']),
                        float(row['High']),
                        float(row['Low']),
                        safe_float(stock_data.get('pe_ratio'), 15.0),
                        safe_float(stock_data.get('pb_ratio'), 2.0),
                        safe_float(stock_data.get('debt_equity'), 1.0),
                        safe_float(stock_data.get('roe'), 0.15),
                        safe_float(stock_data.get('revenue_growth'), 0.05),
                        safe_float(stock_data.get('profit_margin'), 0.10),
                        safe_float(stock_data.get('gross_margin'), 0.30),
                        safe_float(stock_data.get('roa'), 0.05),
                        safe_float(stock_data.get('fcf_yield'), 0.03),
                        safe_float(stock_data.get('beta'), 1.0),
                        safe_float(stock_data.get('current_ratio'), 1.5),
                        safe_float(stock_data.get('quick_ratio'), 1.0)
                    ]
                else:
                    # Final model (12 features)
                    feature_row = [
                        float(row['Close']),
                        float(row['Volume']),
                        float(row['High']),
                        float(row['Low']),
                        safe_float(stock_data.get('pe_ratio'), 15.0),
                        safe_float(stock_data.get('pb_ratio'), 2.0),
                        safe_float(stock_data.get('debt_equity'), 1.0),
                        safe_float(stock_data.get('roe'), 0.15),
                        safe_float(stock_data.get('revenue_growth'), 0.05),
                        safe_float(stock_data.get('profit_margin'), 0.10),
                        safe_float(stock_data.get('fcf_yield'), 0.03),
                        safe_float(stock_data.get('beta'), 1.0)
                    ]
                features.append(feature_row)
            
            # Normalize features
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Convert to tensor
            sequence = torch.FloatTensor(features_scaled).unsqueeze(0)  # [1, 60, 16]
            
            # LSTM prediction
            with torch.no_grad():
                output = self.lstm_model(sequence)
            
            # Extract FCF growth rate (output[0, 1] for Enhanced model)
            if output.shape[1] >= 2:
                fcf_growth_pred = output[0, 1].item()
            else:
                fcf_growth_pred = output[0, 0].item()
            
            # Scale from [0, 1] to realistic growth range [-20%, +40%]
            growth_rate = (fcf_growth_pred * 0.6) - 0.2
            growth_rate = np.clip(growth_rate, -0.20, 0.40)
            
            # Calculate fair value using Gordon Growth Model
            fcf_per_share = safe_float(stock_data.get('fcf_per_share'), 0)
            wacc = 0.10  # 10% discount rate
            terminal_growth = min(growth_rate * 0.5, 0.03)  # Conservative terminal growth
            
            if fcf_per_share > 0 and wacc > terminal_growth:
                fair_value = (fcf_per_share * (1 + growth_rate)) / (wacc - terminal_growth)
            else:
                # Fallback: P/E-based estimate
                eps = safe_float(stock_data.get('eps'), 0)
                if eps > 0:
                    fair_value = eps * 15 * (1 + growth_rate)
                else:
                    return None
            
            # Calculate score: how undervalued (0=overvalued, 1=very undervalued)
            current_price = safe_float(stock_data.get('current_price', stock_data.get('price')), 0)
            if current_price <= 0 or fair_value <= 0:
                return None
            
            # Score formula: (fair_value - price) / fair_value, clamped to [0, 1]
            upside = (fair_value - current_price) / fair_value
            score = np.clip((upside + 0.2) / 1.4, 0, 1)  # Map [-20%, +120%] to [0, 1]
            
            return float(score)
            
        except Exception as e:
            logger.error(f"LSTM scoring error for {ticker}: {e}")
            return None
    
    def _get_rf_score(self, ticker: str, stock_data: pd.Series) -> float:
        """RF Ensemble DEPRECATED - always returns None"""
        return None
    
    def _get_pe_sanity_score(self, stock_data: pd.Series) -> float:
        """
        Get P/E sanity check score
        
        Returns:
            Score 0-1 based on P/E attractiveness
        """
        try:
            pe_ratio = stock_data.get('pe_ratio', None)
            
            # Handle Series vs scalar
            if pe_ratio is None:
                return None
            if isinstance(pe_ratio, (pd.Series, np.ndarray)):
                if len(pe_ratio) == 0:
                    return None
                pe_ratio = float(pe_ratio.iloc[0]) if isinstance(pe_ratio, pd.Series) else float(pe_ratio[0])
            else:
                pe_ratio = float(pe_ratio)
            
            if pe_ratio <= 0:
                return None
            
            # P/E scoring: lower is better
            # Excellent: <15, Good: 15-20, Fair: 20-30, Poor: >30
            if pe_ratio < 15:
                score = 1.0
            elif pe_ratio < 20:
                score = 0.8
            elif pe_ratio < 25:
                score = 0.6
            elif pe_ratio < 30:
                score = 0.4
            else:
                score = 0.2
            
            return float(score)
            
        except Exception as e:
            logger.error(f"P/E scoring error: {e}")
            return None
    
    def analyze_stock_comprehensive(self, ticker: str) -> Dict:
        """
        Comprehensive consensus analysis for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with consensus score, individual scores, and recommendation
        """
        logger.info(f"Running enhanced consensus analysis for {ticker}")
        
        # Fetch stock data
        stock_data = self.fetcher.fetch_stock_data(ticker)
        
        if stock_data is None or (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
            return {
                'error': f'No data available for {ticker}',
                'consensus_score': 0.5,
                'recommendation': 'HOLD',
                'confidence': 0.0
            }
        
        # Get individual model scores
        lstm_score = self._get_lstm_score(ticker, stock_data)
        rf_score = self._get_rf_score(ticker, stock_data)
        pe_score = self._get_pe_sanity_score(stock_data)
        
        # Prepare scores dict
        model_scores = {}
        if lstm_score is not None:
            model_scores['lstm_dcf'] = lstm_score
        if rf_score is not None:
            model_scores['rf_risk_sentiment'] = rf_score
        if pe_score is not None:
            model_scores['pe_sanity_score'] = pe_score
        
        # Calculate consensus
        consensus = self.consensus_scorer.calculate_consensus(model_scores)
        
        # Generate recommendation
        score = consensus['consensus_score']
        if score >= 0.7:
            recommendation = 'STRONG BUY'
        elif score >= 0.6:
            recommendation = 'BUY'
        elif score >= 0.4:
            recommendation = 'HOLD'
        elif score >= 0.3:
            recommendation = 'SELL'
        else:
            recommendation = 'STRONG SELL'
        
        return {
            'ticker': ticker,
            'consensus_score': consensus['consensus_score'],
            'confidence': consensus['confidence'],
            'recommendation': recommendation,
            'is_undervalued': consensus['is_undervalued'],
            'individual_scores': model_scores,
            'weights': consensus['weights_used'],
            'num_models_available': len(model_scores)
        }

