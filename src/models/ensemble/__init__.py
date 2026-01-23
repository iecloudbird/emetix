"""
Ensemble Models Package

Architecture (Jan 2025):
- LSTM-DCF: 50% (Deep learning fair value prediction)
- GARP Score: 25% (Transparent Forward P/E + PEG)
- Risk Score: 25% (Beta + volatility filter)
"""
from .consensus_scorer import ConsensusScorer

__all__ = ['ConsensusScorer']
