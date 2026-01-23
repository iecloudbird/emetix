"""
Initialize analysis package
Comprehensive stock analysis modules for valuation and growth screening

v3.0 Enhancements:
- RedFlagDetector: Beneish M-Score, Altman Z-Score, Piotroski F-Score
- MoatDetector: Recurring revenue, market share momentum, gross margin stability
- PillarScorer: 5-pillar system with Momentum pillar
- AttentionTriggers: 5 triggers (A-E) with solvency guardrails
"""
from .valuation_analyzer import ValuationAnalyzer
from .growth_screener import GrowthScreener
from .stock_screener import StockScreener, screen_top_undervalued, screen_top_undervalued_enhanced
from .quality_growth_gate import QualityGrowthGate, AttentionTriggers
from .pillar_scorer import PillarScorer
from .red_flag_detector import RedFlagDetector
from .moat_detector import MoatDetector

__all__ = [
    'ValuationAnalyzer', 
    'GrowthScreener', 
    'StockScreener', 
    'screen_top_undervalued',
    'screen_top_undervalued_enhanced',  # Backward compatibility alias
    # Phase 3 - Quality Screening Pipeline v3.0
    'QualityGrowthGate',
    'AttentionTriggers',
    'PillarScorer',
    'RedFlagDetector',
    'MoatDetector',
]