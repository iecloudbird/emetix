"""
Pillar Scorer - 5-Pillar Composite Scoring System (v3.0)

Enhanced scoring system with 5 pillars:
1. VALUE (20%): Margin of Safety, P/E vs Sector, FCF Yield
2. QUALITY (20%): FCF ROIC, Profit Margin, ROE, Debt Health
3. GROWTH (20%): Revenue Growth, Earnings Growth, LSTM Forecast
4. SAFETY (20%): Beta, Volatility, Drawdown Risk
5. MOMENTUM (20%): RSI, MA Crossovers, Market Share Momentum (NEW - TLI alignment)

Enhancements in v3.0:
- Added Pillar 5: Momentum for technical timing (TLI alignment)
- Watch sub-categories: High-Quality/Expensive, Cheap/Junk, Needs Research
- Red flag integration for vetoing
- Sector-normalized Z-scores for fair comparison
- LSTM FCF penalty for declining forecasts

Each pillar is 0-100, weighted average = composite score.
"""
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PillarScore:
    """Individual pillar score with component breakdown"""
    name: str
    score: float  # 0-100
    weight: float  # 0.20 each for 5 pillars
    components: Dict[str, float] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


class PillarScorer:
    """
    5-Pillar Composite Scoring System (v3.0)
    
    Scoring Philosophy:
    - Each pillar normalized to 0-100 scale
    - Equal weights (20% each) for balanced assessment
    - Minimum composite score of 50 required to qualify
    - Individual pillar scores help identify strengths/weaknesses
    - Watch sub-categories for nuanced classification
    """
    
    # Pillar weights (must sum to 1.0) - v3.1: Value-focused weighting
    # Prioritize Value + Quality for undervalued quality picks (TLI/GDS alignment)
    PILLAR_WEIGHTS = {
        "value": 0.25,      # Up from 0.20 - core undervalued focus
        "quality": 0.25,    # Up from 0.20 - compounder quality focus
        "growth": 0.20,     # Unchanged
        "safety": 0.15,     # Down from 0.20 - less weight on pure safety
        "momentum": 0.15,   # Down from 0.20 - technical timing less important
    }
    
    # v3.1: TIGHTENED Score thresholds for classification
    MIN_QUALIFIED_SCORE = 60   # Was 50 - raise floor for quality
    BUY_THRESHOLD = 70         # Was 65 - only high-conviction buys
    HOLD_THRESHOLD = 60        # Was 55 - quality holds only
    
    # v3.1: Pillar floor veto - reject if any core pillar too weak
    PILLAR_FLOOR = 40          # Minimum score for any pillar
    CORE_PILLARS = ["value", "quality", "safety"]  # Must meet floor
    
    # Watch sub-category thresholds
    HIGH_QUALITY_EXPENSIVE_THRESHOLD = 70  # Quality/Growth >70 but MoS <0%
    CHEAP_JUNK_THRESHOLD = 70  # Value >70 but Quality/Safety <50%
    
    def __init__(self):
        self.logger = logger
    
    def calculate_composite(self, stock_data: Dict, include_momentum: bool = True) -> Dict:
        """
        Calculate composite score from 5 pillars (v3.0).
        
        Args:
            stock_data: Dictionary with stock metrics
            include_momentum: Include Pillar 5 (Momentum) in scoring
            
        Returns:
            {
                "composite_score": 72.5,
                "qualified": True,
                "classification": "Buy",
                "watch_sub_category": null,  # For Watch: "high_quality_expensive", "cheap_junk", "needs_research"
                "pillars": {
                    "value": {"score": 80, "components": {...}},
                    "quality": {"score": 70, "components": {...}},
                    "growth": {"score": 65, "components": {...}},
                    "safety": {"score": 75, "components": {...}},
                    "momentum": {"score": 60, "components": {...}}
                }
            }
        """
        # Calculate each pillar
        value_pillar = self._calculate_value_pillar(stock_data)
        quality_pillar = self._calculate_quality_pillar(stock_data)
        growth_pillar = self._calculate_growth_pillar(stock_data)
        safety_pillar = self._calculate_safety_pillar(stock_data)
        
        # Pillar 5: Momentum (optional, for v3.0)
        if include_momentum:
            momentum_pillar = self._calculate_momentum_pillar(stock_data)
            
            # Calculate weighted composite (5 pillars at 20% each)
            composite_score = (
                value_pillar.weighted_score +
                quality_pillar.weighted_score +
                growth_pillar.weighted_score +
                safety_pillar.weighted_score +
                momentum_pillar.weighted_score
            )
        else:
            # Fallback to 4-pillar (25% each) for backward compatibility
            momentum_pillar = None
            composite_score = (
                value_pillar.score * 0.25 +
                quality_pillar.score * 0.25 +
                growth_pillar.score * 0.25 +
                safety_pillar.score * 0.25
            )
        
        # Round to 1 decimal
        composite_score = round(composite_score, 1)
        
        # Extract key metrics for classification
        margin_of_safety = stock_data.get("margin_of_safety", 0) or 0
        
        # v3.1: Pillar floor veto check
        pillar_scores = {
            "value": value_pillar.score,
            "quality": quality_pillar.score,
            "safety": safety_pillar.score,
        }
        pillar_floor_failed = False
        failed_pillars = []
        for pillar_name in self.CORE_PILLARS:
            if pillar_scores.get(pillar_name, 0) < self.PILLAR_FLOOR:
                pillar_floor_failed = True
                failed_pillars.append(pillar_name)
        
        # Determine qualification (v3.1: stricter with pillar floor)
        qualified = composite_score >= self.MIN_QUALIFIED_SCORE and not pillar_floor_failed
        
        # Enhanced classification with MoS requirement (v3.1: raised thresholds)
        # Buy requires: MoS ≥25%, Score ≥70, no pillar floor failures
        if margin_of_safety >= 25 and composite_score >= self.BUY_THRESHOLD and not pillar_floor_failed:
            classification = "Buy"
        # Hold requires: MoS -5% to 25%, Score ≥60, no pillar floor failures  
        elif -5 <= margin_of_safety <= 25 and composite_score >= self.HOLD_THRESHOLD and not pillar_floor_failed:
            classification = "Hold"
        else:
            classification = "Watch"
        
        # Determine Watch sub-category (v3.0 enhancement)
        watch_sub_category = None
        if classification == "Watch":
            watch_sub_category = self._determine_watch_subcategory(
                value_score=value_pillar.score,
                quality_score=quality_pillar.score,
                growth_score=growth_pillar.score,
                safety_score=safety_pillar.score,
                margin_of_safety=margin_of_safety
            )
        
        # Build pillars dict
        pillars = {
            "value": {
                "score": value_pillar.score,
                "weight": value_pillar.weight,
                "weighted_score": value_pillar.weighted_score,
                "components": value_pillar.components,
            },
            "quality": {
                "score": quality_pillar.score,
                "weight": quality_pillar.weight,
                "weighted_score": quality_pillar.weighted_score,
                "components": quality_pillar.components,
            },
            "growth": {
                "score": growth_pillar.score,
                "weight": growth_pillar.weight,
                "weighted_score": growth_pillar.weighted_score,
                "components": growth_pillar.components,
            },
            "safety": {
                "score": safety_pillar.score,
                "weight": safety_pillar.weight,
                "weighted_score": safety_pillar.weighted_score,
                "components": safety_pillar.components,
            },
        }
        
        # Add momentum pillar if calculated
        if momentum_pillar:
            pillars["momentum"] = {
                "score": momentum_pillar.score,
                "weight": momentum_pillar.weight,
                "weighted_score": momentum_pillar.weighted_score,
                "components": momentum_pillar.components,
            }
        
        return {
            "composite_score": composite_score,
            "qualified": qualified,
            "classification": classification,
            "watch_sub_category": watch_sub_category,
            "margin_of_safety": margin_of_safety,
            "pillar_floor_check": {
                "passed": not pillar_floor_failed,
                "floor": self.PILLAR_FLOOR,
                "failed_pillars": failed_pillars
            },
            "pillars": pillars,
        }
    
    def _calculate_value_pillar(self, data: Dict) -> PillarScore:
        """
        VALUE Pillar (25%)
        
        Components:
        - Margin of Safety (40%): LSTM-DCF based, higher is better
        - P/E vs Sector (30%): Below sector average is better
        - FCF Yield (30%): Higher is better
        """
        components = {}
        
        # Margin of Safety (0-100 scale)
        # 0% MoS = 0 score, 50%+ MoS = 100 score
        mos = data.get('margin_of_safety', 0) or 0
        mos_score = min(100, max(0, mos * 2))  # 50% MoS = 100 score
        components['margin_of_safety'] = {
            'value': mos,
            'score': round(mos_score, 1),
            'weight': 0.40
        }
        
        # P/E vs Sector (lower is better)
        # Use forward PE if available, else trailing
        pe = data.get('forward_pe') or data.get('pe_ratio') or 0
        sector_pe = data.get('sector_pe', 20) or 20  # Default sector PE
        
        if pe > 0 and sector_pe > 0:
            pe_ratio_to_sector = pe / sector_pe
            # Score: < 0.5 = 100, 0.5-0.75 = 80, 0.75-1.0 = 60, 1.0-1.25 = 40, > 1.25 = 20
            if pe_ratio_to_sector < 0.5:
                pe_score = 100
            elif pe_ratio_to_sector < 0.75:
                pe_score = 80
            elif pe_ratio_to_sector < 1.0:
                pe_score = 60
            elif pe_ratio_to_sector < 1.25:
                pe_score = 40
            else:
                pe_score = 20
        else:
            pe_score = 50  # Neutral if no data
            pe_ratio_to_sector = None
        
        components['pe_vs_sector'] = {
            'value': round(pe_ratio_to_sector, 2) if pe_ratio_to_sector else None,
            'pe': pe,
            'sector_pe': sector_pe,
            'score': pe_score,
            'weight': 0.30
        }
        
        # FCF Yield (higher is better)
        # 0% = 0 score, 5% = 50 score, 10%+ = 100 score
        fcf_yield = data.get('fcf_yield', 0) or 0
        fcf_yield_score = min(100, max(0, fcf_yield * 10))
        components['fcf_yield'] = {
            'value': fcf_yield,
            'score': round(fcf_yield_score, 1),
            'weight': 0.30
        }
        
        # Calculate weighted pillar score
        pillar_score = (
            mos_score * 0.40 +
            pe_score * 0.30 +
            fcf_yield_score * 0.30
        )
        
        return PillarScore(
            name="Value",
            score=round(pillar_score, 1),
            weight=self.PILLAR_WEIGHTS["value"],
            components=components
        )
    
    def _calculate_quality_pillar(self, data: Dict) -> PillarScore:
        """
        QUALITY Pillar (25%)
        
        Components:
        - FCF ROIC (35%): Capital efficiency, higher is better
        - Profit Margin (25%): Business quality, higher is better
        - ROE (20%): Return on equity, higher is better
        - Debt Health (20%): D/E ratio, lower is better
        """
        components = {}
        
        # FCF ROIC (0-100)
        # 0% = 0, 15% = 75, 25%+ = 100
        fcf_roic = data.get('fcf_roic', 0) or 0
        if fcf_roic <= 0:
            fcf_roic_score = 0
        elif fcf_roic >= 25:
            fcf_roic_score = 100
        else:
            fcf_roic_score = min(100, fcf_roic * 4)
        components['fcf_roic'] = {
            'value': fcf_roic,
            'score': round(fcf_roic_score, 1),
            'weight': 0.35
        }
        
        # Profit Margin (0-100)
        # 0% = 0, 10% = 50, 20%+ = 100
        # stock_screener already returns as percentage, no need to multiply by 100
        profit_margin = data.get('profit_margin', 0) or 0
        if profit_margin <= 0:
            margin_score = 0
        elif profit_margin >= 20:
            margin_score = 100
        else:
            margin_score = min(100, profit_margin * 5)
        components['profit_margin'] = {
            'value': round(profit_margin, 1),
            'score': round(margin_score, 1),
            'weight': 0.25
        }
        
        # ROE (0-100)
        # < 0% = 0, 15% = 75, 25%+ = 100
        # stock_screener already returns as percentage
        roe = data.get('roe', 0) or 0
        if roe <= 0:
            roe_score = 0
        elif roe >= 25:
            roe_score = 100
        else:
            roe_score = min(100, roe * 4)
        components['roe'] = {
            'value': round(roe, 1),
            'score': round(roe_score, 1),
            'weight': 0.20
        }
        
        # Debt Health (inverted - lower D/E is better)
        # D/E > 2 = 0, D/E 1-2 = 50, D/E 0.5-1 = 75, D/E < 0.5 = 100
        debt_equity = data.get('debt_equity', 0) or 0
        if debt_equity > 2:
            debt_score = 0
        elif debt_equity > 1:
            debt_score = 50 - (debt_equity - 1) * 50  # Linear 50 to 0
        elif debt_equity > 0.5:
            debt_score = 75 - (debt_equity - 0.5) * 50  # Linear 75 to 50
        elif debt_equity >= 0:
            debt_score = 100 - debt_equity * 50  # Linear 100 to 75
        else:
            debt_score = 50  # Negative D/E is unusual
        components['debt_health'] = {
            'value': debt_equity,
            'score': round(max(0, debt_score), 1),
            'weight': 0.20
        }
        
        # Calculate weighted pillar score
        pillar_score = (
            fcf_roic_score * 0.35 +
            margin_score * 0.25 +
            roe_score * 0.20 +
            max(0, debt_score) * 0.20
        )
        
        return PillarScore(
            name="Quality",
            score=round(pillar_score, 1),
            weight=self.PILLAR_WEIGHTS["quality"],
            components=components
        )
    
    def _calculate_growth_pillar(self, data: Dict) -> PillarScore:
        """
        GROWTH Pillar (25%)
        
        Components:
        - Revenue Growth (40%): Next year estimate or TTM
        - Earnings Growth (30%): Next year estimate or TTM
        - LSTM Forecast (30%): ML-predicted growth
        """
        components = {}
        
        # Revenue Growth (0-100)
        # -10% = 0, 0% = 40, 10% = 60, 25%+ = 100
        revenue_growth = data.get('next_year_revenue_growth') or data.get('revenue_growth', 0) or 0
        if revenue_growth <= -10:
            rev_score = 0
        elif revenue_growth <= 0:
            rev_score = 40 + (revenue_growth + 10) * 4  # -10% to 0% = 0 to 40
        elif revenue_growth <= 25:
            rev_score = 40 + (revenue_growth * 2.4)  # 0% to 25% = 40 to 100
        else:
            rev_score = 100
        components['revenue_growth'] = {
            'value': revenue_growth,
            'score': round(rev_score, 1),
            'weight': 0.40
        }
        
        # Earnings Growth (0-100)
        # Similar scale to revenue
        earnings_growth = data.get('earnings_growth', 0) or 0
        if earnings_growth <= -10:
            earn_score = 0
        elif earnings_growth <= 0:
            earn_score = 40 + (earnings_growth + 10) * 4
        elif earnings_growth <= 25:
            earn_score = 40 + (earnings_growth * 2.4)
        else:
            earn_score = 100
        components['earnings_growth'] = {
            'value': earnings_growth,
            'score': round(earn_score, 1),
            'weight': 0.30
        }
        
        # LSTM Predicted Growth (0-100)
        # Uses model's growth forecast
        lstm_growth = data.get('lstm_predicted_growth', 0) or 0
        if lstm_growth <= -10:
            lstm_score = 0
        elif lstm_growth <= 0:
            lstm_score = 40 + (lstm_growth + 10) * 4
        elif lstm_growth <= 25:
            lstm_score = 40 + (lstm_growth * 2.4)
        else:
            lstm_score = 100
        components['lstm_growth'] = {
            'value': lstm_growth,
            'score': round(lstm_score, 1),
            'weight': 0.30
        }
        
        # Calculate weighted pillar score
        pillar_score = (
            rev_score * 0.40 +
            earn_score * 0.30 +
            lstm_score * 0.30
        )
        
        return PillarScore(
            name="Growth",
            score=round(pillar_score, 1),
            weight=self.PILLAR_WEIGHTS["growth"],
            components=components
        )
    
    def _calculate_safety_pillar(self, data: Dict) -> PillarScore:
        """
        SAFETY Pillar (25%)
        
        Components:
        - Beta (40%): Lower beta is safer
        - Volatility (35%): Lower volatility is safer
        - Drawdown Risk (25%): Distance from 52W high (already down = more risk absorbed)
        """
        components = {}
        
        # Beta (0-100, inverted)
        # < 0.6 = 100, 0.6-0.8 = 85, 0.8-1.0 = 70, 1.0-1.2 = 55, 1.2-1.5 = 40, > 1.5 = 20
        beta = data.get('beta', 1.0) or 1.0
        if beta < 0.6:
            beta_score = 100
        elif beta < 0.8:
            beta_score = 85
        elif beta < 1.0:
            beta_score = 70
        elif beta < 1.2:
            beta_score = 55
        elif beta < 1.5:
            beta_score = 40
        else:
            beta_score = 20
        components['beta'] = {
            'value': beta,
            'score': beta_score,
            'weight': 0.40
        }
        
        # Volatility (0-100, inverted)
        # < 15% = 100, 15-25% = 70, 25-35% = 50, 35-50% = 30, > 50% = 10
        volatility = data.get('volatility', 25) or 25
        if volatility < 15:
            vol_score = 100
        elif volatility < 25:
            vol_score = 70
        elif volatility < 35:
            vol_score = 50
        elif volatility < 50:
            vol_score = 30
        else:
            vol_score = 10
        components['volatility'] = {
            'value': volatility,
            'score': vol_score,
            'weight': 0.35
        }
        
        # Drawdown risk (higher drawdown = more risk already absorbed)
        # If stock already down 50%, it has less downside from high
        # This is a nuanced metric - being down a lot can be good (value) or bad (dying)
        # We combine with FCF check elsewhere
        pct_from_high = data.get('pct_from_52w_high', 0) or 0
        # Near high = more potential downside risk
        # -50% from high = risk absorbed, but also concerning
        # Optimal: -20 to -40% from high (some absorption, not collapsing)
        if pct_from_high >= -5:
            drawdown_score = 40  # Near ATH, high risk
        elif pct_from_high >= -20:
            drawdown_score = 60  # Moderate pullback
        elif pct_from_high >= -40:
            drawdown_score = 80  # Good value zone
        elif pct_from_high >= -60:
            drawdown_score = 60  # Getting concerning
        else:
            drawdown_score = 30  # Severely beaten, high risk
        components['drawdown_risk'] = {
            'value': pct_from_high,
            'score': drawdown_score,
            'weight': 0.25
        }
        
        # Calculate weighted pillar score
        pillar_score = (
            beta_score * 0.40 +
            vol_score * 0.35 +
            drawdown_score * 0.25
        )
        
        return PillarScore(
            name="Safety",
            score=round(pillar_score, 1),
            weight=self.PILLAR_WEIGHTS["safety"],
            components=components
        )
    
    def get_strength_weakness_analysis(self, result: Dict) -> Dict:
        """
        Analyze which pillars are strengths vs weaknesses.
        
        Args:
            result: Output from calculate_composite()
            
        Returns:
            {
                "strengths": ["Value", "Safety"],
                "weaknesses": ["Growth"],
                "balanced": True/False
            }
        """
        pillars = result["pillars"]
        
        scores = {
            name: data["score"] 
            for name, data in pillars.items()
        }
        
        avg_score = sum(scores.values()) / len(scores)
        
        strengths = [name.capitalize() for name, score in scores.items() if score >= avg_score + 10]
        weaknesses = [name.capitalize() for name, score in scores.items() if score <= avg_score - 10]
        
        # Balanced if no major gaps between pillars
        score_range = max(scores.values()) - min(scores.values())
        balanced = score_range < 25
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "balanced": balanced,
            "score_range": round(score_range, 1),
        }

    def _calculate_momentum_pillar(self, data: Dict) -> PillarScore:
        """
        MOMENTUM Pillar (20%) - NEW in v3.0
        
        Aligns with TLI's technical timing and sector leadership focus.
        
        Components:
        - RSI (40%): Relative Strength Index (30-70 optimal range)
        - MA Crossovers (30%): Price vs 50/200 MA
        - Market Share Momentum (30%): Revenue outperformance vs sector
        """
        components = {}
        
        # RSI (0-100)
        # 30-70 = healthy, <30 = oversold (potential buy), >70 = overbought (caution)
        rsi = data.get('rsi', 50) or 50
        if 40 <= rsi <= 60:
            rsi_score = 80  # Healthy range
        elif 30 <= rsi < 40:
            rsi_score = 90  # Approaching oversold - buy opportunity
        elif 60 < rsi <= 70:
            rsi_score = 60  # Getting overbought
        elif rsi < 30:
            rsi_score = 70  # Oversold - could be capitulation
        else:
            rsi_score = 30  # Overbought - caution
        components['rsi'] = {
            'value': rsi,
            'score': rsi_score,
            'weight': 0.40
        }
        
        # MA Crossovers (0-100)
        # Ideal: Price above 50MA, 50MA above 200MA (uptrend)
        price_vs_50ma = data.get('price_vs_50ma', 0) or 0
        price_vs_200ma = data.get('price_vs_200ma', 0) or 0
        ma_50 = data.get('sma_50', 0)
        ma_200 = data.get('sma_200', 0)
        
        # Check golden/death cross
        golden_cross = ma_50 > ma_200 if (ma_50 and ma_200) else None
        
        ma_score = 50  # Default
        if price_vs_50ma > 0 and price_vs_200ma > 0:
            ma_score = 90  # Strong uptrend
        elif price_vs_50ma > 0 and price_vs_200ma <= 0:
            ma_score = 70  # Recovering
        elif price_vs_50ma <= 0 and price_vs_200ma > 0:
            ma_score = 50  # Mixed
        elif price_vs_50ma <= 0 and price_vs_200ma <= 0:
            ma_score = 30  # Downtrend
        
        # Golden/death cross modifier
        if golden_cross is True:
            ma_score = min(100, ma_score + 10)
        elif golden_cross is False:
            ma_score = max(0, ma_score - 10)
        
        components['ma_crossover'] = {
            'value': {
                'price_vs_50ma': round(price_vs_50ma, 2) if price_vs_50ma else None,
                'price_vs_200ma': round(price_vs_200ma, 2) if price_vs_200ma else None,
                'golden_cross': golden_cross
            },
            'score': ma_score,
            'weight': 0.30
        }
        
        # Market Share Momentum (0-100)
        # Revenue growth vs industry/sector
        revenue_growth = data.get('revenue_growth', 0) or 0
        sector_growth = data.get('sector_revenue_growth', 5) or 5  # Assume 5% sector growth
        market_share_change = data.get('market_share_change_yoy', 0) or 0
        
        # Use revenue outperformance if market share not available
        outperformance = market_share_change if market_share_change != 0 else (revenue_growth - sector_growth)
        
        if outperformance >= 10:
            share_score = 100
        elif outperformance >= 5:
            share_score = 80
        elif outperformance >= 0:
            share_score = 60
        elif outperformance >= -5:
            share_score = 40
        else:
            share_score = 20
        
        components['market_share_momentum'] = {
            'value': round(outperformance, 1),
            'score': share_score,
            'weight': 0.30
        }
        
        # Calculate weighted pillar score
        pillar_score = (
            rsi_score * 0.40 +
            ma_score * 0.30 +
            share_score * 0.30
        )
        
        return PillarScore(
            name="Momentum",
            score=round(pillar_score, 1),
            weight=self.PILLAR_WEIGHTS.get("momentum", 0.20),
            components=components
        )
    
    def _determine_watch_subcategory(
        self,
        value_score: float,
        quality_score: float,
        growth_score: float,
        safety_score: float,
        margin_of_safety: float
    ) -> str:
        """
        Determine Watch sub-category for nuanced classification (v3.0).
        
        Sub-categories:
        - high_quality_expensive: Great Quality/Growth but overvalued (MoS < 0%)
        - cheap_junk: Good Value but poor Quality/Safety (potential trap)
        - turnaround_potential: Beaten down but improving metrics
        - needs_research: Doesn't fit clear categories
        
        Args:
            value_score: Value pillar score (0-100)
            quality_score: Quality pillar score (0-100)
            growth_score: Growth pillar score (0-100)
            safety_score: Safety pillar score (0-100)
            margin_of_safety: MoS percentage
            
        Returns:
            Sub-category string
        """
        # High Quality but Expensive: Great business, but overvalued
        # Quality/Growth >70 but MoS < 0%
        if (quality_score >= self.HIGH_QUALITY_EXPENSIVE_THRESHOLD or 
            growth_score >= self.HIGH_QUALITY_EXPENSIVE_THRESHOLD) and margin_of_safety < 0:
            return "high_quality_expensive"
        
        # Cheap Junk: Looks cheap but fundamentally weak
        # Value >70 but Quality or Safety < 50%
        if value_score >= self.CHEAP_JUNK_THRESHOLD and (quality_score < 50 or safety_score < 50):
            return "cheap_junk"
        
        # Turnaround Potential: Beaten down but some quality
        # Value >60 and Quality >50 and Safety <60 (some risk but quality exists)
        if value_score >= 60 and quality_score >= 50 and safety_score < 60:
            return "turnaround_potential"
        
        # Growth at Reasonable Price: Good growth, fair value
        # Growth >65 and Value >50 and MoS between -10% and +10%
        if growth_score >= 65 and value_score >= 50 and -10 <= margin_of_safety <= 10:
            return "garp_candidate"
        
        # Default: Needs more research
        return "needs_research"
    
    def generate_analyst_summary(self, result: Dict, stock_data: Dict) -> str:
        """
        Generate analyst-like narrative summary (v3.0).
        
        Emulates TLI/GDS style qualitative analysis.
        
        Args:
            result: Output from calculate_composite()
            stock_data: Original stock data
            
        Returns:
            Human-readable analysis summary
        """
        ticker = stock_data.get("ticker", "UNKNOWN")
        company_name = stock_data.get("company_name", ticker)
        sector = stock_data.get("sector", "Unknown Sector")
        
        composite = result["composite_score"]
        classification = result["classification"]
        watch_sub = result.get("watch_sub_category")
        mos = result.get("margin_of_safety", 0)
        
        pillars = result["pillars"]
        
        # Get strengths/weaknesses
        analysis = self.get_strength_weakness_analysis(result)
        strengths = analysis["strengths"]
        weaknesses = analysis["weaknesses"]
        
        # Build narrative
        summary_parts = []
        
        # Opening
        summary_parts.append(
            f"**{company_name} ({ticker})** - {sector}\n"
            f"Classification: **{classification}** | Composite: {composite}/100 | MoS: {mos:+.1f}%"
        )
        
        # Pillar highlights
        if strengths:
            summary_parts.append(f"\n✅ **Strengths**: {', '.join(strengths)}")
        if weaknesses:
            summary_parts.append(f"⚠️ **Weaknesses**: {', '.join(weaknesses)}")
        
        # Classification-specific commentary
        if classification == "Buy":
            summary_parts.append(
                f"\n📈 **Investment Thesis**: With {mos:.1f}% margin of safety and strong "
                f"fundamentals across {len(strengths)} pillars, this appears to be a compelling "
                f"value opportunity in {sector}."
            )
        elif classification == "Hold":
            summary_parts.append(
                f"\n📊 **Investment Thesis**: Fairly valued with solid fundamentals. "
                f"Consider adding on 10%+ pullbacks."
            )
        elif classification == "Watch":
            if watch_sub == "high_quality_expensive":
                summary_parts.append(
                    f"\n💎 **Watch Reason**: High-quality compounder trading at premium. "
                    f"Wait for better entry (MoS > 10%)."
                )
            elif watch_sub == "cheap_junk":
                summary_parts.append(
                    f"\n⚠️ **Watch Reason**: Value trap risk. Cheap for a reason—"
                    f"weak Quality ({pillars['quality']['score']}) or Safety ({pillars['safety']['score']})."
                )
            elif watch_sub == "turnaround_potential":
                summary_parts.append(
                    f"\n🔄 **Watch Reason**: Turnaround candidate. Monitor for improving metrics."
                )
            else:
                summary_parts.append(
                    f"\n🔍 **Watch Reason**: Needs deeper research. Mixed signals."
                )
        
        return "\n".join(summary_parts)