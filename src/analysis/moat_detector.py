"""
Moat Detector - Competitive Advantage & Durability Analysis (v3.0)

Detects economic moats for Trigger E: Moat Strength.
Aligns with GreenDotStocks (GDS) compounder focus and TheLongInvestor (TLI) sector leadership.

Moat Indicators:
- Recurring Revenue %: Subscription/SaaS model durability (>70% = strong moat)
- Market Share Momentum: Gaining vs losing share (>5% YoY = positive)
- Gross Margin Stability: Pricing power indicator (>50% + stable = moat)
- Customer Concentration: Revenue diversification (top customer <20% = healthy)
- R&D Intensity: Innovation investment (5-15% optimal for tech)
- Brand Strength: Pricing premium ability

Usage:
    from src.analysis.moat_detector import MoatDetector
    detector = MoatDetector()
    result = detector.evaluate(stock_data)
    if result["has_moat"]:
        print("Moat Type:", result["moat_type"])
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MoatResult:
    """Result of moat evaluation"""
    has_moat: bool
    moat_score: float  # 0-100
    moat_type: str  # "wide", "narrow", "none"
    moat_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0


class MoatDetector:
    """
    Economic moat detector for durable competitive advantages.
    
    Evaluates multiple moat sources:
    1. Recurring Revenue (SaaS/subscription durability)
    2. Market Share Momentum (sector leadership)
    3. Gross Margin Stability (pricing power)
    4. Switching Costs (customer stickiness)
    5. Network Effects (platform value)
    6. Cost Advantages (scale economics)
    """
    
    # Thresholds for moat detection
    RECURRING_REV_STRONG = 70  # >70% recurring = strong moat
    RECURRING_REV_MODERATE = 50  # >50% = moderate moat signal
    MARKET_SHARE_GAIN_THRESHOLD = 5  # >5% YoY gain = positive momentum
    GROSS_MARGIN_MOAT = 50  # >50% suggests pricing power
    GROSS_MARGIN_STABILITY_MAX = 3  # <3% std dev = stable
    R_AND_D_INTENSITY_MIN = 5  # >5% R&D/Sales = innovation focus
    R_AND_D_INTENSITY_MAX = 15  # <15% = efficient R&D
    CUSTOMER_CONCENTRATION_MAX = 20  # Top customer <20% = diversified
    
    # Moat scoring weights
    MOAT_WEIGHTS = {
        "recurring_revenue": 0.25,
        "gross_margin_stability": 0.20,
        "market_share_momentum": 0.20,
        "customer_diversification": 0.15,
        "r_and_d_efficiency": 0.10,
        "scale_advantages": 0.10
    }
    
    def __init__(self):
        self.logger = logger
    
    def detect_recurring_revenue_moat(self, data: Dict) -> Tuple[float, str]:
        """
        Detect recurring revenue moat (SaaS/subscription model).
        
        Returns:
            (score: 0-100, moat_source: str or None)
        """
        recurring_rev_pct = data.get("recurring_revenue_pct", 0)
        subscription_rev_pct = data.get("subscription_revenue_pct", 0)
        
        # Use whichever is available
        pct = max(recurring_rev_pct, subscription_rev_pct)
        
        # Also estimate from sector if not directly available
        if pct == 0:
            sector = data.get("sector", "").lower()
            industry = data.get("industry", "").lower()
            
            # SaaS/Software typically has high recurring
            if "software" in industry or "saas" in industry:
                pct = 60  # Assume moderate recurring
            elif "subscription" in industry:
                pct = 70
            elif sector == "technology":
                pct = 40  # Conservative estimate
        
        if pct >= self.RECURRING_REV_STRONG:
            return 100, "High Recurring Revenue"
        elif pct >= self.RECURRING_REV_MODERATE:
            return 70, "Moderate Recurring Revenue"
        elif pct >= 30:
            return 40, None
        else:
            return 20, None
    
    def detect_gross_margin_moat(self, data: Dict) -> Tuple[float, str]:
        """
        Detect pricing power moat via gross margin analysis.
        
        High, stable gross margins suggest pricing power.
        
        Returns:
            (score: 0-100, moat_source: str or None)
        """
        gross_margin = data.get("gross_margin", 0)
        gross_margin_5yr_avg = data.get("gross_margin_5yr_avg", gross_margin)
        gross_margin_std = data.get("gross_margin_std", 5)  # Estimate if not available
        
        # Score based on level
        if gross_margin >= 80:
            level_score = 100
        elif gross_margin >= 60:
            level_score = 80
        elif gross_margin >= self.GROSS_MARGIN_MOAT:
            level_score = 60
        elif gross_margin >= 30:
            level_score = 40
        else:
            level_score = 20
        
        # Stability bonus/penalty
        stability_modifier = 0
        if gross_margin_std < self.GROSS_MARGIN_STABILITY_MAX:
            stability_modifier = 10  # Stable margins = bonus
        elif gross_margin_std > 10:
            stability_modifier = -15  # Volatile margins = penalty
        
        # Trend bonus
        trend_modifier = 0
        if gross_margin > gross_margin_5yr_avg:
            trend_modifier = 5  # Expanding margins
        elif gross_margin < gross_margin_5yr_avg - 5:
            trend_modifier = -10  # Contracting margins
        
        final_score = max(0, min(100, level_score + stability_modifier + trend_modifier))
        
        moat_source = None
        if gross_margin >= self.GROSS_MARGIN_MOAT and gross_margin_std < self.GROSS_MARGIN_STABILITY_MAX:
            moat_source = "Pricing Power (High Stable Margins)"
        
        return final_score, moat_source
    
    def detect_market_share_momentum(self, data: Dict) -> Tuple[float, str]:
        """
        Detect market share momentum (sector leadership).
        
        Aligns with TLI's focus on "sector leaders."
        
        Returns:
            (score: 0-100, moat_source: str or None)
        """
        market_share_change = data.get("market_share_change_yoy", 0)  # % change
        revenue_vs_industry = data.get("revenue_growth_vs_industry", 0)  # Relative growth
        
        # Estimate if not available: use revenue growth vs sector
        if market_share_change == 0 and revenue_vs_industry == 0:
            revenue_growth = data.get("revenue_growth", 0)
            sector_growth = data.get("sector_revenue_growth", 5)  # Assume 5% sector growth
            revenue_vs_industry = revenue_growth - sector_growth
        
        # Score based on outperformance
        if market_share_change >= self.MARKET_SHARE_GAIN_THRESHOLD:
            score = 100
            moat_source = "Market Share Gains"
        elif market_share_change >= 2:
            score = 75
            moat_source = "Moderate Market Share Gains"
        elif revenue_vs_industry >= 10:
            score = 80
            moat_source = "Outpacing Industry Growth"
        elif revenue_vs_industry >= 5:
            score = 60
            moat_source = None
        elif market_share_change >= 0 or revenue_vs_industry >= 0:
            score = 50
            moat_source = None
        else:
            score = 25
            moat_source = None
        
        return score, moat_source
    
    def detect_customer_diversification(self, data: Dict) -> Tuple[float, str]:
        """
        Detect customer diversification (reduced concentration risk).
        
        Returns:
            (score: 0-100, moat_source: str or None)
        """
        top_customer_pct = data.get("top_customer_revenue_pct", 10)  # Default 10%
        top_5_customers_pct = data.get("top_5_customers_revenue_pct", 30)  # Default 30%
        
        # Score based on concentration
        if top_customer_pct < 10 and top_5_customers_pct < 25:
            score = 100
            moat_source = "Highly Diversified Customer Base"
        elif top_customer_pct < self.CUSTOMER_CONCENTRATION_MAX:
            score = 75
            moat_source = "Well Diversified Customers"
        elif top_customer_pct < 30:
            score = 50
            moat_source = None
        elif top_customer_pct < 50:
            score = 30
            moat_source = None
        else:
            score = 10
            moat_source = None
        
        return score, moat_source
    
    def detect_r_and_d_moat(self, data: Dict) -> Tuple[float, str]:
        """
        Detect R&D/innovation moat.
        
        Returns:
            (score: 0-100, moat_source: str or None)
        """
        r_and_d_pct = data.get("r_and_d_to_revenue", 0)
        patent_count = data.get("patent_count", 0)
        
        # Optimal R&D intensity varies by sector
        sector = data.get("sector", "").lower()
        
        if sector == "technology" or sector == "healthcare":
            optimal_range = (8, 20)
        else:
            optimal_range = (3, 10)
        
        # Score
        if optimal_range[0] <= r_and_d_pct <= optimal_range[1]:
            score = 80
            moat_source = "Innovation Investment"
        elif r_and_d_pct > optimal_range[1]:
            score = 60  # High R&D but might be inefficient
            moat_source = None
        elif r_and_d_pct >= self.R_AND_D_INTENSITY_MIN:
            score = 70
            moat_source = None
        else:
            score = 40
            moat_source = None
        
        # Patent bonus
        if patent_count > 100:
            score = min(100, score + 15)
            moat_source = moat_source or "Patent Portfolio"
        
        return score, moat_source
    
    def detect_scale_advantages(self, data: Dict) -> Tuple[float, str]:
        """
        Detect scale-based cost advantages.
        
        Returns:
            (score: 0-100, moat_source: str or None)
        """
        market_cap = data.get("market_cap", 0)
        operating_margin = data.get("operating_margin", 0)
        sector_avg_margin = data.get("sector_operating_margin", 10)
        
        # Large companies with better margins = scale advantage
        is_large_cap = market_cap >= 10_000_000_000  # $10B+
        margin_advantage = operating_margin - sector_avg_margin
        
        if is_large_cap and margin_advantage >= 10:
            score = 90
            moat_source = "Scale Cost Advantages"
        elif is_large_cap and margin_advantage >= 5:
            score = 70
            moat_source = "Moderate Scale Advantages"
        elif margin_advantage >= 10:
            score = 65
            moat_source = None
        elif is_large_cap:
            score = 55
            moat_source = None
        else:
            score = 40
            moat_source = None
        
        return score, moat_source
    
    def evaluate(self, data: Dict, ticker: str = None) -> Dict:
        """
        Comprehensive moat evaluation.
        
        Args:
            data: Stock metrics dictionary
            ticker: Stock ticker for logging
            
        Returns:
            {
                "has_moat": bool,
                "moat_score": 0-100,
                "moat_type": "wide/narrow/none",
                "moat_sources": [...],
                "components": {...},
                "trigger_e_passed": bool
            }
        """
        ticker = ticker or data.get("ticker", "UNKNOWN")
        moat_sources = []
        components = {}
        
        # Evaluate each moat source
        recurring_score, recurring_moat = self.detect_recurring_revenue_moat(data)
        components["recurring_revenue"] = {"score": recurring_score, "moat": recurring_moat}
        if recurring_moat:
            moat_sources.append(recurring_moat)
        
        margin_score, margin_moat = self.detect_gross_margin_moat(data)
        components["gross_margin_stability"] = {"score": margin_score, "moat": margin_moat}
        if margin_moat:
            moat_sources.append(margin_moat)
        
        share_score, share_moat = self.detect_market_share_momentum(data)
        components["market_share_momentum"] = {"score": share_score, "moat": share_moat}
        if share_moat:
            moat_sources.append(share_moat)
        
        customer_score, customer_moat = self.detect_customer_diversification(data)
        components["customer_diversification"] = {"score": customer_score, "moat": customer_moat}
        if customer_moat:
            moat_sources.append(customer_moat)
        
        rd_score, rd_moat = self.detect_r_and_d_moat(data)
        components["r_and_d_efficiency"] = {"score": rd_score, "moat": rd_moat}
        if rd_moat:
            moat_sources.append(rd_moat)
        
        scale_score, scale_moat = self.detect_scale_advantages(data)
        components["scale_advantages"] = {"score": scale_score, "moat": scale_moat}
        if scale_moat:
            moat_sources.append(scale_moat)
        
        # Calculate weighted moat score
        moat_score = sum(
            components[key]["score"] * self.MOAT_WEIGHTS[key]
            for key in self.MOAT_WEIGHTS
        )
        moat_score = round(moat_score, 1)
        
        # Determine moat type
        if moat_score >= 75 and len(moat_sources) >= 2:
            moat_type = "wide"
            has_moat = True
        elif moat_score >= 60 and len(moat_sources) >= 1:
            moat_type = "narrow"
            has_moat = True
        else:
            moat_type = "none"
            has_moat = False
        
        # Trigger E criteria: Recurring Revenue >70% OR Market Share Momentum >5% gain
        recurring_rev_pct = data.get("recurring_revenue_pct") or data.get("subscription_revenue_pct", 0)
        market_share_change = data.get("market_share_change_yoy", 0)
        
        # Estimate recurring from sector if not available
        sector = data.get("sector", "").lower()
        industry = data.get("industry", "").lower()
        if recurring_rev_pct == 0 and ("software" in industry or "saas" in industry):
            recurring_rev_pct = 70  # Assume SaaS has high recurring
        
        trigger_e_passed = (recurring_rev_pct >= 70) or (market_share_change >= 5)
        
        # Also pass if strong moat score with identifiable sources
        if moat_score >= 70 and len(moat_sources) >= 2:
            trigger_e_passed = True
        
        # Calculate confidence based on data availability
        available_metrics = sum([
            data.get("recurring_revenue_pct", 0) > 0,
            data.get("gross_margin", 0) > 0,
            data.get("market_share_change_yoy", 0) != 0,
            data.get("r_and_d_to_revenue", 0) > 0,
            data.get("market_cap", 0) > 0
        ])
        confidence = (available_metrics / 5) * 100
        
        return {
            "ticker": ticker,
            "has_moat": has_moat,
            "moat_score": moat_score,
            "moat_type": moat_type,
            "moat_sources": moat_sources,
            "components": components,
            "trigger_e_passed": trigger_e_passed,
            "trigger_e_reason": (
                f"Recurring Revenue {recurring_rev_pct}% >= 70%" if recurring_rev_pct >= 70
                else f"Market Share Gain {market_share_change}% >= 5%" if market_share_change >= 5
                else f"Strong Moat Score {moat_score} with sources: {moat_sources}" if trigger_e_passed
                else "Does not meet Trigger E criteria"
            ),
            "confidence": round(confidence, 1)
        }
    
    def quick_moat_check(self, data: Dict) -> Tuple[bool, Dict]:
        """
        Quick check for Trigger E without full evaluation.
        
        Passes if: Recurring Revenue >70% OR Market Share Momentum >5%
        
        Returns:
            (passed: bool, details: dict)
        """
        recurring_rev_pct = data.get("recurring_revenue_pct") or data.get("subscription_revenue_pct", 0)
        market_share_change = data.get("market_share_change_yoy", 0)
        
        # Sector-based estimation
        sector = data.get("sector", "").lower()
        industry = data.get("industry", "").lower()
        if recurring_rev_pct == 0 and ("software" in industry or "saas" in industry):
            recurring_rev_pct = 70
        
        passed = (recurring_rev_pct >= 70) or (market_share_change >= 5)
        
        return passed, {
            "passed": passed,
            "recurring_revenue_pct": recurring_rev_pct,
            "market_share_change_yoy": market_share_change,
            "reason": (
                f"High Recurring Revenue ({recurring_rev_pct}%)" if recurring_rev_pct >= 70
                else f"Market Share Gains ({market_share_change}%)" if market_share_change >= 5
                else "Does not meet moat criteria"
            )
        }


# Export for easy importing
__all__ = ['MoatDetector', 'MoatResult']
