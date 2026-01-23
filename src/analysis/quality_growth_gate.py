"""
Quality Growth Gate - Multi-Path Stock Qualification System

Stocks qualify by meeting ANY of the 4 growth + capital efficiency paths:
- Path 1: Revenue Growth ≥ 10% AND FCF ROIC ≥ 15% (Quality Compounder)
- Path 2: Revenue Growth ≥ 15% AND FCF ROIC ≥ 10% (Balanced Growth)
- Path 3: Revenue Growth ≥ 20% AND FCF ROIC ≥ 5%  (Growth Focused)
- Path 4: Revenue Growth ≥ 25% AND FCF > 0        (Hypergrowth)

This ensures stocks meet fundamental quality thresholds before scoring.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GatePath:
    """Definition of a quality growth path"""
    path_number: int
    name: str
    min_revenue_growth: float  # Percentage (e.g., 10 for 10%)
    min_fcf_roic: float       # Percentage (e.g., 15 for 15%)
    fcf_positive_only: bool   # If True, only requires FCF > 0 instead of ROIC threshold


# Define the 4 paths
QUALITY_GROWTH_PATHS = [
    GatePath(1, "Quality Compounder", 10.0, 15.0, False),
    GatePath(2, "Balanced Growth", 15.0, 10.0, False),
    GatePath(3, "Growth Focused", 20.0, 5.0, False),
    GatePath(4, "Hypergrowth", 25.0, 0.0, True),  # Only requires FCF > 0
]


class QualityGrowthGate:
    """
    Multi-path qualification gate for stock screening.
    
    A stock passes the gate if it meets the criteria of ANY path.
    Higher growth can compensate for lower capital efficiency, and vice versa.
    """
    
    def __init__(self):
        self.paths = QUALITY_GROWTH_PATHS
        self.logger = logger
    
    def evaluate(
        self, 
        revenue_growth: float, 
        fcf_roic: float, 
        free_cash_flow: float = 0
    ) -> Dict:
        """
        Evaluate a stock against all quality growth paths.
        
        Args:
            revenue_growth: Next-year revenue growth as percentage (e.g., 15.0 for 15%)
            fcf_roic: FCF Return on Invested Capital as percentage (e.g., 12.0 for 12%)
            free_cash_flow: Absolute FCF value (for Path 4 FCF-positive check)
            
        Returns:
            {
                "passed": True/False,
                "paths_matched": [1, 2],  # List of matched path numbers
                "best_path": 1,           # Highest quality path matched (lowest number)
                "best_path_name": "Quality Compounder",
                "evaluation_details": {...}
            }
        """
        paths_matched = []
        evaluation_details = {}
        
        for path in self.paths:
            passed_growth = revenue_growth >= path.min_revenue_growth
            
            if path.fcf_positive_only:
                passed_quality = free_cash_flow > 0
                quality_metric = "FCF Positive"
                quality_value = free_cash_flow > 0
            else:
                passed_quality = fcf_roic >= path.min_fcf_roic
                quality_metric = f"FCF ROIC ≥ {path.min_fcf_roic}%"
                quality_value = fcf_roic
            
            path_passed = passed_growth and passed_quality
            
            evaluation_details[f"path_{path.path_number}"] = {
                "name": path.name,
                "passed": path_passed,
                "growth_required": path.min_revenue_growth,
                "growth_actual": revenue_growth,
                "growth_met": passed_growth,
                "quality_metric": quality_metric,
                "quality_actual": quality_value if not path.fcf_positive_only else (free_cash_flow > 0),
                "quality_met": passed_quality,
            }
            
            if path_passed:
                paths_matched.append(path.path_number)
        
        passed = len(paths_matched) > 0
        best_path = min(paths_matched) if paths_matched else None
        best_path_name = None
        
        if best_path:
            best_path_name = next(p.name for p in self.paths if p.path_number == best_path)
        
        return {
            "passed": passed,
            "paths_matched": paths_matched,
            "best_path": best_path,
            "best_path_name": best_path_name,
            "total_paths_matched": len(paths_matched),
            "evaluation_details": evaluation_details,
        }
    
    def get_path_description(self, path_number: int) -> str:
        """Get human-readable description of a path"""
        for path in self.paths:
            if path.path_number == path_number:
                if path.fcf_positive_only:
                    return f"{path.name}: Revenue Growth ≥ {path.min_revenue_growth}% AND FCF Positive"
                else:
                    return f"{path.name}: Revenue Growth ≥ {path.min_revenue_growth}% AND FCF ROIC ≥ {path.min_fcf_roic}%"
        return "Unknown path"
    
    @staticmethod
    def calculate_fcf_roic(
        free_cash_flow: float,
        total_assets: float,
        current_liabilities: float
    ) -> float:
        """
        Calculate FCF ROIC (Return on Invested Capital).
        
        FCF ROIC = FCF / Invested Capital
        Invested Capital = Total Assets - Current Liabilities
        
        Args:
            free_cash_flow: Free cash flow (TTM)
            total_assets: Total assets from balance sheet
            current_liabilities: Current liabilities from balance sheet
            
        Returns:
            FCF ROIC as percentage (e.g., 15.0 for 15%)
        """
        invested_capital = total_assets - current_liabilities
        
        if invested_capital <= 0:
            return 0.0
        
        fcf_roic = (free_cash_flow / invested_capital) * 100
        return round(fcf_roic, 2)


# Trigger definitions for Stage 1 Attention (v3.0)
# Synced with full_universe_scan.py and diagnose_stock.py
# Enhanced with solvency guardrails, dynamic yield, and Trigger E (Moat)
class AttentionTriggers:
    """
    Trigger conditions for Stage 1 Attention List (v3.0).
    Stocks enter attention list if ANY trigger fires.
    
    Triggers (5 paths with guardrails):
    - A: Significant Drop (≥-40% + FCF positive + Solvency Guardrail)
    - B: Quality Growth Gate (4 paths: rev growth + FCF ROIC)
    - C: Deep Value (MoS ≥30% AND Dynamic FCF Yield) - Adjusted for rate environment
    - D: Consistent Growth (3yr CAGR ≥20% + Gross Margin ≥30%)
    - E: Moat Strength (Recurring Revenue >70% OR Market Share Momentum >5%)
    
    Red Flag Veto: Beneish M-Score > -2.22 vetoes entry (earnings manipulation)
    """
    
    # Dynamic yield threshold: 10-Year Treasury + 2% spread
    # For 2026 rate environment (~4.5% treasury), this means ~6.5% minimum
    DEFAULT_TREASURY_RATE = 4.5  # Can be updated dynamically
    FCF_YIELD_SPREAD = 2.0  # Spread above treasury
    
    @classmethod
    def get_dynamic_fcf_yield_threshold(cls, treasury_rate: float = None) -> float:
        """Get FCF Yield threshold adjusted for rate environment."""
        rate = treasury_rate if treasury_rate is not None else cls.DEFAULT_TREASURY_RATE
        return rate + cls.FCF_YIELD_SPREAD
    
    # v3.1: Tighter thresholds for quality filtering
    DROP_THRESHOLD = -50  # Was -40, now -50 for deeper value
    MOS_THRESHOLD = 40    # Was 30, now 40 for stronger value signal
    MIN_ALTMAN_Z = 2.0    # Was 1.8, now 2.0 (safer zone)
    MIN_ICR = 4.0         # Was 3.0, now 4.0 (stronger interest coverage)
    RECURRING_REV_THRESHOLD = 75  # Was 70, now 75 for clearer moat
    MARKET_SHARE_THRESHOLD = 8    # Was 5, now 8 for meaningful momentum
    
    @staticmethod
    def trigger_significant_drop(
        pct_from_52w_high: float,
        free_cash_flow: float,
        altman_z_score: float = None,
        interest_coverage: float = None
    ) -> Tuple[bool, Dict]:
        """
        Trigger A: Significant Drop + Solvency Guardrail (v3.1 - TIGHTENED)
        
        Fires when:
        - Price dropped 50%+ from 52-week high (was 40%)
        - Still FCF positive (not dying business)
        - Solvency check: Altman Z-Score > 2.0 OR Interest Coverage > 4.0 (tightened)
        
        Returns:
            (triggered: bool, metrics: dict)
        """
        # Check solvency guardrail (v3.1: stricter thresholds)
        z_ok = altman_z_score is not None and altman_z_score > AttentionTriggers.MIN_ALTMAN_Z
        icr_ok = interest_coverage is not None and interest_coverage > AttentionTriggers.MIN_ICR
        solvency_ok = z_ok or icr_ok
        
        # v3.1: Require at least one solvency metric (no free pass)
        if altman_z_score is None and interest_coverage is None:
            solvency_ok = False  # Changed from True - must verify solvency
            solvency_note = "Solvency data required - skipping until available"
        else:
            solvency_note = f"Z-Score: {altman_z_score} (>{AttentionTriggers.MIN_ALTMAN_Z}), ICR: {interest_coverage} (>{AttentionTriggers.MIN_ICR})"
        
        conditions = {
            "price_drop_50pct": pct_from_52w_high <= AttentionTriggers.DROP_THRESHOLD,
            "fcf_positive": free_cash_flow > 0,
            "solvency_ok": solvency_ok,
        }
        
        triggered = all(conditions.values())
        
        return triggered, {
            "type": "significant_drop",
            "conditions": conditions,
            "value": pct_from_52w_high,
            "threshold": AttentionTriggers.DROP_THRESHOLD,
            "fcf_positive": free_cash_flow > 0,
            "solvency": {
                "passed": solvency_ok,
                "altman_z_score": altman_z_score,
                "interest_coverage": interest_coverage,
                "note": solvency_note
            },
            "signal": "Beaten down quality with solvency confirmed" if triggered else None
        }
    
    @staticmethod
    def trigger_quality_growth(
        revenue_growth: float,
        fcf_roic: float,
        free_cash_flow: float
    ) -> Tuple[bool, Dict]:
        """
        Trigger B: Quality Growth Gate
        
        Fires when stock passes any of the 4 quality growth paths:
        - Path 1: Revenue ≥10% + FCF ROIC ≥15% (Quality Compounder)
        - Path 2: Revenue ≥15% + FCF ROIC ≥10% (Balanced Growth)
        - Path 3: Revenue ≥20% + FCF ROIC ≥5%  (Growth Focused)
        - Path 4: Revenue ≥25% + FCF > 0       (Hypergrowth)
        
        Returns:
            (triggered: bool, metrics: dict)
        """
        gate = QualityGrowthGate()
        result = gate.evaluate(revenue_growth, fcf_roic, free_cash_flow)
        
        return result["passed"], {
            "type": "quality_growth",
            "path": result["best_path"],
            "path_name": result["best_path_name"],
            "paths_matched": result["paths_matched"],
            "revenue_growth": revenue_growth,
            "fcf_roic": fcf_roic,
            "signal": f"Quality compounder - {result['best_path_name']}" if result["passed"] else None
        }
    
    @classmethod
    def trigger_deep_value(
        cls,
        margin_of_safety: float,
        fcf_yield: float,
        treasury_rate: float = None
    ) -> Tuple[bool, Dict]:
        """
        Trigger C: Deep Value with Dynamic Yield (v3.1 - TIGHTENED)
        
        Fires when BOTH conditions are met:
        - MoS ≥ 40% (was 30% - now requires stronger undervaluation)
        - FCF Yield ≥ (10-Year Treasury + 2%) - Dynamic threshold
        
        For 2026 (~4.5% treasury), this means FCF Yield ≥ 6.5%
        This prevents low-yield traps in high-rate environments.
        
        Returns:
            (triggered: bool, metrics: dict)
        """
        # Get dynamic FCF yield threshold
        yield_threshold = cls.get_dynamic_fcf_yield_threshold(treasury_rate)
        
        # v3.1: Raised MoS threshold from 30% to 40%
        mos_threshold = cls.MOS_THRESHOLD
        triggered = margin_of_safety >= mos_threshold and fcf_yield >= yield_threshold
        
        return triggered, {
            "type": "deep_value",
            "margin_of_safety": margin_of_safety,
            "fcf_yield": fcf_yield,
            "mos_threshold": mos_threshold,
            "fcf_yield_threshold": yield_threshold,
            "treasury_rate": treasury_rate or cls.DEFAULT_TREASURY_RATE,
            "dynamic_threshold_explanation": f"FCF Yield must exceed Treasury ({treasury_rate or cls.DEFAULT_TREASURY_RATE}%) + {cls.FCF_YIELD_SPREAD}% spread",
            "signal": "Deep value - undervalued with rate-adjusted cash generation" if triggered else None
        }
    
    @staticmethod
    def trigger_consistent_growth(
        ticker: str,
        gross_margin: float,
        total_revenue: float
    ) -> Tuple[bool, Dict]:
        """
        Trigger D: Consistent Growth (v2.2 - NEW)
        
        Catches high-growth reinvestors like ADUR that may have negative FCF.
        
        Fires when:
        - 3-year Revenue CAGR ≥ 20%
        - Gross Margin ≥ 30% (path to profitability)
        - Revenue ≥ $50M (meaningful size)
        
        Note: This trigger requires an extra API call for historical data,
        so it's only evaluated if other triggers don't fire.
        
        Returns:
            (triggered: bool, metrics: dict)
        """
        # Pre-check size and margin requirements
        if gross_margin < 30 or total_revenue < 50_000_000:
            return False, {
                "type": "consistent_growth",
                "triggered": False,
                "reason": "Minimum requirements not met",
                "gross_margin": gross_margin,
                "total_revenue": total_revenue
            }
        
        # Calculate CAGR (requires yfinance call)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            income_stmt = stock.income_stmt
            
            if income_stmt is None or income_stmt.empty:
                return False, {"type": "consistent_growth", "triggered": False, "reason": "No income data"}
            
            # Find revenue row
            revenue_row = None
            for row_name in ['Total Revenue', 'Revenue', 'Operating Revenue']:
                if row_name in income_stmt.index:
                    revenue_row = income_stmt.loc[row_name]
                    break
            
            if revenue_row is None:
                return False, {"type": "consistent_growth", "triggered": False, "reason": "No revenue data"}
            
            revenues = revenue_row.dropna().sort_index()
            if len(revenues) < 2:
                return False, {"type": "consistent_growth", "triggered": False, "reason": "Insufficient history"}
            
            # Calculate 3-year CAGR
            latest_revenue = revenues.iloc[-1]
            target_idx = max(0, len(revenues) - 4)  # 3 years back
            oldest_revenue = revenues.iloc[target_idx]
            actual_years = len(revenues) - 1 - target_idx
            
            if actual_years < 2 or oldest_revenue <= 0 or latest_revenue <= 0:
                return False, {"type": "consistent_growth", "triggered": False, "reason": "Invalid revenue data"}
            
            cagr = ((latest_revenue / oldest_revenue) ** (1 / actual_years) - 1) * 100
            
            triggered = cagr >= 20
            
            return triggered, {
                "type": "consistent_growth",
                "cagr": round(cagr, 1),
                "years": actual_years,
                "gross_margin": gross_margin,
                "latest_revenue": float(latest_revenue),
                "oldest_revenue": float(oldest_revenue),
                "note": "High-growth reinvestor (may have negative FCF)" if triggered else None,
                "signal": f"Consistent {cagr:.0f}% CAGR growth" if triggered else None
            }
            
        except Exception as e:
            return False, {"type": "consistent_growth", "triggered": False, "reason": f"CAGR calculation failed: {str(e)}"}
    
    @classmethod
    def trigger_moat_strength(
        cls,
        recurring_revenue_pct: float,
        market_share_change: float,
        sector: str = None,
        industry: str = None
    ) -> Tuple[bool, Dict]:
        """
        Trigger E: Moat Strength (v3.1 - TIGHTENED)
        
        Aligns with GDS compounder focus and TLI sector leadership.
        
        Fires when:
        - Recurring Revenue > 75% of Total (was 70%), OR
        - Market Share Momentum > 8% YoY gain (was 5%)
        
        v3.1: REMOVED SaaS auto-assumption to prevent tech bloat.
        Stocks must have actual recurring revenue data to qualify.
        
        Returns:
            (triggered: bool, metrics: dict)
        """
        # v3.1: NO MORE auto-assumption for SaaS - must have actual data
        # This prevents tech bloat from inflated trigger passes
        estimated_recurring = recurring_revenue_pct
        estimation_note = None
        
        # Only estimate if we have reasonable confidence indicators
        if recurring_revenue_pct == 0 and industry:
            industry_lower = industry.lower()
            # v3.1: Only estimate for pure subscription businesses with conservative value
            if "subscription" in industry_lower and "software" in industry_lower:
                estimated_recurring = 60  # Conservative estimate, won't trigger (need 75)
                estimation_note = "Subscription software - conservative estimate (actual data preferred)"
            # No longer assuming for generic "software" or "saas"
        
        # v3.1: Raised thresholds
        has_recurring_moat = estimated_recurring >= cls.RECURRING_REV_THRESHOLD
        has_market_share_moat = market_share_change >= cls.MARKET_SHARE_THRESHOLD
        
        triggered = has_recurring_moat or has_market_share_moat
        
        moat_reason = []
        if has_recurring_moat:
            moat_reason.append(f"High Recurring Revenue ({estimated_recurring}% ≥{cls.RECURRING_REV_THRESHOLD}%)")
        if has_market_share_moat:
            moat_reason.append(f"Market Share Gains ({market_share_change}% ≥{cls.MARKET_SHARE_THRESHOLD}%)")
        
        return triggered, {
            "type": "moat_strength",
            "recurring_revenue_pct": recurring_revenue_pct,
            "estimated_recurring_pct": estimated_recurring,
            "estimation_note": estimation_note,
            "market_share_change": market_share_change,
            "has_recurring_moat": has_recurring_moat,
            "has_market_share_moat": has_market_share_moat,
            "thresholds": {
                "recurring": cls.RECURRING_REV_THRESHOLD,
                "market_share": cls.MARKET_SHARE_THRESHOLD
            },
            "moat_sources": moat_reason,
            "signal": f"Durable moat: {', '.join(moat_reason)}" if triggered else None
        }
    
    @classmethod
    def evaluate_all_triggers(
        cls,
        stock_data: Dict,
        ticker: str = None,
        treasury_rate: float = None,
        include_red_flags: bool = True
    ) -> Dict:
        """
        Evaluate all triggers for a stock (v3.0).
        
        Args:
            stock_data: Dictionary with stock metrics
            ticker: Stock ticker (required for Trigger D CAGR calculation)
            treasury_rate: Current 10-year treasury rate for dynamic yield threshold
            include_red_flags: Whether to check red flags (Beneish M-Score)
            
        Returns:
            {
                "any_triggered": True/False,
                "vetoed": True/False,  # If red flags detected
                "triggers": [
                    {"type": "significant_drop", ...},
                    {"type": "quality_growth", "path": 2, ...}
                ],
                "red_flags": [...]  # If any
            }
        """
        triggers_fired = []
        red_flags = []
        vetoed = False
        
        # Extract metrics with defaults
        pct_from_52w_high = stock_data.get('pct_from_52w_high', 0) or 0
        free_cash_flow = stock_data.get('free_cash_flow', 0) or 0
        revenue_growth = stock_data.get('revenue_growth', 0) or 0
        fcf_roic = stock_data.get('fcf_roic', 0) or 0
        margin_of_safety = stock_data.get('margin_of_safety', 0) or 0
        fcf_yield = stock_data.get('fcf_yield', 0) or 0
        gross_margin = stock_data.get('gross_margin', 0) or 0
        total_revenue = stock_data.get('total_revenue') or stock_data.get('revenue', 0) or 0
        
        # v3.0: Additional metrics for solvency and moats
        altman_z_score = stock_data.get('altman_z_score')
        interest_coverage = stock_data.get('interest_coverage')
        recurring_revenue_pct = stock_data.get('recurring_revenue_pct', 0) or 0
        market_share_change = stock_data.get('market_share_change_yoy', 0) or 0
        sector = stock_data.get('sector')
        industry = stock_data.get('industry')
        
        # Red flag check (v3.0)
        if include_red_flags:
            beneish_m_score = stock_data.get('beneish_m_score')
            if beneish_m_score is not None and beneish_m_score > -2.22:
                red_flags.append({
                    "type": "earnings_manipulation",
                    "beneish_m_score": beneish_m_score,
                    "threshold": -2.22,
                    "severity": "high"
                })
                vetoed = True
        
        # If vetoed, return early
        if vetoed:
            return {
                "any_triggered": False,
                "vetoed": True,
                "veto_reason": "Red flag detected (potential earnings manipulation)",
                "trigger_count": 0,
                "triggers": [],
                "red_flags": red_flags
            }
        
        # Trigger A: Significant Drop with Solvency Guardrail (v3.0)
        triggered, metrics = cls.trigger_significant_drop(
            pct_from_52w_high, free_cash_flow, altman_z_score, interest_coverage
        )
        if triggered:
            triggers_fired.append(metrics)
        
        # Trigger B: Quality Growth Gate (4 paths)
        triggered, metrics = cls.trigger_quality_growth(revenue_growth, fcf_roic, free_cash_flow)
        if triggered:
            triggers_fired.append(metrics)
        
        # Trigger C: Deep Value with Dynamic Yield (v3.0)
        triggered, metrics = cls.trigger_deep_value(margin_of_safety, fcf_yield, treasury_rate)
        if triggered:
            triggers_fired.append(metrics)
        
        # Trigger D: Consistent Growth (only if other triggers didn't fire)
        # This saves API calls since CAGR requires extra yfinance request
        if not triggers_fired and ticker:
            triggered, metrics = cls.trigger_consistent_growth(ticker, gross_margin, total_revenue)
            if triggered:
                triggers_fired.append(metrics)
        
        # Trigger E: Moat Strength (v3.0 - always check, no API calls)
        triggered, metrics = cls.trigger_moat_strength(
            recurring_revenue_pct, market_share_change, sector, industry
        )
        if triggered:
            triggers_fired.append(metrics)
        
        return {
            "any_triggered": len(triggers_fired) > 0,
            "vetoed": False,
            "trigger_count": len(triggers_fired),
            "triggers": triggers_fired,
            "red_flags": red_flags
        }
