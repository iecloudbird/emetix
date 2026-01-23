"""
Red Flag Detector - Financial Health & Manipulation Screening (v3.0)

Detects potential value traps, earnings manipulation, and solvency risks.
Provides guardrails for the attention scan to prevent false positives.

Key Metrics:
- Beneish M-Score: Earnings manipulation detection (> -2.22 = red flag)
- Altman Z-Score: Bankruptcy prediction (< 1.8 = distress zone)
- Interest Coverage Ratio: Debt servicing ability (< 3.0 = concern)
- Piotroski F-Score: Financial strength (< 4 = weak fundamentals)

Usage:
    from src.analysis.red_flag_detector import RedFlagDetector
    detector = RedFlagDetector()
    result = detector.evaluate(stock_data)
    if result["has_red_flags"]:
        print("VETO: ", result["red_flags"])
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RedFlagResult:
    """Result of red flag evaluation"""
    has_red_flags: bool
    red_flags: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    solvency_score: float = 0.0  # 0-100, higher = safer
    manipulation_risk: str = "low"  # low/medium/high
    confidence: float = 0.0  # Data quality confidence


class RedFlagDetector:
    """
    Financial health and manipulation detector.
    
    Provides guardrails for attention scan to filter out:
    1. Potential bankruptcies (Altman Z-Score)
    2. Earnings manipulators (Beneish M-Score)
    3. Debt-burdened companies (Interest Coverage)
    4. Weak fundamentals (Piotroski F-Score)
    """
    
    # Thresholds
    BENEISH_MANIPULATION_THRESHOLD = -2.22  # Above = likely manipulation
    ALTMAN_DISTRESS_THRESHOLD = 1.8  # Below = distress zone
    ALTMAN_SAFE_THRESHOLD = 3.0  # Above = safe zone
    INTEREST_COVERAGE_MIN = 3.0  # Below = debt concern
    PIOTROSKI_WEAK_THRESHOLD = 4  # Below = weak fundamentals
    
    def __init__(self):
        self.logger = logger
    
    def calculate_beneish_m_score(self, data: Dict) -> Optional[float]:
        """
        Calculate Beneish M-Score for earnings manipulation detection.
        
        M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI 
                  + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
        
        Components:
        - DSRI: Days Sales in Receivables Index
        - GMI: Gross Margin Index
        - AQI: Asset Quality Index
        - SGI: Sales Growth Index
        - DEPI: Depreciation Index
        - SGAI: SG&A Index
        - TATA: Total Accruals to Total Assets
        - LVGI: Leverage Index
        
        M-Score > -2.22 suggests high probability of manipulation
        
        Returns:
            M-Score value or None if insufficient data
        """
        try:
            # Current year metrics
            revenue = data.get("total_revenue") or data.get("revenue", 0)
            receivables = data.get("accounts_receivable") or data.get("receivables", 0)
            gross_profit = data.get("gross_profit", 0)
            total_assets = data.get("total_assets", 1)
            current_assets = data.get("current_assets", 0)
            ppe = data.get("property_plant_equipment") or data.get("ppe", 0)
            depreciation = data.get("depreciation", 0)
            sga = data.get("sga_expense") or data.get("operating_expenses", 0)
            net_income = data.get("net_income", 0)
            cfo = data.get("operating_cash_flow") or data.get("cash_from_operations", 0)
            total_liabilities = data.get("total_liabilities", 0)
            
            # Prior year metrics (use approximations if not available)
            # Typically would need prior year data; use current * 0.95 as estimate
            prior_revenue = data.get("prior_revenue") or revenue * 0.95
            prior_receivables = data.get("prior_receivables") or receivables * 0.95
            prior_gross_profit = data.get("prior_gross_profit") or gross_profit * 0.97
            prior_total_assets = data.get("prior_total_assets") or total_assets * 0.95
            prior_current_assets = data.get("prior_current_assets") or current_assets * 0.95
            prior_ppe = data.get("prior_ppe") or ppe * 0.95
            prior_depreciation = data.get("prior_depreciation") or depreciation * 0.95
            prior_sga = data.get("prior_sga") or sga * 0.95
            prior_liabilities = data.get("prior_liabilities") or total_liabilities * 0.95
            
            # Guard against division by zero
            if revenue <= 0 or prior_revenue <= 0 or total_assets <= 0:
                return None
            
            # Calculate indices
            # DSRI: Days Sales in Receivables Index
            dsri = (receivables / revenue) / (prior_receivables / prior_revenue) if prior_receivables > 0 else 1.0
            
            # GMI: Gross Margin Index
            gm_current = gross_profit / revenue if revenue > 0 else 0
            gm_prior = prior_gross_profit / prior_revenue if prior_revenue > 0 else gm_current
            gmi = gm_prior / gm_current if gm_current > 0 else 1.0
            
            # AQI: Asset Quality Index
            aq_current = 1 - (current_assets + ppe) / total_assets
            aq_prior = 1 - (prior_current_assets + prior_ppe) / prior_total_assets
            aqi = aq_current / aq_prior if aq_prior != 0 else 1.0
            
            # SGI: Sales Growth Index
            sgi = revenue / prior_revenue if prior_revenue > 0 else 1.0
            
            # DEPI: Depreciation Index
            dep_rate_current = depreciation / (depreciation + ppe) if (depreciation + ppe) > 0 else 0
            dep_rate_prior = prior_depreciation / (prior_depreciation + prior_ppe) if (prior_depreciation + prior_ppe) > 0 else dep_rate_current
            depi = dep_rate_prior / dep_rate_current if dep_rate_current > 0 else 1.0
            
            # SGAI: SG&A Index
            sga_rate_current = sga / revenue if revenue > 0 else 0
            sga_rate_prior = prior_sga / prior_revenue if prior_revenue > 0 else sga_rate_current
            sgai = sga_rate_current / sga_rate_prior if sga_rate_prior > 0 else 1.0
            
            # TATA: Total Accruals to Total Assets
            tata = (net_income - cfo) / total_assets
            
            # LVGI: Leverage Index
            leverage_current = total_liabilities / total_assets if total_assets > 0 else 0
            leverage_prior = prior_liabilities / prior_total_assets if prior_total_assets > 0 else leverage_current
            lvgi = leverage_current / leverage_prior if leverage_prior > 0 else 1.0
            
            # Calculate M-Score
            m_score = (
                -4.84 
                + 0.920 * dsri 
                + 0.528 * gmi 
                + 0.404 * aqi 
                + 0.892 * sgi 
                + 0.115 * depi 
                - 0.172 * sgai 
                + 4.679 * tata 
                - 0.327 * lvgi
            )
            
            return round(m_score, 2)
            
        except Exception as e:
            self.logger.debug(f"Beneish M-Score calculation failed: {e}")
            return None
    
    def calculate_altman_z_score(self, data: Dict) -> Optional[float]:
        """
        Calculate Altman Z-Score for bankruptcy prediction.
        
        For manufacturing companies:
        Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        
        Where:
        A = Working Capital / Total Assets
        B = Retained Earnings / Total Assets
        C = EBIT / Total Assets
        D = Market Value of Equity / Total Liabilities
        E = Sales / Total Assets
        
        Interpretation:
        - Z > 3.0: Safe zone
        - 1.8 < Z < 3.0: Grey zone
        - Z < 1.8: Distress zone (high bankruptcy risk)
        
        Returns:
            Z-Score value or None if insufficient data
        """
        try:
            # Extract required metrics
            total_assets = data.get("total_assets", 0)
            current_assets = data.get("current_assets", 0)
            current_liabilities = data.get("current_liabilities", 0)
            retained_earnings = data.get("retained_earnings", 0)
            ebit = data.get("ebit") or data.get("operating_income", 0)
            market_cap = data.get("market_cap", 0)
            total_liabilities = data.get("total_liabilities", 0)
            revenue = data.get("total_revenue") or data.get("revenue", 0)
            
            # Guard against division by zero
            if total_assets <= 0 or total_liabilities <= 0:
                return None
            
            # Calculate components
            working_capital = current_assets - current_liabilities
            
            a = working_capital / total_assets
            b = retained_earnings / total_assets
            c = ebit / total_assets
            d = market_cap / total_liabilities if total_liabilities > 0 else 0
            e = revenue / total_assets
            
            # Calculate Z-Score
            z_score = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e
            
            return round(z_score, 2)
            
        except Exception as e:
            self.logger.debug(f"Altman Z-Score calculation failed: {e}")
            return None
    
    def calculate_interest_coverage_ratio(self, data: Dict) -> Optional[float]:
        """
        Calculate Interest Coverage Ratio.
        
        ICR = EBIT / Interest Expense
        
        Interpretation:
        - ICR > 5.0: Very safe
        - ICR 3.0-5.0: Adequate
        - ICR < 3.0: Concern
        - ICR < 1.5: High risk
        
        Returns:
            Interest Coverage Ratio or None if insufficient data
        """
        try:
            ebit = data.get("ebit") or data.get("operating_income", 0)
            interest_expense = data.get("interest_expense", 0)
            
            # If no interest expense, company is debt-free (very safe)
            if interest_expense <= 0:
                return 100.0  # Effectively infinite
            
            icr = ebit / interest_expense
            return round(icr, 2)
            
        except Exception as e:
            self.logger.debug(f"Interest Coverage calculation failed: {e}")
            return None
    
    def calculate_piotroski_f_score(self, data: Dict) -> Optional[int]:
        """
        Calculate Piotroski F-Score for financial strength.
        
        9-point scoring system (1 point each):
        
        Profitability (4 points):
        1. Positive Net Income
        2. Positive Operating Cash Flow
        3. ROA increasing YoY
        4. Operating Cash Flow > Net Income (quality of earnings)
        
        Leverage/Liquidity (3 points):
        5. Decrease in Long-Term Debt / Total Assets
        6. Increase in Current Ratio
        7. No new equity issuance
        
        Operating Efficiency (2 points):
        8. Increase in Gross Margin
        9. Increase in Asset Turnover
        
        Interpretation:
        - 8-9: Strong
        - 5-7: Average
        - 0-4: Weak
        
        Returns:
            F-Score (0-9) or None if insufficient data
        """
        try:
            f_score = 0
            
            # Profitability
            net_income = data.get("net_income", 0)
            operating_cf = data.get("operating_cash_flow") or data.get("cash_from_operations", 0)
            roa = data.get("roa", 0)
            prior_roa = data.get("prior_roa", roa - 1)  # Estimate if not available
            
            if net_income > 0:
                f_score += 1
            if operating_cf > 0:
                f_score += 1
            if roa > prior_roa:
                f_score += 1
            if operating_cf > net_income:
                f_score += 1
            
            # Leverage/Liquidity
            long_term_debt = data.get("long_term_debt", 0)
            prior_long_term_debt = data.get("prior_long_term_debt", long_term_debt * 1.05)
            total_assets = data.get("total_assets", 1)
            prior_total_assets = data.get("prior_total_assets", total_assets * 0.95)
            current_assets = data.get("current_assets", 0)
            current_liabilities = data.get("current_liabilities", 1)
            prior_current_ratio = data.get("prior_current_ratio", 
                                          (current_assets / current_liabilities) * 0.95 if current_liabilities > 0 else 1)
            shares_outstanding = data.get("shares_outstanding", 0)
            prior_shares = data.get("prior_shares_outstanding", shares_outstanding)
            
            debt_ratio = long_term_debt / total_assets if total_assets > 0 else 0
            prior_debt_ratio = prior_long_term_debt / prior_total_assets if prior_total_assets > 0 else debt_ratio
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            
            if debt_ratio < prior_debt_ratio:
                f_score += 1
            if current_ratio > prior_current_ratio:
                f_score += 1
            if shares_outstanding <= prior_shares:
                f_score += 1
            
            # Operating Efficiency
            gross_margin = data.get("gross_margin", 0)
            prior_gross_margin = data.get("prior_gross_margin", gross_margin - 1)
            revenue = data.get("total_revenue") or data.get("revenue", 0)
            prior_revenue = data.get("prior_revenue", revenue * 0.95)
            
            asset_turnover = revenue / total_assets if total_assets > 0 else 0
            prior_asset_turnover = prior_revenue / prior_total_assets if prior_total_assets > 0 else asset_turnover
            
            if gross_margin > prior_gross_margin:
                f_score += 1
            if asset_turnover > prior_asset_turnover:
                f_score += 1
            
            return f_score
            
        except Exception as e:
            self.logger.debug(f"Piotroski F-Score calculation failed: {e}")
            return None
    
    def evaluate(self, data: Dict, ticker: str = None) -> Dict:
        """
        Comprehensive red flag evaluation.
        
        Args:
            data: Stock metrics dictionary
            ticker: Stock ticker for logging
            
        Returns:
            {
                "has_red_flags": bool,
                "veto_entry": bool,  # Should prevent attention list entry
                "red_flags": [...],
                "warnings": [...],
                "solvency": {...},
                "manipulation_risk": "low/medium/high",
                "piotroski_f_score": 0-9,
                "confidence": 0-100
            }
        """
        ticker = ticker or data.get("ticker", "UNKNOWN")
        red_flags = []
        warnings = []
        veto_entry = False
        
        # Calculate all scores
        m_score = self.calculate_beneish_m_score(data)
        z_score = self.calculate_altman_z_score(data)
        icr = self.calculate_interest_coverage_ratio(data)
        f_score = self.calculate_piotroski_f_score(data)
        
        # Evaluate Beneish M-Score (Earnings Manipulation)
        manipulation_risk = "low"
        if m_score is not None:
            if m_score > self.BENEISH_MANIPULATION_THRESHOLD:
                red_flags.append({
                    "type": "earnings_manipulation",
                    "metric": "beneish_m_score",
                    "value": m_score,
                    "threshold": self.BENEISH_MANIPULATION_THRESHOLD,
                    "severity": "high",
                    "description": f"Beneish M-Score {m_score} > {self.BENEISH_MANIPULATION_THRESHOLD} suggests potential earnings manipulation"
                })
                manipulation_risk = "high"
                veto_entry = True
            elif m_score > self.BENEISH_MANIPULATION_THRESHOLD - 0.5:
                warnings.append({
                    "type": "manipulation_warning",
                    "metric": "beneish_m_score",
                    "value": m_score,
                    "description": "Beneish M-Score approaching manipulation threshold"
                })
                manipulation_risk = "medium"
        
        # Evaluate Altman Z-Score (Bankruptcy Risk)
        solvency_zone = "unknown"
        if z_score is not None:
            if z_score < self.ALTMAN_DISTRESS_THRESHOLD:
                red_flags.append({
                    "type": "bankruptcy_risk",
                    "metric": "altman_z_score",
                    "value": z_score,
                    "threshold": self.ALTMAN_DISTRESS_THRESHOLD,
                    "severity": "high",
                    "description": f"Altman Z-Score {z_score} < {self.ALTMAN_DISTRESS_THRESHOLD} indicates distress zone"
                })
                solvency_zone = "distress"
                # Only veto if also low ICR
                if icr and icr < self.INTEREST_COVERAGE_MIN:
                    veto_entry = True
            elif z_score < self.ALTMAN_SAFE_THRESHOLD:
                warnings.append({
                    "type": "solvency_warning",
                    "metric": "altman_z_score",
                    "value": z_score,
                    "description": "Altman Z-Score in grey zone"
                })
                solvency_zone = "grey"
            else:
                solvency_zone = "safe"
        
        # Evaluate Interest Coverage
        if icr is not None and icr < self.INTEREST_COVERAGE_MIN:
            if icr < 1.5:
                red_flags.append({
                    "type": "debt_burden",
                    "metric": "interest_coverage",
                    "value": icr,
                    "threshold": 1.5,
                    "severity": "high",
                    "description": f"Interest Coverage Ratio {icr} < 1.5 indicates severe debt burden"
                })
            else:
                warnings.append({
                    "type": "debt_concern",
                    "metric": "interest_coverage",
                    "value": icr,
                    "description": f"Interest Coverage Ratio {icr} below preferred minimum of 3.0"
                })
        
        # Evaluate Piotroski F-Score
        if f_score is not None and f_score < self.PIOTROSKI_WEAK_THRESHOLD:
            warnings.append({
                "type": "weak_fundamentals",
                "metric": "piotroski_f_score",
                "value": f_score,
                "threshold": self.PIOTROSKI_WEAK_THRESHOLD,
                "description": f"Piotroski F-Score {f_score}/9 indicates weak fundamentals"
            })
        
        # Calculate solvency score (0-100)
        solvency_score = 50  # Default
        if z_score is not None:
            if z_score >= 3.0:
                solvency_score = 90
            elif z_score >= 2.5:
                solvency_score = 75
            elif z_score >= 1.8:
                solvency_score = 55
            else:
                solvency_score = 25
        
        if icr is not None:
            if icr >= 5.0:
                solvency_score = min(100, solvency_score + 10)
            elif icr < 3.0:
                solvency_score = max(0, solvency_score - 15)
        
        # Calculate confidence based on data availability
        data_points = sum([
            m_score is not None,
            z_score is not None,
            icr is not None,
            f_score is not None
        ])
        confidence = (data_points / 4) * 100
        
        return {
            "ticker": ticker,
            "has_red_flags": len(red_flags) > 0,
            "veto_entry": veto_entry,
            "red_flags": red_flags,
            "warnings": warnings,
            "solvency": {
                "altman_z_score": z_score,
                "zone": solvency_zone,
                "interest_coverage": icr,
                "score": round(solvency_score, 1)
            },
            "manipulation_risk": manipulation_risk,
            "beneish_m_score": m_score,
            "piotroski_f_score": f_score,
            "confidence": round(confidence, 1)
        }
    
    def check_solvency_guardrail(self, data: Dict) -> Tuple[bool, Dict]:
        """
        Quick solvency check for Trigger A guardrail.
        
        Passes if: Altman Z-Score > 1.8 OR Interest Coverage > 3.0
        
        Returns:
            (passed: bool, details: dict)
        """
        z_score = self.calculate_altman_z_score(data)
        icr = self.calculate_interest_coverage_ratio(data)
        
        z_ok = z_score is not None and z_score > self.ALTMAN_DISTRESS_THRESHOLD
        icr_ok = icr is not None and icr > self.INTEREST_COVERAGE_MIN
        
        passed = z_ok or icr_ok
        
        return passed, {
            "passed": passed,
            "altman_z_score": z_score,
            "interest_coverage": icr,
            "z_score_ok": z_ok,
            "icr_ok": icr_ok,
            "reason": "Solvency confirmed" if passed else "Solvency concerns - potential distress"
        }


# Export for easy importing
__all__ = ['RedFlagDetector', 'RedFlagResult']
