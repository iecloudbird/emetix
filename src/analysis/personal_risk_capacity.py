"""
Personal Risk Capacity Framework - Core Service

Thesis Core Innovation: Matching stock risk to investor capacity

Conceptual Model:
- Risk Capacity = How much can you AFFORD to lose?
- Risk Tolerance = How much volatility can you STOMACH?  
- Risk Required = How much risk NEEDED for your goals?

Emotional Buffer Formula:
    Required MoS = Base MoS × Emotional Buffer Factor
    
    Where Emotional Buffer Factor:
    - Professional: 1.0x
    - Experienced: 1.25x
    - Intermediate: 1.5x
    - Beginner: 1.75x
    - First-time: 2.0x

Phase 2 Implementation (Jan 2026)
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from config.logging_config import get_logger

logger = get_logger(__name__)


# Emotional Buffer Factors by Experience Level
EMOTIONAL_BUFFER_FACTORS = {
    'first_time': 2.0,
    'beginner': 1.75,
    'intermediate': 1.5,
    'experienced': 1.25,
    'professional': 1.0
}

# Base Margin of Safety Threshold (20% is commonly used)
BASE_MOS_THRESHOLD = 20.0

# Risk Tolerance Weights
PANIC_SELL_WEIGHTS = {
    'definitely_sell': 0.0,   # Very low tolerance
    'probably_sell': 0.25,
    'hold_nervously': 0.50,
    'hold_confidently': 0.75,
    'buy_more': 1.0           # Very high tolerance
}

# Investment Horizon Risk Adjustments
HORIZON_RISK_MULTIPLIERS = {
    'short': 0.5,       # Can only handle low-risk
    'medium': 0.75,
    'long': 1.0,
    'very_long': 1.25   # Can handle higher risk
}

# Beta ranges by profile
BETA_RANGES = {
    'conservative': {'min': 0.0, 'max': 0.8},
    'moderate': {'min': 0.5, 'max': 1.2},
    'aggressive': {'min': 0.8, 'max': 2.0}
}


@dataclass
class RiskProfile:
    """Investor Risk Profile"""
    profile_id: str
    created_at: datetime
    
    # Input data
    experience_level: str
    investment_horizon: str
    emergency_fund_months: int
    monthly_investment_pct: float
    max_loss_tolerable_pct: float
    panic_sell_response: str
    volatility_comfort: int
    current_portfolio_value: Optional[float]
    
    # Calculated scores
    risk_capacity_score: float = 0.0
    risk_tolerance_score: float = 0.0
    emotional_buffer_factor: float = 1.0
    overall_profile: str = "moderate"


class PersonalRiskCapacityService:
    """
    Personal Risk Capacity Framework Service
    
    Core Thesis Innovation:
    - Match stock risk to investor capacity
    - Apply Emotional Buffer based on experience
    - Provide position sizing recommendations
    """
    
    def __init__(self):
        self.profiles: Dict[str, RiskProfile] = {}
        logger.info("PersonalRiskCapacityService initialized")
    
    def assess_risk_profile(
        self,
        experience_level: str,
        investment_horizon: str,
        emergency_fund_months: int,
        monthly_investment_pct: float,
        max_loss_tolerable_pct: float,
        panic_sell_response: str,
        volatility_comfort: int,
        current_portfolio_value: Optional[float] = None
    ) -> Dict:
        """
        Assess investor's personal risk profile
        
        Returns comprehensive risk assessment with:
        - Risk Capacity Score (financial ability)
        - Risk Tolerance Score (emotional ability)
        - Emotional Buffer Factor
        - Overall Profile Classification
        - Suitable Beta Range
        - Recommendations
        """
        profile_id = str(uuid.uuid4())[:8]
        created_at = datetime.now()
        
        # Calculate Risk Capacity (financial ability to absorb losses)
        risk_capacity = self._calculate_risk_capacity(
            emergency_fund_months,
            monthly_investment_pct,
            max_loss_tolerable_pct
        )
        
        # Calculate Risk Tolerance (emotional ability to handle volatility)
        risk_tolerance = self._calculate_risk_tolerance(
            panic_sell_response,
            volatility_comfort,
            investment_horizon
        )
        
        # Calculate Emotional Buffer
        emotional_buffer = self._calculate_emotional_buffer(experience_level)
        
        # Determine Overall Profile
        avg_score = (risk_capacity['score'] + risk_tolerance['score']) / 2
        overall_profile = self._classify_profile(avg_score)
        
        # Get suitable beta range
        beta_range = BETA_RANGES.get(overall_profile, BETA_RANGES['moderate'])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_profile,
            risk_capacity,
            risk_tolerance,
            emotional_buffer,
            experience_level
        )
        
        # Store profile
        profile = RiskProfile(
            profile_id=profile_id,
            created_at=created_at,
            experience_level=experience_level,
            investment_horizon=investment_horizon,
            emergency_fund_months=emergency_fund_months,
            monthly_investment_pct=monthly_investment_pct,
            max_loss_tolerable_pct=max_loss_tolerable_pct,
            panic_sell_response=panic_sell_response,
            volatility_comfort=volatility_comfort,
            current_portfolio_value=current_portfolio_value,
            risk_capacity_score=risk_capacity['score'],
            risk_tolerance_score=risk_tolerance['score'],
            emotional_buffer_factor=emotional_buffer['factor'],
            overall_profile=overall_profile
        )
        self.profiles[profile_id] = profile
        
        return {
            'profile_id': profile_id,
            'created_at': created_at.isoformat(),
            'risk_capacity': risk_capacity,
            'risk_tolerance': risk_tolerance,
            'emotional_buffer': emotional_buffer,
            'overall_risk_profile': overall_profile,
            'suitable_beta_range': beta_range,
            'recommendations': recommendations
        }
    
    def _calculate_risk_capacity(
        self,
        emergency_fund_months: int,
        monthly_investment_pct: float,
        max_loss_tolerable_pct: float
    ) -> Dict:
        """
        Calculate Risk Capacity Score (financial ability)
        
        Factors:
        - Emergency fund buffer (more = higher capacity)
        - Monthly investment rate (higher = more capacity)
        - Maximum tolerable loss (higher = more capacity)
        """
        # Emergency fund score (0-40 points)
        # 6+ months is considered adequate
        ef_score = min(40, emergency_fund_months * 6.67)
        
        # Monthly investment score (0-30 points)
        # Higher savings rate = more capacity
        invest_score = min(30, monthly_investment_pct * 1.5)
        
        # Max loss score (0-30 points)
        # Ability to absorb larger losses
        loss_score = min(30, max_loss_tolerable_pct * 0.6)
        
        total_score = ef_score + invest_score + loss_score
        
        # Calculate max position % based on capacity
        max_position_pct = min(20, 5 + (total_score / 10))
        
        reasoning_parts = []
        if emergency_fund_months >= 6:
            reasoning_parts.append(f"{emergency_fund_months}-month emergency fund provides good buffer")
        else:
            reasoning_parts.append(f"Consider building emergency fund to 6+ months")
        
        if monthly_investment_pct >= 15:
            reasoning_parts.append(f"{monthly_investment_pct}% savings rate shows strong capacity")
        
        return {
            'score': round(total_score, 1),
            'max_loss_affordable': max_loss_tolerable_pct,
            'max_position_pct': round(max_position_pct, 1),
            'reasoning': "; ".join(reasoning_parts) or "Standard risk capacity"
        }
    
    def _calculate_risk_tolerance(
        self,
        panic_sell_response: str,
        volatility_comfort: int,
        investment_horizon: str
    ) -> Dict:
        """
        Calculate Risk Tolerance Score (emotional ability)
        
        Factors:
        - Panic sell tendency
        - Volatility comfort
        - Investment horizon
        """
        # Panic response score (0-40 points)
        panic_weight = PANIC_SELL_WEIGHTS.get(panic_sell_response, 0.5)
        panic_score = panic_weight * 40
        
        # Volatility comfort score (0-30 points)
        vol_score = (volatility_comfort - 1) * 7.5  # 1-5 scale to 0-30
        
        # Horizon adjustment (0-30 points)
        horizon_mult = HORIZON_RISK_MULTIPLIERS.get(investment_horizon, 0.75)
        horizon_score = horizon_mult * 30
        
        total_score = panic_score + vol_score + horizon_score
        
        # Determine volatility tolerance category
        if total_score < 35:
            vol_tolerance = "Low"
            panic_risk = "High"
        elif total_score < 65:
            vol_tolerance = "Medium"
            panic_risk = "Medium"
        else:
            vol_tolerance = "High"
            panic_risk = "Low"
        
        reasoning = f"Based on {panic_sell_response.replace('_', ' ')} response to 30% drops"
        if investment_horizon == 'very_long':
            reasoning += "; long horizon supports higher tolerance"
        
        return {
            'score': round(total_score, 1),
            'volatility_tolerance': vol_tolerance,
            'panic_risk': panic_risk,
            'reasoning': reasoning
        }
    
    def _calculate_emotional_buffer(self, experience_level: str) -> Dict:
        """
        Calculate Emotional Buffer Factor
        
        First-time investors need 2x the margin of safety to account
        for their tendency to panic sell during normal market volatility.
        """
        factor = EMOTIONAL_BUFFER_FACTORS.get(experience_level, 1.5)
        base_mos = BASE_MOS_THRESHOLD
        adjusted_mos = base_mos * factor
        
        if experience_level == 'first_time':
            reasoning = "First-time investors should require 2x margin of safety to avoid panic selling"
        elif experience_level == 'beginner':
            reasoning = "Beginners benefit from extra safety margin while building experience"
        elif experience_level == 'intermediate':
            reasoning = "Moderate buffer applied for intermediate experience"
        elif experience_level == 'experienced':
            reasoning = "Experienced investors can accept lower margins"
        else:
            reasoning = "Professional-level investors can use standard valuation thresholds"
        
        return {
            'factor': factor,
            'base_mos_threshold': base_mos,
            'adjusted_mos_threshold': adjusted_mos,
            'reasoning': reasoning
        }
    
    def _classify_profile(self, avg_score: float) -> str:
        """Classify overall risk profile"""
        if avg_score < 40:
            return 'conservative'
        elif avg_score < 70:
            return 'moderate'
        else:
            return 'aggressive'
    
    def _generate_recommendations(
        self,
        profile: str,
        capacity: Dict,
        tolerance: Dict,
        buffer: Dict,
        experience: str
    ) -> List[str]:
        """Generate personalized recommendations"""
        recs = []
        
        # Profile-based recommendations
        if profile == 'conservative':
            recs.append("Focus on low-beta stocks (beta < 0.8) for portfolio stability")
            recs.append("Prioritize dividend-paying stocks for income")
            recs.append("Consider maximum 5-10% allocation per position")
        elif profile == 'aggressive':
            recs.append("Can consider higher-beta growth stocks for potential returns")
            recs.append("Ensure adequate emergency fund before aggressive allocation")
        
        # Emotional buffer recommendation
        if buffer['factor'] > 1.5:
            recs.append(f"Only consider stocks with MoS > {buffer['adjusted_mos_threshold']:.0f}% (your emotional buffer)")
        
        # Experience-based
        if experience in ['first_time', 'beginner']:
            recs.append("Start with well-established companies before exploring smaller stocks")
            recs.append("Paper trade or use small positions to build experience")
        
        # Risk tolerance
        if tolerance['panic_risk'] == 'High':
            recs.append("Set stop-loss orders to prevent emotional selling decisions")
            recs.append("Consider dollar-cost averaging to reduce timing stress")
        
        return recs
    
    def calculate_position_sizing(
        self,
        stock_data: Dict,
        portfolio_value: float,
        profile_id: Optional[str] = None,
        override_capacity_score: Optional[float] = None
    ) -> Dict:
        """
        Calculate personalized position sizing recommendation
        
        Based on:
        - Stock's margin of safety (edge)
        - Stock's beta/volatility (risk)
        - Investor's risk capacity (constraint)
        - Emotional buffer (adjustment)
        """
        # Get profile or use defaults
        if profile_id and profile_id in self.profiles:
            profile = self.profiles[profile_id]
            capacity_score = profile.risk_capacity_score
            buffer_factor = profile.emotional_buffer_factor
        else:
            capacity_score = override_capacity_score or 50.0  # Default moderate
            buffer_factor = 1.5  # Default intermediate
        
        # Extract stock data
        ticker = stock_data.get('ticker', 'UNKNOWN')
        current_price = stock_data.get('current_price', 0)
        mos = stock_data.get('margin_of_safety', 0)
        beta = stock_data.get('beta', 1.0)
        volatility = stock_data.get('volatility', 30)
        valuation_signal = stock_data.get('recommendation', 'HOLD')
        
        # Adjusted MoS threshold for this investor
        adjusted_mos_threshold = BASE_MOS_THRESHOLD * buffer_factor
        
        # Position sizing calculation
        risk_factors = []
        
        # Base max position from risk capacity (5-15%)
        base_max_pct = 5 + (capacity_score / 100) * 10
        
        # Adjust for stock characteristics
        if mos < adjusted_mos_threshold:
            # Below personal threshold - reduce position or avoid
            if mos < 0:
                max_position_pct = 0
                risk_factors.append(f"Stock overvalued by {abs(mos):.1f}%")
            else:
                max_position_pct = base_max_pct * 0.5
                risk_factors.append(f"MoS {mos:.1f}% below your {adjusted_mos_threshold:.0f}% threshold")
        else:
            # Good margin of safety
            mos_bonus = min(1.5, 1 + (mos - adjusted_mos_threshold) / 50)
            max_position_pct = base_max_pct * mos_bonus
            
        # Beta adjustment
        if beta > 1.3:
            max_position_pct *= 0.75
            risk_factors.append(f"High beta ({beta:.2f}) - reduced position")
        elif beta < 0.7:
            max_position_pct *= 1.2
            risk_factors.append(f"Low beta ({beta:.2f}) - suitable for stability")
        
        # Volatility adjustment
        if volatility > 40:
            max_position_pct *= 0.8
            risk_factors.append(f"High volatility ({volatility:.0f}%)")
        
        # Cap at 20% maximum per position
        max_position_pct = min(20, max(0, max_position_pct))
        
        # Calculate dollar amounts
        max_position_value = portfolio_value * (max_position_pct / 100)
        max_shares = int(max_position_value / current_price) if current_price > 0 else 0
        
        # Confidence based on data quality
        confidence = 0.8 if mos > 0 else 0.5
        
        methodology = f"Kelly-inspired sizing with emotional buffer ({buffer_factor}x)"
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'margin_of_safety': mos,
            'stock_beta': beta,
            'stock_volatility': volatility,
            'valuation_signal': valuation_signal,
            'max_position_pct': round(max_position_pct, 2),
            'max_position_value': round(max_position_value, 2),
            'max_shares': max_shares,
            'sizing_methodology': methodology,
            'risk_factors': risk_factors,
            'confidence': confidence
        }
    
    def get_profile(self, profile_id: str) -> Optional[RiskProfile]:
        """Get stored risk profile"""
        return self.profiles.get(profile_id)
    
    def filter_stocks_for_profile(
        self,
        stocks: List[Dict],
        profile_id: str
    ) -> List[Dict]:
        """
        Filter stock list to only show suitable stocks for profile
        
        Returns stocks that match:
        - Beta within suitable range
        - MoS above adjusted threshold
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return stocks
        
        beta_range = BETA_RANGES.get(profile.overall_profile, BETA_RANGES['moderate'])
        mos_threshold = BASE_MOS_THRESHOLD * profile.emotional_buffer_factor
        
        suitable = []
        for stock in stocks:
            beta = stock.get('beta', 1.0)
            mos = stock.get('margin_of_safety', 0)
            
            # Check beta range
            if not (beta_range['min'] <= beta <= beta_range['max']):
                continue
            
            # Check MoS threshold
            if mos < mos_threshold:
                continue
            
            # Add profile suitability metadata
            stock['profile_suitability'] = {
                'suitable': True,
                'beta_in_range': True,
                'mos_above_threshold': True,
                'adjusted_mos_threshold': mos_threshold
            }
            suitable.append(stock)
        
        return suitable    
    def annotate_stocks_with_suitability(
        self,
        stocks: List[Dict],
        profile_id: str
    ) -> List[Dict]:
        """
        Annotate all stocks with suitability indicators for a profile
        
        Unlike filter_stocks_for_profile, this returns ALL stocks but adds
        suitability metadata to each. Frontend can use this to:
        - Highlight suitable stocks
        - Dim unsuitable stocks
        - Show warning icons
        """
        profile = self.get_profile(profile_id)
        if not profile:
            # No profile - return stocks unchanged
            return stocks
        
        beta_range = BETA_RANGES.get(profile.overall_profile, BETA_RANGES['moderate'])
        mos_threshold = BASE_MOS_THRESHOLD * profile.emotional_buffer_factor
        
        annotated = []
        for stock in stocks:
            stock = stock.copy()  # Don't modify original
            
            beta = stock.get('beta', 1.0)
            mos = stock.get('margin_of_safety', 0)
            
            # Check suitability criteria
            beta_in_range = beta_range['min'] <= beta <= beta_range['max']
            mos_above_threshold = mos >= mos_threshold
            
            # Calculate suitability score
            if beta_in_range and mos_above_threshold:
                suitability = 'EXCELLENT'
            elif beta_in_range or mos_above_threshold:
                suitability = 'MODERATE'
            else:
                suitability = 'POOR'
            
            # Add suitability metadata
            stock['profile_suitability'] = {
                'rating': suitability,
                'suitable': beta_in_range and mos_above_threshold,
                'beta_in_range': beta_in_range,
                'mos_above_threshold': mos_above_threshold,
                'profile_beta_range': beta_range,
                'adjusted_mos_threshold': round(mos_threshold, 1),
                'reasons': []
            }
            
            # Add reasons for suitability issues
            if not beta_in_range:
                if beta < beta_range['min']:
                    stock['profile_suitability']['reasons'].append(
                        f"Beta {beta:.2f} below your minimum {beta_range['min']}"
                    )
                else:
                    stock['profile_suitability']['reasons'].append(
                        f"Beta {beta:.2f} above your maximum {beta_range['max']}"
                    )
            
            if not mos_above_threshold:
                stock['profile_suitability']['reasons'].append(
                    f"MoS {mos:.1f}% below your threshold of {mos_threshold:.0f}%"
                )
            
            annotated.append(stock)
        
        return annotated