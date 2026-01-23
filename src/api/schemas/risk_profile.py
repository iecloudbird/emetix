"""
Personal Risk Capacity Framework - API Schemas

Thesis Core Innovation: Matching stock risk to investor capacity
- Risk Capacity: How much can you AFFORD to lose?
- Risk Tolerance: How much volatility can you STOMACH?
- Emotional Buffer: Extra MoS needed for inexperienced investors

Phase 2 Implementation (Jan 2026)
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


class ExperienceLevel(str, Enum):
    """Investor experience classification"""
    FIRST_TIME = "first_time"      # Emotional Buffer: 2.0x
    BEGINNER = "beginner"          # Emotional Buffer: 1.75x
    INTERMEDIATE = "intermediate"  # Emotional Buffer: 1.5x
    EXPERIENCED = "experienced"    # Emotional Buffer: 1.25x
    PROFESSIONAL = "professional"  # Emotional Buffer: 1.0x


class InvestmentHorizon(str, Enum):
    """Investment time horizon"""
    SHORT = "short"          # < 1 year
    MEDIUM = "medium"        # 1-5 years
    LONG = "long"            # 5-10 years
    VERY_LONG = "very_long"  # 10+ years


class PanicSellResponse(str, Enum):
    """Response to portfolio drop scenario"""
    DEFINITELY_SELL = "definitely_sell"      # Very low tolerance
    PROBABLY_SELL = "probably_sell"          # Low tolerance
    HOLD_NERVOUSLY = "hold_nervously"        # Medium tolerance
    HOLD_CONFIDENTLY = "hold_confidently"    # High tolerance
    BUY_MORE = "buy_more"                    # Very high tolerance


class RiskQuestionnaireRequest(BaseModel):
    """Personal Risk Questionnaire Input
    
    Questions designed to assess:
    1. Risk Capacity (financial ability to absorb losses)
    2. Risk Tolerance (emotional ability to handle volatility)
    3. Experience Level (for Emotional Buffer calculation)
    """
    
    # Demographics & Experience
    experience_level: ExperienceLevel = Field(
        description="Your investing experience level"
    )
    investment_horizon: InvestmentHorizon = Field(
        description="How long do you plan to hold investments?"
    )
    
    # Risk Capacity Questions
    emergency_fund_months: int = Field(
        ge=0, le=36,
        description="How many months of expenses do you have in emergency savings?"
    )
    monthly_investment_pct: float = Field(
        ge=0, le=100,
        description="What percentage of your income can you invest monthly? (0-100)"
    )
    max_loss_tolerable_pct: float = Field(
        ge=0, le=100,
        description="What maximum percentage loss can you financially survive? (0-100)"
    )
    
    # Risk Tolerance Questions (Behavioral)
    panic_sell_response: PanicSellResponse = Field(
        description="If your portfolio dropped 30%, what would you do?"
    )
    volatility_comfort: int = Field(
        ge=1, le=5,
        description="Rate your comfort with daily price swings (1=hate, 5=love)"
    )
    
    # Optional: Existing portfolio context
    current_portfolio_value: Optional[float] = Field(
        default=None,
        description="Current portfolio value (optional, for position sizing)"
    )


class EmotionalBufferResult(BaseModel):
    """Emotional Buffer calculation result"""
    factor: float = Field(description="Multiplier for Margin of Safety threshold")
    base_mos_threshold: float = Field(description="Base MoS threshold (e.g., 20%)")
    adjusted_mos_threshold: float = Field(description="Adjusted MoS threshold after buffer")
    reasoning: str = Field(description="Explanation for the buffer factor")


class RiskCapacityScore(BaseModel):
    """Risk Capacity assessment (financial ability)"""
    score: float = Field(ge=0, le=100, description="Risk capacity score 0-100")
    max_loss_affordable: float = Field(description="Maximum $ loss you can absorb")
    max_position_pct: float = Field(description="Maximum position size as % of portfolio")
    reasoning: str


class RiskToleranceScore(BaseModel):
    """Risk Tolerance assessment (emotional ability)"""
    score: float = Field(ge=0, le=100, description="Risk tolerance score 0-100")
    volatility_tolerance: str = Field(description="Low/Medium/High volatility tolerance")
    panic_risk: str = Field(description="Low/Medium/High risk of panic selling")
    reasoning: str


class RiskProfileResponse(BaseModel):
    """Complete Personal Risk Profile Response"""
    
    # Profile Summary
    profile_id: str = Field(description="Unique profile identifier")
    created_at: str = Field(description="Profile creation timestamp")
    
    # Core Scores
    risk_capacity: RiskCapacityScore
    risk_tolerance: RiskToleranceScore
    emotional_buffer: EmotionalBufferResult
    
    # Aggregate Classification
    overall_risk_profile: str = Field(
        description="Conservative / Moderate / Aggressive"
    )
    suitable_beta_range: Dict[str, float] = Field(
        description="Recommended beta range for this profile"
    )
    
    # Recommendations
    recommendations: List[str] = Field(
        description="Personalized investment recommendations"
    )


class PositionSizingRequest(BaseModel):
    """Request for position sizing recommendation"""
    ticker: str = Field(description="Stock ticker symbol")
    profile_id: Optional[str] = Field(
        default=None, 
        description="Risk profile ID (if previously saved)"
    )
    portfolio_value: float = Field(
        gt=0, 
        description="Total portfolio value"
    )
    current_allocation_pct: float = Field(
        ge=0, le=100, default=0,
        description="Current allocation to this stock (%)"
    )


class PositionSizingResponse(BaseModel):
    """Position sizing recommendation
    
    Based on:
    - Stock's margin of safety
    - Stock's beta/volatility
    - Investor's risk capacity
    - Investor's emotional buffer
    """
    ticker: str
    current_price: float
    
    # Stock characteristics
    margin_of_safety: float
    stock_beta: float
    stock_volatility: float
    valuation_signal: str = Field(description="BUY/HOLD/SELL signal")
    
    # Position sizing recommendations
    max_position_pct: float = Field(
        description="Maximum recommended position as % of portfolio"
    )
    max_position_value: float = Field(
        description="Maximum recommended position in $"
    )
    max_shares: int = Field(
        description="Maximum recommended number of shares"
    )
    
    # Reasoning
    sizing_methodology: str
    risk_factors: List[str]
    confidence: float = Field(ge=0, le=1)


class EnhancedStockResponse(BaseModel):
    """Extended stock response with position sizing meta-info
    
    For modular frontend implementation:
    - Base stock data in main response
    - Position sizing in separate field
    - Tooltips can render from this data
    """
    
    # Base stock data (existing)
    ticker: str
    company_name: str
    current_price: float
    fair_value: float
    margin_of_safety: float
    valuation_score: float
    recommendation: str  # BUY, HOLD, SELL
    risk_level: str
    
    # Position Sizing Meta (NEW - for tooltip/hover)
    position_sizing_meta: Optional[Dict] = Field(
        default=None,
        description="Position sizing info for tooltip display"
    )
    
    # Emotional Buffer Context (NEW)
    emotional_buffer_note: Optional[str] = Field(
        default=None,
        description="Note about MoS adjustment for investor experience"
    )
    
    # Suitability for Profile (NEW)
    profile_suitability: Optional[Dict] = Field(
        default=None,
        description="How suitable this stock is for the investor's profile"
    )
