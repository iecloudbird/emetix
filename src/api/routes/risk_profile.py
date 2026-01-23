"""
Personal Risk Capacity API Routes

Phase 2 Implementation - Thesis Core Innovation

Endpoints:
- POST /risk-profile: Create/assess risk profile from questionnaire
- GET /risk-profile/{id}: Get stored risk profile
- POST /position-sizing: Get position recommendation for a stock
- GET /stocks/suitable: Get stocks filtered by risk profile
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional

from src.api.schemas.risk_profile import (
    RiskQuestionnaireRequest,
    RiskProfileResponse,
    PositionSizingRequest,
    PositionSizingResponse,
    EnhancedStockResponse
)
from src.analysis.personal_risk_capacity import PersonalRiskCapacityService
from src.analysis.stock_screener import StockScreener

router = APIRouter(prefix="/api/risk-profile", tags=["Personal Risk Capacity"])

# Singleton service instance
_risk_service: Optional[PersonalRiskCapacityService] = None
_screener: Optional[StockScreener] = None


def get_risk_service() -> PersonalRiskCapacityService:
    """Get or create risk capacity service"""
    global _risk_service
    if _risk_service is None:
        _risk_service = PersonalRiskCapacityService()
    return _risk_service


def get_screener() -> StockScreener:
    """Get or create stock screener"""
    global _screener
    if _screener is None:
        _screener = StockScreener(
            use_extended_universe=True,
            enable_lstm=True,
            enable_consensus=True
        )
    return _screener


@router.post(
    "/assess",
    summary="Assess Personal Risk Profile",
    description="""
    Submit personal risk questionnaire to receive:
    - Risk Capacity Score (financial ability to absorb losses)
    - Risk Tolerance Score (emotional ability to handle volatility)
    - Emotional Buffer Factor (MoS multiplier based on experience)
    - Overall Risk Profile (Conservative/Moderate/Aggressive)
    - Personalized Recommendations
    
    **Thesis Innovation**: This endpoint implements the Personal Risk Capacity Framework,
    matching stock risk to individual investor capacity.
    """
)
async def assess_risk_profile(questionnaire: RiskQuestionnaireRequest):
    """Create and assess a personal risk profile"""
    service = get_risk_service()
    
    try:
        result = service.assess_risk_profile(
            experience_level=questionnaire.experience_level.value,
            investment_horizon=questionnaire.investment_horizon.value,
            emergency_fund_months=questionnaire.emergency_fund_months,
            monthly_investment_pct=questionnaire.monthly_investment_pct,
            max_loss_tolerable_pct=questionnaire.max_loss_tolerable_pct,
            panic_sell_response=questionnaire.panic_sell_response.value,
            volatility_comfort=questionnaire.volatility_comfort,
            current_portfolio_value=questionnaire.current_portfolio_value
        )
        
        return {
            "status": "success",
            "data": result,
            "meta": {
                "thesis_concept": "Personal Risk Capacity Framework",
                "emotional_buffer_note": f"Your MoS threshold: {result['emotional_buffer']['adjusted_mos_threshold']:.0f}% (base 20% × {result['emotional_buffer']['factor']}x buffer)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/profile/{profile_id}",
    summary="Get Risk Profile",
    description="Retrieve a previously created risk profile by ID"
)
async def get_risk_profile(profile_id: str):
    """Get stored risk profile"""
    service = get_risk_service()
    profile = service.get_profile(profile_id)
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
    
    return {
        "status": "success",
        "data": {
            "profile_id": profile.profile_id,
            "created_at": profile.created_at.isoformat(),
            "experience_level": profile.experience_level,
            "risk_capacity_score": profile.risk_capacity_score,
            "risk_tolerance_score": profile.risk_tolerance_score,
            "emotional_buffer_factor": profile.emotional_buffer_factor,
            "overall_profile": profile.overall_profile
        }
    }


@router.post(
    "/position-sizing",
    summary="Get Position Sizing Recommendation",
    description="""
    Calculate personalized position sizing for a stock based on:
    - Stock's margin of safety (your edge)
    - Stock's beta/volatility (risk level)
    - Your risk capacity (financial constraint)
    - Your emotional buffer (experience adjustment)
    
    Returns:
    - Maximum recommended position as % of portfolio
    - Maximum dollar amount to invest
    - Maximum number of shares to buy
    - Risk factors to consider
    
    **Frontend Usage**: This data can be displayed as tooltip on hover over the signal column.
    """
)
async def get_position_sizing(request: PositionSizingRequest):
    """Calculate position sizing recommendation"""
    service = get_risk_service()
    screener = get_screener()
    
    try:
        # Get stock data
        stock_data = screener.analyze_single_stock(request.ticker)
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"Stock {request.ticker} not found")
        
        # Calculate position sizing
        result = service.calculate_position_sizing(
            stock_data=stock_data,
            portfolio_value=request.portfolio_value,
            profile_id=request.profile_id
        )
        
        return {
            "status": "success",
            "data": result,
            "meta": {
                "tooltip_text": f"Max position: {result['max_position_pct']:.1f}% (${result['max_position_value']:,.0f})",
                "frontend_hint": "Display in tooltip on signal column hover"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/suitable-stocks",
    summary="Get Stocks Suitable for Profile",
    description="""
    Get a filtered list of stocks that match the investor's risk profile:
    - Beta within suitable range for profile type
    - Margin of Safety above adjusted threshold (with emotional buffer)
    
    **Thesis Innovation**: Personalizes the watchlist to show only stocks
    suitable for THIS investor's risk capacity.
    """
)
async def get_suitable_stocks(
    profile_id: str = Query(description="Risk profile ID from /assess endpoint"),
    n: int = Query(default=10, ge=1, le=50, description="Number of stocks to return"),
    rescan: bool = Query(default=False, description="Force rescan, ignoring cache")
):
    """Get stocks filtered by risk profile suitability"""
    service = get_risk_service()
    screener = get_screener()
    
    profile = service.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
    
    try:
        # Get full watchlist
        result = screener.get_undervalued_stocks(
            n=50,  # Get more, then filter
            rescan=rescan
        )
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail="Failed to get stocks")
        
        stocks = result['data']
        
        # Filter for profile suitability
        suitable_stocks = service.filter_stocks_for_profile(stocks, profile_id)
        
        # Limit to requested count
        suitable_stocks = suitable_stocks[:n]
        
        return {
            "status": "success",
            "profile": {
                "id": profile_id,
                "type": profile.overall_profile,
                "emotional_buffer": profile.emotional_buffer_factor,
                "mos_threshold": 20.0 * profile.emotional_buffer_factor
            },
            "data": suitable_stocks,
            "total": len(suitable_stocks),
            "filtered_from": len(stocks),
            "meta": {
                "thesis_concept": "Personalized Watchlist - stocks matching YOUR risk capacity",
                "filter_applied": f"Beta {profile.overall_profile} range, MoS > {20 * profile.emotional_buffer_factor:.0f}%"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/methodology",
    summary="Personal Risk Capacity Methodology",
    description="Explain the thesis innovation behind Personal Risk Capacity Framework"
)
async def get_methodology():
    """Return methodology documentation"""
    return {
        "status": "success",
        "methodology": {
            "name": "Personal Risk Capacity Framework",
            "thesis_contribution": "Matching stock risk to investor capacity - not just 'is this stock risky?' but 'is this stock risky FOR YOU?'",
            
            "core_concepts": {
                "risk_capacity": {
                    "definition": "How much can you AFFORD to lose?",
                    "factors": ["Emergency fund months", "Monthly savings rate", "Maximum tolerable loss %"],
                    "output": "Maximum position size as % of portfolio"
                },
                "risk_tolerance": {
                    "definition": "How much volatility can you STOMACH?",
                    "factors": ["Panic sell tendency", "Volatility comfort (1-5)", "Investment horizon"],
                    "output": "Suitable beta range for portfolio"
                },
                "emotional_buffer": {
                    "definition": "Extra margin of safety for inexperienced investors",
                    "formula": "Required MoS = Base MoS (20%) × Emotional Buffer Factor",
                    "factors": {
                        "first_time": "2.0x (need 40% MoS)",
                        "beginner": "1.75x (need 35% MoS)",
                        "intermediate": "1.5x (need 30% MoS)",
                        "experienced": "1.25x (need 25% MoS)",
                        "professional": "1.0x (need 20% MoS)"
                    },
                    "reasoning": "First-time investors tend to panic sell during normal volatility; higher MoS provides emotional cushion"
                }
            },
            
            "position_sizing": {
                "methodology": "Kelly-inspired with personal adjustment",
                "formula": "max_position = min(kelly_fraction, personal_limit) × emotional_buffer",
                "inputs": ["Margin of Safety (edge)", "Beta/Volatility (risk)", "Risk Capacity (constraint)"],
                "output": "Maximum recommended position in % and $"
            },
            
            "social_impact": {
                "democratization": "Institutional-grade risk assessment for retail investors",
                "harm_reduction": "Prevents overexposure to unsuitable stocks",
                "financial_literacy": "Educational explanations for all metrics",
                "behavioral_nudges": "Emotional buffer concept helps investors avoid panic selling"
            },
            
            "frontend_integration": {
                "questionnaire": "Simple 7-question form to create risk profile",
                "personalized_watchlist": "Only show stocks matching risk profile",
                "position_sizing_tooltip": "Hover over signal column for sizing recommendation",
                "suitability_indicator": "Icon showing if stock matches profile"
            }
        }
    }
