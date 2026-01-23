"""Test Personal Risk Capacity Framework"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.personal_risk_capacity import PersonalRiskCapacityService

service = PersonalRiskCapacityService()

# Test risk profile assessment
result = service.assess_risk_profile(
    experience_level='beginner',
    investment_horizon='long',
    emergency_fund_months=6,
    monthly_investment_pct=15,
    max_loss_tolerable_pct=25,
    panic_sell_response='hold_nervously',
    volatility_comfort=3,
    current_portfolio_value=50000
)

print('=== Risk Profile Assessment ===')
print(f"Profile ID: {result['profile_id']}")
print(f"Risk Capacity: {result['risk_capacity']['score']}/100")
print(f"Risk Tolerance: {result['risk_tolerance']['score']}/100")
print(f"Emotional Buffer: {result['emotional_buffer']['factor']}x")
print(f"Adjusted MoS Threshold: {result['emotional_buffer']['adjusted_mos_threshold']}%")
print(f"Overall Profile: {result['overall_risk_profile']}")
print(f"Beta Range: {result['suitable_beta_range']}")
print(f"\nRecommendations:")
for rec in result['recommendations']:
    print(f"  - {rec}")
