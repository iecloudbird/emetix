"""
Free Cash Flow-Based DCF Model for Intrinsic Valuation
Implements sophisticated FCF projection with terminal value calculation
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from config.logging_config import get_logger

logger = get_logger(__name__)


class FCFDCFModel:
    """
    Free Cash Flow Discounted Cash Flow Model
    
    Uses projected FCF for accurate fair value estimation
    Formula: Fair Value = Sum(FCF_t / (1+WACC)^t) + Terminal Value / (1+WACC)^n
    """
    
    def __init__(
        self,
        discount_rate: float = 0.10,
        terminal_growth_rate: float = 0.025,
        projection_years: int = 5
    ):
        """
        Args:
            discount_rate: Weighted Average Cost of Capital (WACC) - default 10%
            terminal_growth_rate: Perpetual growth rate - default 2.5%
            projection_years: Number of years to project FCF - default 5
        """
        self.discount_rate = discount_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.projection_years = projection_years
        self.logger = logger
        
        # Validation
        if terminal_growth_rate >= discount_rate:
            self.logger.warning(
                f"Terminal growth ({terminal_growth_rate}) >= discount rate ({discount_rate}). "
                f"Setting terminal growth to {discount_rate * 0.8:.3f}"
            )
            self.terminal_growth_rate = discount_rate * 0.8
    
    def calculate_intrinsic_value(
        self,
        current_fcf: float,
        fcf_growth_rates: List[float],
        shares_outstanding: float,
        net_debt: float = 0
    ) -> Dict:
        """
        Calculate intrinsic value per share using FCF projections
        
        Args:
            current_fcf: Current year Free Cash Flow
            fcf_growth_rates: List of growth rates for projection years
            shares_outstanding: Total shares outstanding
            net_debt: Total debt - cash (negative if net cash)
            
        Returns:
            Dictionary with valuation details
        """
        try:
            # Validate inputs
            if current_fcf <= 0:
                self.logger.warning(f"Negative/zero FCF: {current_fcf}. Using conservative approach.")
                current_fcf = abs(current_fcf) * 0.5
            
            # Extend growth rates if needed
            if len(fcf_growth_rates) < self.projection_years:
                # Use last provided rate or default conservative 3%
                default_rate = fcf_growth_rates[-1] if fcf_growth_rates else 0.03
                fcf_growth_rates.extend(
                    [default_rate] * (self.projection_years - len(fcf_growth_rates))
                )
            
            # Project FCF for each year
            projected_fcf = []
            fcf = current_fcf
            
            for year in range(self.projection_years):
                growth_rate = fcf_growth_rates[year]
                fcf = fcf * (1 + growth_rate)
                projected_fcf.append(fcf)
            
            # Calculate present value of projected FCF
            pv_fcf = []
            for year, fcf in enumerate(projected_fcf, start=1):
                pv = fcf / ((1 + self.discount_rate) ** year)
                pv_fcf.append(pv)
            
            # Calculate Terminal Value (Gordon Growth Model)
            terminal_fcf = projected_fcf[-1] * (1 + self.terminal_growth_rate)
            terminal_value = terminal_fcf / (self.discount_rate - self.terminal_growth_rate)
            
            # Present value of terminal value
            pv_terminal_value = terminal_value / ((1 + self.discount_rate) ** self.projection_years)
            
            # Enterprise Value = Sum of PV(FCF) + PV(Terminal Value)
            enterprise_value = sum(pv_fcf) + pv_terminal_value
            
            # Equity Value = Enterprise Value - Net Debt
            equity_value = enterprise_value - net_debt
            
            # Intrinsic Value per Share
            intrinsic_value_per_share = equity_value / shares_outstanding
            
            return {
                'intrinsic_value_per_share': intrinsic_value_per_share,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'projected_fcf': projected_fcf,
                'pv_fcf': pv_fcf,
                'terminal_value': terminal_value,
                'pv_terminal_value': pv_terminal_value,
                'discount_rate': self.discount_rate,
                'terminal_growth_rate': self.terminal_growth_rate,
                'shares_outstanding': shares_outstanding,
                'net_debt': net_debt,
                'method': 'FCF_DCF'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating FCF DCF: {str(e)}")
            return None
    
    def calculate_with_market_price(
        self,
        current_fcf: float,
        fcf_growth_rates: List[float],
        shares_outstanding: float,
        current_price: float,
        net_debt: float = 0,
        margin_of_safety: float = 0.25
    ) -> Dict:
        """
        Calculate intrinsic value and compare with market price
        
        Args:
            current_fcf: Current year Free Cash Flow
            fcf_growth_rates: List of growth rates for projection years
            shares_outstanding: Total shares outstanding
            current_price: Current market price per share
            net_debt: Total debt - cash
            margin_of_safety: Safety margin percentage (default 25%)
            
        Returns:
            Dictionary with valuation and investment decision
        """
        dcf_result = self.calculate_intrinsic_value(
            current_fcf=current_fcf,
            fcf_growth_rates=fcf_growth_rates,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt
        )
        
        if not dcf_result:
            return None
        
        intrinsic_value = dcf_result['intrinsic_value_per_share']
        conservative_value = intrinsic_value * (1 - margin_of_safety)
        
        # Calculate upside/downside
        upside_pct = ((intrinsic_value - current_price) / current_price) * 100
        margin_pct = ((intrinsic_value - current_price) / intrinsic_value) * 100
        
        # Investment decision logic
        is_undervalued = current_price < conservative_value
        
        return {
            **dcf_result,
            'current_price': current_price,
            'conservative_value': conservative_value,
            'upside_potential_pct': upside_pct,
            'margin_to_intrinsic_pct': margin_pct,
            'margin_of_safety': margin_of_safety,
            'is_undervalued': is_undervalued,
            'price_to_value_ratio': current_price / intrinsic_value,
            'recommendation': self._get_recommendation(upside_pct, is_undervalued),
            'confidence_level': self._assess_confidence(fcf_growth_rates, current_fcf)
        }
    
    def sensitivity_analysis(
        self,
        current_fcf: float,
        fcf_growth_rates: List[float],
        shares_outstanding: float,
        net_debt: float = 0,
        discount_rate_range: tuple = (0.08, 0.12),
        terminal_growth_range: tuple = (0.02, 0.03)
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on discount rate and terminal growth
        
        Args:
            current_fcf: Current year FCF
            fcf_growth_rates: Growth rates for projection
            shares_outstanding: Shares outstanding
            net_debt: Net debt
            discount_rate_range: (min, max) for discount rate
            terminal_growth_range: (min, max) for terminal growth
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        # Test different discount rates
        discount_rates = np.linspace(discount_rate_range[0], discount_rate_range[1], 5)
        terminal_rates = np.linspace(terminal_growth_range[0], terminal_growth_range[1], 5)
        
        for dr in discount_rates:
            for tgr in terminal_rates:
                # Temporarily change rates
                original_dr = self.discount_rate
                original_tgr = self.terminal_growth_rate
                
                self.discount_rate = dr
                self.terminal_growth_rate = tgr
                
                result = self.calculate_intrinsic_value(
                    current_fcf=current_fcf,
                    fcf_growth_rates=fcf_growth_rates,
                    shares_outstanding=shares_outstanding,
                    net_debt=net_debt
                )
                
                if result:
                    results.append({
                        'discount_rate': dr,
                        'terminal_growth_rate': tgr,
                        'intrinsic_value': result['intrinsic_value_per_share']
                    })
                
                # Restore original rates
                self.discount_rate = original_dr
                self.terminal_growth_rate = original_tgr
        
        df = pd.DataFrame(results)
        return df.pivot(
            index='discount_rate',
            columns='terminal_growth_rate',
            values='intrinsic_value'
        )
    
    def _get_recommendation(self, upside_pct: float, is_undervalued: bool) -> str:
        """Generate investment recommendation"""
        if is_undervalued and upside_pct > 50:
            return "STRONG BUY"
        elif is_undervalued and upside_pct > 25:
            return "BUY"
        elif upside_pct > 10:
            return "HOLD"
        elif upside_pct > -10:
            return "NEUTRAL"
        elif upside_pct > -25:
            return "REDUCE"
        else:
            return "SELL"
    
    def _assess_confidence(self, growth_rates: List[float], current_fcf: float) -> str:
        """Assess confidence in valuation"""
        # Check consistency of growth rates
        if len(growth_rates) < 3:
            return "LOW"
        
        growth_volatility = np.std(growth_rates)
        
        # Check FCF quality
        fcf_quality = "HIGH" if current_fcf > 0 else "LOW"
        
        if growth_volatility < 0.05 and fcf_quality == "HIGH":
            return "HIGH"
        elif growth_volatility < 0.15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def estimate_growth_rates(
        self,
        historical_fcf: List[float],
        industry_growth: float = 0.05
    ) -> List[float]:
        """
        Estimate future growth rates based on historical FCF
        
        Args:
            historical_fcf: List of historical FCF values (oldest to newest)
            industry_growth: Industry average growth rate
            
        Returns:
            List of estimated growth rates for projection years
        """
        try:
            if len(historical_fcf) < 2:
                return [industry_growth] * self.projection_years
            
            # Calculate historical growth rates
            historical_growth = []
            for i in range(1, len(historical_fcf)):
                if historical_fcf[i-1] != 0:
                    growth = (historical_fcf[i] - historical_fcf[i-1]) / abs(historical_fcf[i-1])
                    historical_growth.append(growth)
            
            if not historical_growth:
                return [industry_growth] * self.projection_years
            
            # Average historical growth
            avg_growth = np.mean(historical_growth)
            
            # Create declining growth projection (conservative)
            estimated_rates = []
            for year in range(self.projection_years):
                # Gradually decline to industry average
                weight = (self.projection_years - year) / self.projection_years
                rate = avg_growth * weight + industry_growth * (1 - weight)
                
                # Cap at reasonable limits
                rate = max(min(rate, 0.25), -0.10)  # Between -10% and 25%
                estimated_rates.append(rate)
            
            return estimated_rates
            
        except Exception as e:
            self.logger.error(f"Error estimating growth rates: {str(e)}")
            return [industry_growth] * self.projection_years


# Example usage
if __name__ == "__main__":
    print("=== FCF-Based DCF Valuation Example ===\n")
    
    # Initialize model
    fcf_model = FCFDCFModel(
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        projection_years=5
    )
    
    # Example: Apple-like company
    result = fcf_model.calculate_with_market_price(
        current_fcf=100_000_000_000,  # $100B current FCF
        fcf_growth_rates=[0.12, 0.10, 0.08, 0.06, 0.05],  # Declining growth
        shares_outstanding=16_000_000_000,  # 16B shares
        current_price=175.00,  # Current market price
        net_debt=50_000_000_000,  # $50B net debt
        margin_of_safety=0.25
    )
    
    if result:
        print(f"Intrinsic Value per Share: ${result['intrinsic_value_per_share']:.2f}")
        print(f"Conservative Value (25% MoS): ${result['conservative_value']:.2f}")
        print(f"Current Market Price: ${result['current_price']:.2f}")
        print(f"Upside Potential: {result['upside_potential_pct']:.2f}%")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence Level: {result['confidence_level']}")
        print(f"\nProjected FCF (5 years):")
        for year, fcf in enumerate(result['projected_fcf'], start=1):
            print(f"  Year {year}: ${fcf/1e9:.2f}B")
        print(f"\nTerminal Value: ${result['terminal_value']/1e9:.2f}B")
        print(f"PV of Terminal Value: ${result['pv_terminal_value']/1e9:.2f}B")
        
        # Sensitivity analysis
        print("\n=== Sensitivity Analysis ===")
        sensitivity = fcf_model.sensitivity_analysis(
            current_fcf=100_000_000_000,
            fcf_growth_rates=[0.12, 0.10, 0.08, 0.06, 0.05],
            shares_outstanding=16_000_000_000,
            net_debt=50_000_000_000
        )
        print(sensitivity.round(2))
