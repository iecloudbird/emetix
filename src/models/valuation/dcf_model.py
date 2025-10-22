"""
Discounted Cash Flow (DCF) model for intrinsic valuation
Based on Phase 2 draft code
"""
import pandas as pd
from config.logging_config import get_logger

logger = get_logger(__name__)


class DCFModel:
    """
    Simple DCF calculator for fair value estimation
    Formula: fair_value = (EPS * (1 + growth)) / (discount - growth)
    """
    
    def __init__(self, discount_rate: float = 0.10):
        """
        Args:
            discount_rate: Required rate of return (default 10%)
        """
        self.discount_rate = discount_rate
        self.logger = logger
    
    def calculate_fair_value(self, eps: float, growth_rate: float) -> dict:
        """
        Calculate fair value using simplified DCF
        
        Args:
            eps: Earnings per share
            growth_rate: Expected growth rate (e.g., 0.15 for 15%)
        
        Returns:
            Dictionary with fair value and metrics
        """
        try:
            if growth_rate >= self.discount_rate:
                self.logger.warning("Growth rate >= discount rate. Using conservative estimate.")
                growth_rate = self.discount_rate * 0.8
            
            # Simplified DCF formula
            fair_value = (eps * (1 + growth_rate)) / (self.discount_rate - growth_rate)
            
            return {
                'fair_value': fair_value,
                'eps': eps,
                'growth_rate': growth_rate,
                'discount_rate': self.discount_rate,
                'method': 'DCF'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating DCF: {str(e)}")
            return None
    
    def calculate_intrinsic_value_with_margin(
        self, 
        eps: float, 
        growth_rate: float, 
        current_price: float,
        margin_of_safety: float = 0.25
    ) -> dict:
        """
        Calculate intrinsic value with margin of safety
        
        Args:
            eps: Earnings per share
            growth_rate: Expected growth rate
            current_price: Current market price
            margin_of_safety: Safety margin (default 25%)
        
        Returns:
            Dictionary with valuation details
        """
        dcf_result = self.calculate_fair_value(eps, growth_rate)
        
        if not dcf_result:
            return None
        
        fair_value = dcf_result['fair_value']
        conservative_value = fair_value * (1 - margin_of_safety)
        
        upside_pct = ((fair_value - current_price) / current_price) * 100
        
        return {
            **dcf_result,
            'conservative_value': conservative_value,
            'current_price': current_price,
            'upside_potential_pct': upside_pct,
            'margin_of_safety': margin_of_safety,
            'is_undervalued': current_price < conservative_value,
            'recommendation': self._get_recommendation(upside_pct)
        }
    
    def _get_recommendation(self, upside_pct: float) -> str:
        """Get investment recommendation"""
        if upside_pct > 40:
            return "STRONG BUY"
        elif upside_pct > 20:
            return "BUY"
        elif upside_pct > 0:
            return "HOLD"
        elif upside_pct > -20:
            return "UNDERPERFORM"
        else:
            return "SELL"
    
    def batch_calculate(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DCF for multiple stocks
        
        Args:
            stocks_df: DataFrame with 'eps', 'growth_rate', 'current_price' columns
        
        Returns:
            DataFrame with valuation results
        """
        results = []
        
        for _, row in stocks_df.iterrows():
            result = self.calculate_intrinsic_value_with_margin(
                eps=row['eps'],
                growth_rate=row.get('growth_rate', 0.05),
                current_price=row['current_price']
            )
            
            if result:
                results.append(result)
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    dcf = DCFModel(discount_rate=0.10)
    
    # Single stock
    result = dcf.calculate_intrinsic_value_with_margin(
        eps=5.50,
        growth_rate=0.15,
        current_price=75.00
    )
    
    print("DCF Valuation:")
    print(f"Fair Value: ${result['fair_value']:.2f}")
    print(f"Conservative Value: ${result['conservative_value']:.2f}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Upside Potential: {result['upside_potential_pct']:.2f}%")
    print(f"Recommendation: {result['recommendation']}")
