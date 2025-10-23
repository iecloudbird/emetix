"""
LSTM Growth Rate Forecaster for DCF Components
Based on: https://www.revocm.com/articles/lstm-networks-estimating-growth-rates-dcf-models

This model forecasts growth rates for:
- Revenue
- Capital Expenditures (CapEx)
- Depreciation & Amortization (D&A)
- Net Operating Profit After Tax (NOPAT)

These forecasts are used to calculate Free Cash Flow to Firm (FCFF):
FCFF = NOPAT + D&A - CapEx - Change in NWC
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config.logging_config import get_logger

logger = get_logger(__name__)


class LSTMGrowthForecaster(nn.Module):
    """
    LSTM model for forecasting growth rates of DCF components.
    
    Unlike the original LSTM-DCF that predicts stock prices, this model:
    1. Takes normalized time-series of Revenue, CapEx, D&A, NOPAT
    2. Forecasts growth rates for each component
    3. Used in DCF valuation to project future cash flows
    
    Architecture:
    - Input: Normalized financial metrics over sequence length
    - LSTM layers: Capture temporal dependencies
    - Output: Growth rate predictions for next period
    """
    
    def __init__(
        self,
        input_size: int = 4,  # Revenue, CapEx, D&A, NOPAT
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 4  # Growth rates for each component
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        logger.info(f"Initialized LSTM Growth Forecaster: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Contains normalized [Revenue, CapEx, D&A, NOPAT]
        
        Returns:
            Growth rate predictions of shape (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, output_size) - growth rates
        
        return out
    
    def forecast_growth_rates(
        self,
        historical_data: np.ndarray,
        normalization_params: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Forecast growth rates for DCF components
        
        Args:
            historical_data: Array of shape (sequence_length, 4) with normalized values
            normalization_params: Dict with 'mean' and 'std' for denormalization
        
        Returns:
            Dictionary with forecasted growth rates:
            {
                'revenue_growth': float,
                'capex_growth': float,
                'da_growth': float,
                'nopat_growth': float
            }
        """
        self.eval()
        
        with torch.no_grad():
            # Prepare input
            x = torch.FloatTensor(historical_data).unsqueeze(0)  # (1, seq_len, 4)
            
            # Get predictions
            growth_rates = self.forward(x).squeeze(0).numpy()  # (4,)
            
            # Map to component names
            components = ['revenue', 'capex', 'da', 'nopat']
            forecasts = {
                f'{comp}_growth': float(growth_rates[i])
                for i, comp in enumerate(components)
            }
            
            logger.info(f"Forecasted growth rates: {forecasts}")
            return forecasts
    
    def multi_period_forecast(
        self,
        historical_data: np.ndarray,
        periods: int = 5,
        industry_growth: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[float]]:
        """
        Forecast growth rates for multiple periods with industry convergence
        
        Following the article's approach:
        - Year 1: LSTM forecast
        - Years 2-5: Linear interpolation to industry average
        
        Args:
            historical_data: Array of shape (sequence_length, 4)
            periods: Number of years to forecast
            industry_growth: Optional industry average growth rates
        
        Returns:
            Dictionary with growth rate trajectories for each component
        """
        # Get Year 1 forecast
        year1_growth = self.forecast_growth_rates(
            historical_data,
            normalization_params={'mean': 0, 'std': 1}  # Placeholder
        )
        
        # Default industry growth rates (can be customized)
        if industry_growth is None:
            industry_growth = {
                'revenue_growth': 0.05,  # 5% industry average
                'capex_growth': 0.04,
                'da_growth': 0.03,
                'nopat_growth': 0.05
            }
        
        # Initialize trajectories
        trajectories = {key: [] for key in year1_growth.keys()}
        
        # Linear interpolation to industry average
        for key in year1_growth.keys():
            lstm_rate = year1_growth[key]
            industry_rate = industry_growth[key]
            
            for t in range(periods):
                # Linear interpolation: LSTM → industry over 5 years
                weight_lstm = max(0, 1 - t / (periods - 1))
                weight_industry = min(1, t / (periods - 1))
                
                interpolated_rate = weight_lstm * lstm_rate + weight_industry * industry_rate
                trajectories[key].append(interpolated_rate)
        
        logger.info(f"Multi-period forecast complete: {periods} years")
        return trajectories
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        return self


class DCFValuationWithLSTM:
    """
    DCF Valuation using LSTM-forecasted growth rates
    
    Implements the FCFF method:
    FCFF = NOPAT + D&A - CapEx - Change in NWC
    
    Where:
    - NOPAT = EBIT * (1 - Tax Rate)
    - Growth rates forecasted by LSTM
    """
    
    def __init__(self, lstm_model: LSTMGrowthForecaster):
        self.lstm_model = lstm_model
        self.logger = logger
    
    def calculate_fcff(
        self,
        nopat: float,
        da: float,
        capex: float,
        nwc_change: float = 0
    ) -> float:
        """
        Calculate Free Cash Flow to Firm
        
        FCFF = NOPAT + D&A - CapEx - ΔNW C
        """
        fcff = nopat + da - capex - nwc_change
        return fcff
    
    def project_fcff(
        self,
        current_metrics: Dict[str, float],
        growth_trajectories: Dict[str, List[float]],
        periods: int = 5
    ) -> List[float]:
        """
        Project FCFF for multiple periods using LSTM growth rates
        
        Args:
            current_metrics: Current year values {nopat, da, capex, revenue}
            growth_trajectories: Growth rates from LSTM multi-period forecast
            periods: Number of years to project
        
        Returns:
            List of projected FCFF values
        """
        fcff_projections = []
        
        # Initialize with current values
        nopat = current_metrics.get('nopat', 0)
        da = current_metrics.get('da', 0)
        capex = current_metrics.get('capex', 0)
        
        for t in range(periods):
            # Apply growth rates
            nopat *= (1 + growth_trajectories['nopat_growth'][t])
            da *= (1 + growth_trajectories['da_growth'][t])
            capex *= (1 + growth_trajectories['capex_growth'][t])
            
            # Calculate FCFF
            fcff = self.calculate_fcff(nopat, da, capex, nwc_change=0)
            fcff_projections.append(fcff)
        
        return fcff_projections
    
    def dcf_valuation(
        self,
        fcff_projections: List[float],
        wacc: float = 0.08,
        terminal_growth: float = 0.03,
        shares_outstanding: float = 1e9
    ) -> Dict[str, float]:
        """
        Calculate DCF valuation from FCFF projections
        
        Args:
            fcff_projections: List of projected FCFF values
            wacc: Weighted Average Cost of Capital
            terminal_growth: Terminal growth rate
            shares_outstanding: Number of shares
        
        Returns:
            Dictionary with valuation results
        """
        # Present value of projected FCFF
        pv_fcff = sum(
            fcff / ((1 + wacc) ** (t + 1))
            for t, fcff in enumerate(fcff_projections)
        )
        
        # Terminal value
        last_fcff = fcff_projections[-1]
        terminal_value = (last_fcff * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** len(fcff_projections))
        
        # Enterprise value
        enterprise_value = pv_fcff + pv_terminal
        
        # Fair value per share (simplified - not accounting for debt/cash)
        fair_value_per_share = enterprise_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'pv_fcff': pv_fcff,
            'terminal_value': terminal_value,
            'pv_terminal': pv_terminal,
            'fair_value_per_share': fair_value_per_share,
            'fcff_projections': fcff_projections
        }
    
    def full_valuation(
        self,
        historical_data: np.ndarray,
        current_metrics: Dict[str, float],
        wacc: float = 0.08,
        terminal_growth: float = 0.03,
        shares_outstanding: float = 1e9,
        projection_years: int = 5,
        industry_growth: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Complete LSTM-DCF valuation pipeline
        
        1. Forecast growth rates using LSTM
        2. Project FCFF for multiple periods
        3. Calculate DCF valuation
        
        Returns:
            Complete valuation results
        """
        # Step 1: Multi-period growth forecast
        growth_trajectories = self.lstm_model.multi_period_forecast(
            historical_data,
            periods=projection_years,
            industry_growth=industry_growth
        )
        
        # Step 2: Project FCFF
        fcff_projections = self.project_fcff(
            current_metrics,
            growth_trajectories,
            periods=projection_years
        )
        
        # Step 3: DCF valuation
        valuation = self.dcf_valuation(
            fcff_projections,
            wacc=wacc,
            terminal_growth=terminal_growth,
            shares_outstanding=shares_outstanding
        )
        
        # Add growth trajectories to results
        valuation['growth_trajectories'] = growth_trajectories
        
        self.logger.info(f"LSTM-DCF Valuation complete: Fair Value = ${valuation['fair_value_per_share']:.2f}")
        
        return valuation


if __name__ == "__main__":
    # Example usage
    print("LSTM Growth Forecaster for DCF")
    print("=" * 60)
    
    # Initialize model
    model = LSTMGrowthForecaster(
        input_size=4,  # Revenue, CapEx, D&A, NOPAT
        hidden_size=64,
        num_layers=2
    )
    
    # Example historical data (normalized)
    # Shape: (sequence_length, 4)
    historical_data = np.random.randn(20, 4)  # 20 quarters of data
    
    # Test forecast
    growth_rates = model.forecast_growth_rates(
        historical_data,
        normalization_params={'mean': 0, 'std': 1}
    )
    
    print("\nForecasted Growth Rates:")
    for key, value in growth_rates.items():
        print(f"  {key}: {value*100:.2f}%")
    
    # Test multi-period forecast
    trajectories = model.multi_period_forecast(historical_data, periods=5)
    
    print("\n5-Year Growth Trajectory (Revenue):")
    for year, growth in enumerate(trajectories['revenue_growth'], 1):
        print(f"  Year {year}: {growth*100:.2f}%")
    
    # Test DCF valuation
    dcf = DCFValuationWithLSTM(model)
    current_metrics = {
        'nopat': 50e9,  # $50B
        'da': 10e9,     # $10B
        'capex': 15e9,  # $15B
        'revenue': 200e9  # $200B
    }
    
    valuation = dcf.full_valuation(
        historical_data,
        current_metrics,
        shares_outstanding=15e9,  # 15B shares
        projection_years=5
    )
    
    print(f"\nDCF Valuation Results:")
    print(f"  Enterprise Value: ${valuation['enterprise_value']/1e9:.2f}B")
    print(f"  Fair Value/Share: ${valuation['fair_value_per_share']:.2f}")
    print(f"  Terminal Value (PV): ${valuation['pv_terminal']/1e9:.2f}B")
