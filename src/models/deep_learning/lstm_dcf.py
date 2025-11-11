"""
LSTM-DCF Hybrid Model for Stock Valuation
Combines LSTM forecasting with DCF valuation framework
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)


class LSTMDCFModel(pl.LightningModule):
    """
    Hybrid LSTM-DCF Model
    
    Architecture:
    - LSTM layers for time-series forecasting
    - Dropout for regularization
    - Linear layer for FCFF/growth prediction
    - DCF valuation logic for fair value calculation
    
    Pattern: Follows project's model persistence conventions
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        output_size: int = 2,
        wacc: float = 0.08,
        terminal_growth: float = 0.03
    ):
        """
        Initialize LSTM-DCF model for growth rate prediction
        
        Args:
            input_size: Number of input features (16 recommended: core + margins + normalized)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            output_size: Number of outputs (2 for revenue_growth + fcf_growth, 3 to add ebitda_growth)
            wacc: Weighted Average Cost of Capital
            terminal_growth: Terminal growth rate for DCF
        """
        super().__init__()
        self.save_hyperparameters()
        
        # LSTM architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer - now predicts multiple growth rates
        self.fc = nn.Linear(hidden_size, output_size)  # Output: [revenue_growth, fcf_growth, (optional ebitda_growth)]
        
        # DCF parameters
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, sequence_length, features)
            
        Returns:
            Predicted growth rates (batch, output_size)
            - output_size=2: [revenue_growth, fcf_growth]
            - output_size=3: [revenue_growth, fcf_growth, ebitda_growth]
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction
    
    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step for growth rate prediction"""
        sequences, targets = batch
        predictions = self.forward(sequences)
        
        # Handle both single and multi-output cases
        if self.hparams.output_size == 1:
            predictions = predictions.squeeze()
            targets = targets.squeeze()
        
        loss = self.criterion(predictions, targets)
        
        # Log individual losses for multi-output
        if self.hparams.output_size > 1:
            for i in range(self.hparams.output_size):
                self.log(f'train_loss_output_{i}', self.criterion(predictions[:, i], targets[:, i]))
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """PyTorch Lightning validation step for growth rate prediction"""
        sequences, targets = batch
        predictions = self.forward(sequences)
        
        # Handle both single and multi-output cases
        if self.hparams.output_size == 1:
            predictions = predictions.squeeze()
            targets = targets.squeeze()
        
        loss = self.criterion(predictions, targets)
        
        # Log individual losses for multi-output
        if self.hparams.output_size > 1:
            for i in range(self.hparams.output_size):
                self.log(f'val_loss_output_{i}', self.criterion(predictions[:, i], targets[:, i]))
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Optimizer configuration"""
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
    
    def forecast_fcff(
        self, 
        sequence: torch.Tensor, 
        periods: int = 10,
        scaler = None,
        fcff_feature_idx: int = -1,
        shares_outstanding: float = 1.0,
        use_per_share: bool = True
    ) -> List[float]:
        """
        Forecast FCFF for multiple periods
        
        Args:
            sequence: Input sequence (1, seq_len, features)
            periods: Number of forecast periods
            scaler: Optional scaler to inverse-transform predictions
            fcff_feature_idx: Index of FCFF feature in scaler (default: -1 for last feature)
            shares_outstanding: Shares outstanding (used if use_per_share=False)
            use_per_share: If True, returns per-share FCFF. If False, returns aggregate FCFF
            
        Returns:
            List of forecasted FCFF values (per-share or aggregate based on use_per_share)
        """
        self.eval()
        forecasts = []
        
        with torch.no_grad():
            current_seq = sequence.clone()
            for _ in range(periods):
                pred = self.forward(current_seq)
                pred_value = pred.item()
                
                # Inverse transform if scaler provided
                if scaler is not None:
                    try:
                        n_features = scaler.n_features_in_
                        dummy = np.zeros((1, n_features))
                        dummy[0, fcff_feature_idx] = pred_value
                        denormalized = scaler.inverse_transform(dummy)
                        pred_value = denormalized[0, fcff_feature_idx]
                        
                        # Scale to aggregate if requested
                        if not use_per_share:
                            pred_value *= shares_outstanding
                    except Exception as e:
                        logger.warning(f"Inverse transform failed, using normalized value: {e}")
                
                forecasts.append(pred_value)
                
                # Update sequence (simplified - append prediction)
                # In production, update with actual features
                new_step = torch.zeros(1, 1, sequence.shape[2])
                new_step[0, 0, 0] = pred  # Update first feature with prediction
                current_seq = torch.cat([current_seq[:, 1:, :], new_step], dim=1)
        
        return forecasts
    
    def dcf_valuation(
        self, 
        fcff_forecasts: List[float], 
        current_shares: float = 1.0,
        current_price: float = None
    ) -> Dict[str, float]:
        """
        Calculate DCF fair value from FCFF forecasts
        
        Args:
            fcff_forecasts: List of forecasted FCFF (per-share recommended)
            current_shares: Shares outstanding (use 1.0 for per-share FCFF)
            current_price: Current stock price for calibration (optional)
            
        Returns:
            Dict with fair_value, terminal_value, pv_fcff, enterprise_value
        """
        if not fcff_forecasts or current_shares <= 0:
            logger.warning("Invalid inputs for DCF valuation")
            return {
                'fair_value': 0,
                'pv_fcff': 0,
                'terminal_value': 0,
                'pv_terminal_value': 0,
                'enterprise_value': 0,
                'calibrated_fair_value': 0
            }
        
        # Present value of forecast period
        pv_fcff = sum(
            fcff / ((1 + self.wacc) ** t)
            for t, fcff in enumerate(fcff_forecasts, start=1)
        )
        
        # Terminal value
        terminal_fcff = fcff_forecasts[-1] * (1 + self.terminal_growth)
        terminal_value = terminal_fcff / (self.wacc - self.terminal_growth)
        pv_terminal = terminal_value / ((1 + self.wacc) ** len(fcff_forecasts))
        
        # Total enterprise value
        enterprise_value = pv_fcff + pv_terminal
        fair_value_per_share = enterprise_value / current_shares
        
        # Calibration: If model overestimates significantly, apply dampening
        calibrated_fair_value = fair_value_per_share
        if current_price and current_price > 0:
            valuation_ratio = fair_value_per_share / current_price
            if valuation_ratio > 5:  # Model predicts 5x+ current price
                # Apply logarithmic dampening to extreme valuations
                dampening_factor = 0.3 + 0.7 * (1 / np.log10(valuation_ratio + 1))
                calibrated_fair_value = current_price * (1 + (valuation_ratio - 1) * dampening_factor)
                logger.info(f"Applied calibration: ${fair_value_per_share:.2f} → ${calibrated_fair_value:.2f} (ratio: {valuation_ratio:.2f}x)")
        
        return {
            'fair_value': float(fair_value_per_share),
            'calibrated_fair_value': float(calibrated_fair_value),
            'pv_fcff': float(pv_fcff),
            'terminal_value': float(terminal_value),
            'pv_terminal_value': float(pv_terminal),
            'enterprise_value': float(enterprise_value)
        }
    
    def predict_stock_value(
        self,
        sequence: torch.Tensor,
        current_price: float,
        shares_outstanding: float,
        scaler = None,
        fcff_feature_idx: int = -1
    ) -> Dict[str, float]:
        """
        Complete valuation prediction
        
        Args:
            sequence: Input sequence tensor
            current_price: Current stock price
            shares_outstanding: Number of shares outstanding
            scaler: Optional scaler for inverse transformation
            fcff_feature_idx: Index of FCFF feature in scaler
            
        Returns:
            Complete valuation metrics
        """
        # Forecast FCFF
        fcff_forecasts = self.forecast_fcff(
            sequence, 
            periods=10, 
            scaler=scaler, 
            fcff_feature_idx=fcff_feature_idx,
            shares_outstanding=shares_outstanding
        )
        
        # DCF valuation
        dcf_result = self.dcf_valuation(fcff_forecasts, shares_outstanding)
        
        # Calculate gap
        fair_value = dcf_result['fair_value']
        gap_percent = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
        
        return {
            **dcf_result,
            'current_price': float(current_price),
            'fair_value_gap': float(gap_percent),
            'is_undervalued': gap_percent > 20,  # 20% threshold
            'fcff_forecast': [float(f) for f in fcff_forecasts]
        }
    
    def save_model(self, path: str):
        """
        Save model weights
        
        Pattern: Compatible with project's persistence convention
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        logger.info(f"LSTM-DCF model saved: {save_path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
        logger.info(f"LSTM-DCF model loaded: {path}")
