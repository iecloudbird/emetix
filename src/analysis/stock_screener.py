"""
Enhanced US Stock Market Screener
Advanced screening with LSTM-DCF fair values, sector comparisons, and AI-driven insights

Features:
1. LSTM-DCF fair value estimation for each stock
2. Sector average P/E for baseline comparison
3. Justification for each ranking
4. Multi-agent LangChain integration for AI insights
5. Comprehensive output for investment decisions
6. Chart data endpoints for frontend visualization
"""
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from config.logging_config import get_logger
from config.settings import CACHE_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.data.fetchers import YFinanceFetcher
from src.data.fetchers.ticker_universe import TickerUniverseFetcher
from src.analysis.personal_risk_capacity import PersonalRiskCapacityService

logger = get_logger(__name__)


# Default Sector Benchmarks (fallback when dynamic calculation unavailable)
# Includes average growth rates for ensemble averaging
DEFAULT_SECTOR_BENCHMARKS = {
    'Technology': {'avg_pe': 28.0, 'avg_pb': 6.0, 'avg_roe': 18.0, 'avg_margin': 20.0, 'avg_growth': 15.0},
    'Healthcare': {'avg_pe': 22.0, 'avg_pb': 4.0, 'avg_roe': 15.0, 'avg_margin': 15.0, 'avg_growth': 10.0},
    'Financial Services': {'avg_pe': 14.0, 'avg_pb': 1.5, 'avg_roe': 12.0, 'avg_margin': 25.0, 'avg_growth': 8.0},
    'Consumer Cyclical': {'avg_pe': 20.0, 'avg_pb': 4.0, 'avg_roe': 15.0, 'avg_margin': 10.0, 'avg_growth': 10.0},
    'Consumer Defensive': {'avg_pe': 22.0, 'avg_pb': 5.0, 'avg_roe': 20.0, 'avg_margin': 12.0, 'avg_growth': 6.0},
    'Industrials': {'avg_pe': 20.0, 'avg_pb': 4.0, 'avg_roe': 14.0, 'avg_margin': 10.0, 'avg_growth': 8.0},
    'Energy': {'avg_pe': 12.0, 'avg_pb': 1.8, 'avg_roe': 15.0, 'avg_margin': 10.0, 'avg_growth': 5.0},
    'Utilities': {'avg_pe': 18.0, 'avg_pb': 1.8, 'avg_roe': 10.0, 'avg_margin': 15.0, 'avg_growth': 4.0},
    'Basic Materials': {'avg_pe': 15.0, 'avg_pb': 2.5, 'avg_roe': 12.0, 'avg_margin': 12.0, 'avg_growth': 6.0},
    'Communication Services': {'avg_pe': 18.0, 'avg_pb': 3.5, 'avg_roe': 14.0, 'avg_margin': 18.0, 'avg_growth': 8.0},
    'Real Estate': {'avg_pe': 35.0, 'avg_pb': 2.0, 'avg_roe': 8.0, 'avg_margin': 30.0, 'avg_growth': 5.0},
    'Unknown': {'avg_pe': 20.0, 'avg_pb': 3.0, 'avg_roe': 12.0, 'avg_margin': 12.0, 'avg_growth': 7.0},
}

# Global reference (updated dynamically during scans)
SECTOR_BENCHMARKS = DEFAULT_SECTOR_BENCHMARKS.copy()


class StockScreener:
    """
    Stock screener with LSTM-DCF fair values, sector comparisons,
    and AI-driven insights from LangChain agents
    """
    
    # S&P 500 + Extended Universe tickers
    SP500_TICKERS = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 
        'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'TXN', 'QCOM', 'IBM',
        'NOW', 'INTU', 'AMAT', 'ADI', 'LRCX', 'MU', 'SNPS', 'CDNS',
        'MRVL', 'KLAC', 'NXPI', 'MCHP', 'ON', 'FTNT', 'PANW', 'CRWD',
        
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
        'BMY', 'AMGN', 'MDT', 'GILD', 'CVS', 'ISRG', 'VRTX', 'REGN',
        'SYK', 'BSX', 'ZTS', 'BDX', 'CI', 'HUM', 'ELV', 'MCK',
        
        # Consumer
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT',
        'COST', 'WMT', 'PG', 'KO', 'PEP', 'MDLZ', 'MO', 'PM',
        'CL', 'EL', 'GIS', 'KHC', 'SYY', 'STZ', 'HSY', 'K',
        
        # Financial
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
        'AXP', 'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC', 'TFC',
        
        # Industrial
        'CAT', 'DE', 'UNP', 'HON', 'UPS', 'RTX', 'BA', 'LMT',
        'GE', 'MMM', 'EMR', 'ITW', 'ETN', 'PH', 'CMI', 'PCAR',
        
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
        'OXY', 'PXD', 'DVN', 'HAL', 'KMI', 'WMB', 'OKE', 'FANG',
        
        # Communications
        'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
        
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE',
        
        # Real Estate
        'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'SPG', 'O', 'WELL',
        
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
    ]
    
    EXTENDED_TICKERS = [
        'PLTR', 'NET', 'DDOG', 'ZS', 'TEAM', 'SNOW', 'MDB', 'OKTA',
        'TWLO', 'SQ', 'SHOP', 'WDAY', 'VEEV', 'HUBS', 'TTD', 'ZM',
        'LULU', 'DECK', 'ULTA', 'POOL', 'BBY', 'DG', 'DLTR', 'FIVE',
        'BURL', 'RH', 'WSM', 'ROST', 'TJX', 'ORLY', 'AZO', 'AAP',
        'DXCM', 'IDXX', 'ALGN', 'PODD', 'HOLX', 'IQV', 'A', 'WAT',
        'GWW', 'FAST', 'ODFL', 'J', 'LII', 'GNRC', 'TT', 'DOV',
        'ASML', 'TSM', 'ARM', 'SMCI', 'CRUS', 'MPWR',
    ]
    
    DEFAULT_CRITERIA = {
        'min_market_cap': 1_000_000_000,
        'max_pe_ratio': 50,
        'min_pe_ratio': 0,
        'max_debt_equity': 3.0,
        'min_volume': 100_000,
        'min_roe': 0,
    }
    
    def __init__(
        self,
        use_extended_universe: bool = True,
        use_full_universe: bool = False,  # NEW: Scan all US stocks from NASDAQ
        max_universe_tickers: Optional[int] = None,  # Limit for testing/performance
        criteria: Optional[Dict] = None,
        cache_enabled: bool = True,
        cache_expiry_hours: int = 6,
        max_workers: int = 10,
        enable_lstm: bool = True,
        enable_consensus: bool = True,  # Multi-model consensus scoring (LSTM+GARP+Risk)
        enable_ai_insights: bool = False,  # Optional AI insights
        enable_education_mode: bool = False,  # Educational explanations for social impact
        use_dynamic_benchmarks: bool = True  # Calculate sector benchmarks from scanned data
    ):
        self.logger = logger
        self.fetcher = YFinanceFetcher()
        
        # Build ticker universe
        if use_full_universe:
            # Dynamic universe from NASDAQ (5000+ tickers)
            universe_fetcher = TickerUniverseFetcher()
            self.tickers = universe_fetcher.get_all_us_tickers(max_tickers=max_universe_tickers)
            self.logger.info(f"📊 Using FULL US market universe: {len(self.tickers)} tickers")
        else:
            # Legacy hardcoded list
            self.tickers = self.SP500_TICKERS.copy()
            if use_extended_universe:
                self.tickers.extend(self.EXTENDED_TICKERS)
            self.tickers = list(set(self.tickers))
        
        # Merge criteria
        self.criteria = self.DEFAULT_CRITERIA.copy()
        if criteria:
            self.criteria.update(criteria)
        
        # Cache settings
        self.cache_enabled = cache_enabled
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_dir = CACHE_DIR / "stock_screener"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.enable_lstm = enable_lstm
        self.enable_consensus = enable_consensus
        self.enable_ai_insights = enable_ai_insights
        self.enable_education_mode = enable_education_mode
        self.use_dynamic_benchmarks = use_dynamic_benchmarks
        
        # Dynamic sector benchmarks (calculated from scanned universe)
        self._dynamic_benchmarks = None
        
        # LSTM model components
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_feature_cols = None
        self.lstm_target_cols = None
        self.lstm_sequence_length = 60  # Default, will be overwritten by checkpoint
        
        # Consensus scorer (LSTM 50% + GARP 25% + Risk 25%)
        self.consensus_scorer = None
        
        # Personal Risk Capacity Service (for position sizing)
        self.risk_capacity_service = PersonalRiskCapacityService()
        
        # Load ML models
        if enable_lstm:
            self._load_lstm_model()
        if enable_consensus:
            self._load_consensus_scorer()
        
        # AI Agents (lazy loading)
        self._valuation_agent = None
        self._risk_agent = None
        
        # Log initialization status
        models_status = []
        if self.lstm_model: models_status.append("LSTM-DCF")
        if self.consensus_scorer: models_status.append("Consensus(LSTM+GARP+Risk)")
        self.logger.info(f"StockScreener initialized: {len(self.tickers)} tickers, Models: {', '.join(models_status) or 'None'}, Education: {'ON' if enable_education_mode else 'OFF'}")
    
    def _load_lstm_model(self):
        """Load LSTM-DCF Enhanced model for fair value estimation with proper scaler"""
        try:
            from src.models.deep_learning.lstm_dcf import LSTMDCFModel
            
            # Load Enhanced model (16 features, trained on quarterly financials)
            model_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
            if not model_path.exists():
                self.logger.warning(f"⚠️ LSTM model not found: {model_path}")
                return
            
            # Use new from_checkpoint() method for auto-loading with correct dimensions
            self.lstm_model, metadata = LSTMDCFModel.from_checkpoint(str(model_path))
            
            # Extract metadata
            self.lstm_scaler = metadata.get('scaler', None)
            self.lstm_feature_cols = metadata.get('feature_cols', None)
            self.lstm_target_cols = metadata.get('target_cols', None)
            self.lstm_sequence_length = metadata.get('sequence_length', 60)
            
            hp = metadata.get('hyperparameters', {})
            self.logger.info(f"✅ LSTM-DCF Enhanced loaded: {hp.get('input_size', '?')} features, {hp.get('output_size', '?')} outputs, seq_len={self.lstm_sequence_length}")
            
        except Exception as e:
            self.logger.warning(f"LSTM load error: {e}")
            self.lstm_model = None
            self.lstm_scaler = None

    def _load_consensus_scorer(self):
        """Load ConsensusScorer for multi-model weighted voting
        
        ARCHITECTURE SHIFT (Jan 2025):
        - RF Ensemble DEPRECATED (99.93% P/E importance = just a P/E filter)
        - Replaced with transparent GARP scoring (Forward P/E + PEG)
        - Weights now: LSTM 50%, GARP 25%, Risk 25%
        """
        try:
            from src.models.ensemble.consensus_scorer import ConsensusScorer
            
            # REVISED WEIGHTS (Jan 2025) - No RF, use transparent GARP
            # RF was 99.93% P/E, so we replace it with explicit P/E + PEG scoring
            self.consensus_scorer = ConsensusScorer(weights={
                'lstm_dcf': 0.50,        # Primary: Fair value from growth forecast
                'garp_score': 0.25,      # Forward P/E + PEG (replaces RF)
                'risk_score': 0.25       # Beta + volatility filter
            })
            self.logger.info("✅ ConsensusScorer initialized (LSTM 50%, GARP 25%, Risk 25%)")
            
        except Exception as e:
            self.logger.warning(f"ConsensusScorer load error: {e}")
            self.consensus_scorer = None

    def _get_valuation_agent(self):
        """Lazy load valuation agent"""
        if self._valuation_agent is None and self.enable_ai_insights:
            try:
                from src.agents.valuation_agent import ValuationAgent
                self._valuation_agent = ValuationAgent()
            except Exception as e:
                self.logger.warning(f"Valuation agent unavailable: {e}")
        return self._valuation_agent
    
    def _get_risk_agent(self):
        """Lazy load risk agent"""
        if self._risk_agent is None and self.enable_ai_insights:
            try:
                from src.agents.risk_agent import RiskAgent
                self._risk_agent = RiskAgent()
            except Exception as e:
                self.logger.warning(f"Risk agent unavailable: {e}")
        return self._risk_agent
    
    def get_ticker_universe(self) -> List[str]:
        return self.tickers
    
    def set_ticker_universe(self, tickers: List[str]):
        self.tickers = list(set(tickers))
    
    def calculate_dynamic_sector_benchmarks(self, stock_data: List[Dict]) -> Dict:
        """
        Calculate sector benchmarks dynamically from scanned stock universe.
        This provides more accurate, real-time sector comparisons.
        
        Args:
            stock_data: List of stock data dictionaries from scan
            
        Returns:
            Dictionary of sector benchmarks with avg_pe, avg_pb, avg_roe, avg_margin
        """
        if not stock_data:
            return DEFAULT_SECTOR_BENCHMARKS.copy()
        
        df = pd.DataFrame(stock_data)
        benchmarks = {}
        
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector]
            
            # Filter valid P/E ratios (positive, reasonable range)
            valid_pe = sector_data[(sector_data['pe_ratio'] > 0) & (sector_data['pe_ratio'] < 100)]['pe_ratio']
            valid_pb = sector_data[(sector_data['pb_ratio'] > 0) & (sector_data['pb_ratio'] < 50)]['pb_ratio']
            valid_roe = sector_data[(sector_data['roe'] > -50) & (sector_data['roe'] < 100)]['roe']
            valid_margin = sector_data[(sector_data['profit_margin'] > -50) & (sector_data['profit_margin'] < 100)]['profit_margin']
            
            benchmarks[sector] = {
                'avg_pe': round(valid_pe.median(), 1) if len(valid_pe) >= 3 else DEFAULT_SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_BENCHMARKS['Unknown'])['avg_pe'],
                'avg_pb': round(valid_pb.median(), 2) if len(valid_pb) >= 3 else DEFAULT_SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_BENCHMARKS['Unknown'])['avg_pb'],
                'avg_roe': round(valid_roe.median(), 1) if len(valid_roe) >= 3 else DEFAULT_SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_BENCHMARKS['Unknown'])['avg_roe'],
                'avg_margin': round(valid_margin.median(), 1) if len(valid_margin) >= 3 else DEFAULT_SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_BENCHMARKS['Unknown'])['avg_margin'],
                'sample_size': len(sector_data),
                'source': 'dynamic' if len(valid_pe) >= 3 else 'default'
            }
        
        # Add Unknown fallback
        if 'Unknown' not in benchmarks:
            benchmarks['Unknown'] = DEFAULT_SECTOR_BENCHMARKS['Unknown'].copy()
            benchmarks['Unknown']['sample_size'] = 0
            benchmarks['Unknown']['source'] = 'default'
        
        self.logger.info(f"📊 Dynamic sector benchmarks calculated from {len(stock_data)} stocks across {len(benchmarks)} sectors")
        return benchmarks
    
    def get_sector_benchmark(self, sector: str) -> Dict:
        """
        Get benchmark for a specific sector (dynamic if available, else default).
        
        Args:
            sector: Sector name
            
        Returns:
            Dictionary with avg_pe, avg_pb, avg_roe, avg_margin
        """
        if self._dynamic_benchmarks and sector in self._dynamic_benchmarks:
            return self._dynamic_benchmarks[sector]
        return SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])
    
    def update_global_benchmarks(self):
        """Update the global SECTOR_BENCHMARKS with dynamic values"""
        global SECTOR_BENCHMARKS
        if self._dynamic_benchmarks:
            SECTOR_BENCHMARKS.update(self._dynamic_benchmarks)
            self.logger.info("✅ Global sector benchmarks updated with dynamic values")

    def get_current_benchmarks(self) -> Dict:
        """
        Get current sector benchmarks with metadata.
        Useful for API endpoints and frontend display.
        
        Returns:
            Dictionary with benchmarks and metadata
        """
        dynamic_count = sum(1 for b in SECTOR_BENCHMARKS.values() if b.get('source') == 'dynamic')
        
        return {
            'benchmarks': SECTOR_BENCHMARKS.copy(),
            'metadata': {
                'use_dynamic': self.use_dynamic_benchmarks,
                'dynamic_count': dynamic_count,
                'total_sectors': len(SECTOR_BENCHMARKS),
                'last_updated': datetime.now().isoformat() if self._dynamic_benchmarks else None
            }
        }

    def _format_market_cap(self, market_cap: float) -> str:
        if market_cap >= 1_000_000_000_000:
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"${market_cap / 1_000_000:.2f}M"
        return f"${market_cap:,.0f}"
    
    def _calculate_ensemble_growth(
        self, 
        lstm_growth: float, 
        data: Dict,
        lstm_weight: float = 0.50,
        historical_weight: float = 0.30,
        sector_weight: float = 0.20
    ) -> float:
        """
        Calculate ensemble growth rate combining LSTM, historical, and sector averages.
        
        This tempers extreme LSTM predictions with more conservative estimates.
        
        Args:
            lstm_growth: LSTM predicted growth rate (as decimal, e.g., 0.15 for 15%)
            data: Stock data dictionary with revenue_growth, earnings_growth, sector
            lstm_weight: Weight for LSTM prediction (default 50%)
            historical_weight: Weight for historical growth (default 30%)
            sector_weight: Weight for sector average (default 20%)
            
        Returns:
            Blended growth rate as decimal
        """
        # Get historical growth (average of revenue and earnings growth)
        revenue_growth = data.get('revenue_growth', 0) / 100  # Convert from % to decimal
        earnings_growth = data.get('earnings_growth', 0) / 100
        
        # Use average of revenue and earnings, defaulting to LSTM if no data
        if revenue_growth != 0 or earnings_growth != 0:
            historical_growth = (revenue_growth + earnings_growth) / 2
            # Clip historical to reasonable range
            historical_growth = np.clip(historical_growth, -0.30, 0.50)
        else:
            historical_growth = lstm_growth  # Fall back to LSTM
            historical_weight = 0  # Give weight to LSTM instead
            lstm_weight += 0.30
        
        # Get sector average growth
        sector = data.get('sector', 'Unknown')
        sector_benchmarks = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])
        sector_growth = sector_benchmarks.get('avg_growth', 7.0) / 100  # Convert to decimal
        
        # Normalize weights to sum to 1
        total_weight = lstm_weight + historical_weight + sector_weight
        if total_weight > 0:
            lstm_weight /= total_weight
            historical_weight /= total_weight
            sector_weight /= total_weight
        
        # Calculate weighted ensemble
        ensemble_growth = (
            lstm_weight * lstm_growth +
            historical_weight * historical_growth +
            sector_weight * sector_growth
        )
        
        # Clip final result to reasonable range
        ensemble_growth = np.clip(ensemble_growth, -0.30, 0.50)
        
        self.logger.debug(
            f"Ensemble growth: LSTM={lstm_growth*100:.1f}%, Historical={historical_growth*100:.1f}%, "
            f"Sector={sector_growth*100:.1f}% → Ensemble={ensemble_growth*100:.1f}%"
        )
        
        return ensemble_growth
    
    def _calculate_lstm_fair_value(self, ticker: str, data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate fair value using LSTM-DCF model with CORRECT feature extraction
        
        The LSTM model was trained on quarterly financial statement data:
        - revenue, capex, da, fcf, operating_cf, ebitda, total_assets, net_income
        - operating_income, operating_margin, net_margin, fcf_margin, ebitda_margin
        - revenue_per_asset, fcf_per_asset, ebitda_per_asset
        
        Returns:
            Tuple of (fair_value, predicted_growth_rate_percentage)
        """
        if not self.lstm_model:
            return None, None
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly financial statements (what the model was trained on)
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow
            
            if income_stmt.empty or cash_flow.empty:
                self.logger.debug(f"{ticker}: No quarterly financials available")
                return None, None
            
            # Build feature sequence from quarterly data
            # Need at least 8 quarters (2 years) for meaningful sequence
            n_quarters = min(len(income_stmt.columns), len(cash_flow.columns), 8)
            
            if n_quarters < 4:
                self.logger.debug(f"{ticker}: Not enough quarterly data ({n_quarters} quarters)")
                return None, None
            
            features_list = []
            
            for i in range(n_quarters):
                try:
                    # Extract values from each quarter (columns are dates, most recent first)
                    col_is = income_stmt.columns[i] if i < len(income_stmt.columns) else None
                    col_cf = cash_flow.columns[i] if i < len(cash_flow.columns) else None
                    col_bs = balance_sheet.columns[i] if i < len(balance_sheet.columns) else None
                    
                    # Income statement items
                    revenue = self._safe_get(income_stmt, 'Total Revenue', col_is, 0)
                    net_income = self._safe_get(income_stmt, 'Net Income', col_is, 0)
                    operating_income = self._safe_get(income_stmt, 'Operating Income', col_is, 0)
                    ebitda = self._safe_get(income_stmt, 'EBITDA', col_is, 
                              self._safe_get(income_stmt, 'Normalized EBITDA', col_is, 0))
                    
                    # Cash flow items
                    operating_cf = self._safe_get(cash_flow, 'Operating Cash Flow', col_cf,
                                    self._safe_get(cash_flow, 'Cash Flow From Continuing Operating Activities', col_cf, 0))
                    capex = abs(self._safe_get(cash_flow, 'Capital Expenditure', col_cf, 0))
                    da = self._safe_get(cash_flow, 'Depreciation And Amortization', col_cf,
                          self._safe_get(cash_flow, 'Depreciation', col_cf, 0))
                    
                    fcf = operating_cf - capex
                    
                    # Balance sheet
                    total_assets = self._safe_get(balance_sheet, 'Total Assets', col_bs, 1)
                    
                    # Calculate margins
                    operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
                    net_margin = (net_income / revenue * 100) if revenue > 0 else 0
                    fcf_margin = (fcf / revenue * 100) if revenue > 0 else 0
                    ebitda_margin = (ebitda / revenue * 100) if revenue > 0 else 0
                    
                    # Asset efficiency ratios
                    revenue_per_asset = revenue / total_assets if total_assets > 0 else 0
                    fcf_per_asset = fcf / total_assets if total_assets > 0 else 0
                    ebitda_per_asset = ebitda / total_assets if total_assets > 0 else 0
                    
                    # Feature vector matching training order
                    quarter_features = [
                        revenue, capex, da, fcf, operating_cf, ebitda,
                        total_assets, net_income, operating_income,
                        operating_margin, net_margin, fcf_margin, ebitda_margin,
                        revenue_per_asset, fcf_per_asset, ebitda_per_asset
                    ]
                    
                    features_list.append(quarter_features)
                    
                except Exception as e:
                    self.logger.debug(f"{ticker} Q{i} feature extraction error: {e}")
                    continue
            
            if len(features_list) < 4:
                return None, None
            
            # Reverse to chronological order (oldest first, like training)
            features_list = features_list[::-1]
            
            # Pad to sequence length if needed (repeat last quarter)
            while len(features_list) < self.lstm_sequence_length:
                features_list.append(features_list[-1])
            
            # Use only last N quarters
            features_list = features_list[-self.lstm_sequence_length:]
            
            features_array = np.array(features_list, dtype=np.float32)
            
            # Replace NaN/inf with 0
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply the SAVED StandardScaler from training
            if self.lstm_scaler is not None:
                features_scaled = self.lstm_scaler.transform(features_array)
            else:
                # Fallback: simple standardization
                features_scaled = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
            
            X_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.lstm_model(X_tensor)
            
            # Model outputs [revenue_growth, fcf_growth] as percentages
            # Model already clamps to [-50, +100] range in forward()
            if prediction.shape[1] == 2:
                revenue_growth_pred = prediction[0, 0].item()
                fcf_growth_pred = prediction[0, 1].item()
                # Use FCF growth for valuation
                predicted_growth = fcf_growth_pred
            else:
                predicted_growth = prediction[0, 0].item()
            
            # The model outputs are in percentage terms (-50 to +100 range)
            # Convert to decimal and apply conservative clip
            lstm_growth_rate = predicted_growth / 100.0  # Convert from % to decimal
            lstm_growth_rate = np.clip(lstm_growth_rate, -0.30, 0.50)  # -30% to +50%
            
            # USE ENSEMBLE AVERAGING to temper extreme predictions
            # Combines LSTM (50%), Historical (30%), Sector Average (20%)
            predicted_growth_rate = self._calculate_ensemble_growth(lstm_growth_rate, data)
            
            # Get current price for sanity checks
            current_price = data['current_price']
            market_cap = data['market_cap']
            
            # Calculate fair value using Gordon Growth Model
            fcf = data.get('free_cash_flow', 0)
            
            shares = market_cap / current_price if current_price > 0 else 1
            fcf_per_share = fcf / shares if shares > 0 else 0
            
            wacc = 0.10  # 10% discount rate
            terminal_growth = min(max(predicted_growth_rate * 0.3, 0.01), 0.03)  # 1-3%
            
            fair_value = None
            
            if fcf_per_share > 0 and predicted_growth_rate < wacc - terminal_growth:
                # Gordon Growth: FV = FCF * (1 + g) / (WACC - g)
                fair_value = (fcf_per_share * (1 + predicted_growth_rate)) / (wacc - terminal_growth)
            else:
                # EPS-based fallback
                eps = data.get('eps', 0)
                if eps > 0:
                    peg_adjusted_pe = 15 * (1 + predicted_growth_rate)
                    peg_adjusted_pe = np.clip(peg_adjusted_pe, 8, 30)  # Reasonable P/E range
                    fair_value = eps * peg_adjusted_pe
            
            # Light sanity check: prevent extreme outliers but allow model predictions
            if fair_value is not None and fair_value > 0:
                # Only reject truly extreme predictions (>10x or <0.1x)
                ratio = fair_value / current_price
                if ratio > 10 or ratio < 0.1:
                    self.logger.debug(
                        f"{ticker}: Extreme fair value ${fair_value:.2f} (ratio {ratio:.1f}x), using fallback"
                    )
                    return None, None
                
                return round(fair_value, 2), round(predicted_growth_rate * 100, 2)
            
            return None, None
            
        except Exception as e:
            self.logger.debug(f"LSTM calculation failed for {ticker}: {e}")
            return None, None
    
    def _safe_get(self, df: pd.DataFrame, row_name: str, col, default=0):
        """Safely extract value from financial statement DataFrame"""
        try:
            if col is None or df.empty:
                return default
            if row_name in df.index:
                val = df.loc[row_name, col]
                if pd.notna(val):
                    return float(val)
            return default
        except Exception:
            return default
    
    def _calculate_traditional_fair_value(self, data: Dict) -> float:
        """Calculate traditional DCF fair value using P/E and P/B multiples"""
        current_price = data['current_price']
        pe_ratio = data['pe_ratio']
        pb_ratio = data['pb_ratio']
        eps = data.get('eps', 0)
        book_value = data.get('book_value', 0)
        
        # Sector-adjusted fair P/E
        sector = data.get('sector', 'Unknown')
        sector_pe = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])['avg_pe']
        
        # P/E based fair value
        if eps > 0:
            pe_fair_value = eps * sector_pe
        else:
            pe_fair_value = current_price
        
        # P/B based fair value
        sector_pb = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])['avg_pb']
        if book_value > 0:
            pb_fair_value = book_value * sector_pb
        else:
            pb_fair_value = current_price
        
        # Weighted average
        return round((pe_fair_value * 0.6 + pb_fair_value * 0.3 + current_price * 0.1), 2)
    
    def _calculate_margin_of_safety(self, current_price: float, fair_value: float) -> float:
        """Calculate margin of safety (upside potential)
        
        Returns raw percentage - extreme values serve as guidance indicators.
        High MoS (>100%) suggests significant model conviction or uncertainty.
        """
        if current_price <= 0 or fair_value <= 0:
            return 0
        mos = ((fair_value - current_price) / current_price) * 100
        return round(mos, 2)
    
    def _generate_justification(self, data: Dict, score: float, sector_data: Dict) -> str:
        """Generate comprehensive justification for the ranking
        
        Enhanced for long-term value investing with forward-looking analysis.
        """
        justifications = []
        warnings = []
        
        # === VALUATION ANALYSIS ===
        mos = data.get('margin_of_safety', 0)
        if mos > 30:
            justifications.append(f"🎯 Significantly undervalued: {mos:.0f}% upside to fair value")
        elif mos > 10:
            justifications.append(f"📈 Undervalued: {mos:.0f}% margin of safety")
        elif mos > 0:
            justifications.append(f"Slightly undervalued: {mos:.0f}% upside potential")
        elif mos > -10:
            warnings.append("⚠️ Near fair value (limited upside)")
        else:
            warnings.append(f"⚠️ Appears overvalued by {abs(mos):.0f}%")
        
        # === FORWARD-LOOKING P/E ANALYSIS ===
        forward_pe = data.get('forward_pe', 0)
        trailing_pe = data.get('pe_ratio', 0)
        sector_pe = sector_data.get('avg_pe', 20)
        
        if forward_pe > 0 and trailing_pe > 0:
            if forward_pe < trailing_pe * 0.9:
                justifications.append(f"📉 Forward P/E ({forward_pe:.1f}) < Trailing ({trailing_pe:.1f}) suggests earnings growth ahead")
            elif forward_pe > trailing_pe * 1.1:
                warnings.append(f"Forward P/E ({forward_pe:.1f}) > Trailing may signal declining earnings")
        
        if forward_pe > 0:
            fpe_vs_sector = ((forward_pe / sector_pe) - 1) * 100
            if fpe_vs_sector < -20:
                justifications.append(f"Forward P/E {abs(fpe_vs_sector):.0f}% below sector average")
        elif trailing_pe > 0:
            pe_diff = ((sector_pe - trailing_pe) / sector_pe) * 100
            if pe_diff > 20:
                justifications.append(f"P/E of {trailing_pe:.1f} is {abs(pe_diff):.0f}% below sector avg ({sector_pe:.0f})")
        
        # === GROWTH METRICS (PEG) ===
        peg = data.get('peg_ratio', 0)
        if 0 < peg < 1.0:
            justifications.append(f"🚀 Excellent PEG of {peg:.2f} (growth at attractive price)")
        elif 0 < peg < 1.5:
            justifications.append(f"Good PEG of {peg:.2f} (reasonable valuation for growth)")
        elif peg > 3.0:
            warnings.append(f"High PEG of {peg:.2f} (expensive for growth rate)")
        
        # === EARNINGS & REVENUE GROWTH ===
        earnings_growth = data.get('earnings_growth', 0)
        revenue_growth = data.get('revenue_growth', 0)
        
        if earnings_growth > 20:
            justifications.append(f"💪 Strong earnings growth: {earnings_growth:.0f}%")
        elif earnings_growth > 10:
            justifications.append(f"Solid earnings growth: {earnings_growth:.0f}%")
        elif earnings_growth < -10:
            warnings.append(f"Declining earnings: {earnings_growth:.0f}%")
        
        if revenue_growth > 15:
            justifications.append(f"Growing revenues: {revenue_growth:.0f}% YoY")
        
        # === PROFITABILITY & QUALITY ===
        roe = data.get('roe', 0)
        sector_roe = sector_data.get('avg_roe', 15)
        profit_margin = data.get('profit_margin', 0)
        
        if roe > sector_roe * 1.3:
            justifications.append(f"Superior ROE of {roe:.1f}% (sector: {sector_roe:.0f}%)")
        elif roe > sector_roe:
            justifications.append(f"Above-average ROE: {roe:.1f}%")
        
        if profit_margin > 20:
            justifications.append(f"High profit margin: {profit_margin:.1f}%")
        
        # === CASH FLOW ===
        fcf_yield = data.get('fcf_yield', 0)
        if fcf_yield > 8:
            justifications.append(f"💰 Excellent FCF yield: {fcf_yield:.1f}%")
        elif fcf_yield > 5:
            justifications.append(f"Strong FCF yield: {fcf_yield:.1f}%")
        elif fcf_yield < 0:
            warnings.append("Negative free cash flow")
        
        # === FINANCIAL HEALTH ===
        de = data.get('debt_equity', 0)
        if de < 0.3:
            justifications.append(f"Conservative balance sheet (D/E: {de:.2f})")
        elif de > 1.5:
            warnings.append(f"⚠️ High leverage (D/E: {de:.2f})")
        
        # === RISK METRICS ===
        beta = data.get('beta', 1.0)
        if beta < 0.8:
            justifications.append(f"Defensive stock (Beta: {beta:.2f})")
        elif beta > 1.4:
            warnings.append(f"Higher volatility (Beta: {beta:.2f})")
        
        # === TECHNICAL CONTEXT ===
        pct_from_high = data.get('pct_from_52w_high', 0)
        if pct_from_high < -30:
            justifications.append(f"Trading {abs(pct_from_high):.0f}% below 52-week high (potential rebound)")
        elif pct_from_high < -15:
            justifications.append(f"Pulled back {abs(pct_from_high):.0f}% from 52-week high")
        
        # Combine justifications, prioritizing positives but including warnings
        combined = justifications[:5]  # Top 5 positives
        if warnings and len(combined) < 6:
            combined.extend(warnings[:2])  # Add up to 2 warnings
        
        if not combined:
            combined.append("Meets fundamental screening criteria")
        
        return "; ".join(combined)
    
    def _fetch_stock_data(self, ticker: str) -> Optional[Dict]:
        """Fetch comprehensive stock data including fair value calculations"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'marketCap' not in info:
                return None
            
            hist = stock.history(period="1y")
            if hist.empty:
                return None
            
            current_price = info.get('currentPrice', info.get('regularMarketPrice', hist['Close'].iloc[-1]))
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            
            # 52-week metrics
            high_52w = info.get('fiftyTwoWeekHigh', hist['High'].max())
            low_52w = info.get('fiftyTwoWeekLow', hist['Low'].min())
            pct_from_high = ((current_price - high_52w) / high_52w * 100) if high_52w > 0 else 0
            
            # YTD return
            ytd_hist = stock.history(period="ytd")
            ytd_return = 0
            if not ytd_hist.empty and len(ytd_hist) > 1:
                ytd_return = ((ytd_hist['Close'].iloc[-1] - ytd_hist['Close'].iloc[0]) / 
                             ytd_hist['Close'].iloc[0] * 100)
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 10 else 0
            
            # Moving Averages (50-day and 200-day)
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
            
            # Price position relative to MAs
            price_vs_50ma = ((current_price - sma_50) / sma_50 * 100) if sma_50 else None
            price_vs_200ma = ((current_price - sma_200) / sma_200 * 100) if sma_200 else None
            
            # FCF ROIC calculation (FCF / Invested Capital)
            # Invested Capital = Total Assets - Current Liabilities
            # Need to fetch from balance sheet since info dict doesn't have these
            free_cash_flow = info.get('freeCashflow', 0) or 0
            total_assets = 0
            current_liabilities = 0
            try:
                balance_sheet = stock.quarterly_balance_sheet
                if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                    latest_col = balance_sheet.columns[0]  # Most recent quarter
                    total_assets = balance_sheet.loc['Total Assets', latest_col] if 'Total Assets' in balance_sheet.index else 0
                    current_liabilities = balance_sheet.loc['Current Liabilities', latest_col] if 'Current Liabilities' in balance_sheet.index else 0
            except Exception as e:
                self.logger.debug(f"{ticker}: Could not fetch balance sheet: {e}")
            
            invested_capital = total_assets - current_liabilities
            fcf_roic = (free_cash_flow / invested_capital * 100) if invested_capital > 0 else 0
            
            # Next year revenue growth estimate from analysts
            # yfinance provides growth estimates in info dict
            next_year_rev_growth = None
            try:
                growth_estimates = info.get('revenueGrowth', 0) or 0
                # Try to get analyst estimates if available
                analyst_rev_growth = info.get('revenueQuarterlyGrowth', 0) or 0
                next_year_rev_growth = (growth_estimates * 100) if growth_estimates else (analyst_rev_growth * 100)
            except Exception:
                next_year_rev_growth = 0
            
            # Base data
            data = {
                'ticker': ticker,
                'company_name': info.get('shortName', info.get('longName', ticker)),
                'sector': sector,
                'industry': info.get('industry', 'Unknown'),
                'current_price': round(current_price, 2),
                'market_cap': market_cap,
                'market_cap_formatted': self._format_market_cap(market_cap),
                
                # Valuation
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'pb_ratio': info.get('priceToBook', 0) or 0,
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0) or 0,
                'peg_ratio': info.get('pegRatio', 0) or 0,
                'ev_ebitda': info.get('enterpriseToEbitda', 0) or 0,
                
                # Financial health
                'debt_equity': (info.get('debtToEquity', 0) or 0) / 100,
                'current_ratio': info.get('currentRatio', 0) or 0,
                'quick_ratio': info.get('quickRatio', 0) or 0,
                
                # Profitability
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'roa': (info.get('returnOnAssets', 0) or 0) * 100,
                'profit_margin': (info.get('profitMargins', 0) or 0) * 100,
                'gross_margin': (info.get('grossMargins', 0) or 0) * 100,
                'operating_margin': (info.get('operatingMargins', 0) or 0) * 100,
                
                # Growth
                'revenue_growth': (info.get('revenueGrowth', 0) or 0) * 100,
                'earnings_growth': (info.get('earningsGrowth', 0) or 0) * 100,
                
                # Cash flow
                'free_cash_flow': info.get('freeCashflow', 0) or 0,
                'operating_cash_flow': info.get('operatingCashflow', 0) or 0,
                'fcf_yield': ((info.get('freeCashflow', 0) or 0) / market_cap * 100) if market_cap > 0 else 0,
                
                # Capital Efficiency (Phase 3)
                'total_assets': total_assets,
                'current_liabilities': current_liabilities,
                'invested_capital': invested_capital,
                'fcf_roic': round(fcf_roic, 2),
                
                # Next year estimates (Phase 3)
                'next_year_revenue_growth': round(next_year_rev_growth, 2) if next_year_rev_growth else None,
                
                # Moving Averages (Phase 3)
                'sma_50': round(sma_50, 2) if sma_50 else None,
                'sma_200': round(sma_200, 2) if sma_200 else None,
                'price_vs_50ma': round(price_vs_50ma, 2) if price_vs_50ma else None,
                'price_vs_200ma': round(price_vs_200ma, 2) if price_vs_200ma else None,
                
                # Dividend
                'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                
                # Risk
                'beta': info.get('beta', 1.0) or 1.0,
                'volatility': round(volatility, 2),
                
                # Performance
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
                'pct_from_52w_high': round(pct_from_high, 2),
                'ytd_return': round(ytd_return, 2),
                
                # Volume
                'avg_volume': info.get('averageVolume', 0) or 0,
                
                # Analyst
                'analyst_target': info.get('targetMeanPrice', 0) or 0,
                
                # Additional for fair value
                'eps': info.get('trailingEps', 0) or 0,
                'book_value': info.get('bookValue', 0) or 0,
                
                # Sector benchmarks
                'sector_avg_pe': SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])['avg_pe'],
                'sector_avg_pb': SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])['avg_pb'],
                'sector_avg_roe': SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])['avg_roe'],
            }
            
            # Calculate fair values
            # 1. Traditional DCF
            data['traditional_fair_value'] = self._calculate_traditional_fair_value(data)
            
            # 2. LSTM-DCF Fair Value
            lstm_fv, lstm_growth = self._calculate_lstm_fair_value(ticker, data)
            data['lstm_fair_value'] = lstm_fv
            data['lstm_predicted_growth'] = lstm_growth
            
            # 3. Select best fair value
            if lstm_fv and lstm_fv > 0:
                # Use LSTM if reasonable
                if 0.2 < lstm_fv / current_price < 5:
                    data['fair_value'] = lstm_fv
                    data['fair_value_method'] = 'LSTM-DCF'
                else:
                    data['fair_value'] = data['traditional_fair_value']
                    data['fair_value_method'] = 'Traditional DCF (LSTM out of range)'
            else:
                data['fair_value'] = data['traditional_fair_value']
                data['fair_value_method'] = 'Traditional DCF'
            
            # 4. Margin of Safety
            data['margin_of_safety'] = self._calculate_margin_of_safety(current_price, data['fair_value'])
            
            # Analyst upside
            if data['analyst_target'] > 0:
                data['analyst_upside'] = round(((data['analyst_target'] - current_price) / current_price * 100), 2)
            else:
                data['analyst_upside'] = 0
            
            data['last_updated'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            self.logger.debug(f"Error fetching {ticker}: {e}")
            return None
    
    def _apply_filters(self, data: Dict) -> bool:
        """Apply screening filters"""
        try:
            if data['market_cap'] < self.criteria['min_market_cap']:
                return False
            
            pe = data['pe_ratio']
            if pe <= self.criteria['min_pe_ratio'] or pe > self.criteria['max_pe_ratio']:
                if pe != 0:
                    return False
            
            if data['debt_equity'] > self.criteria['max_debt_equity']:
                return False
            
            if data['avg_volume'] < self.criteria['min_volume']:
                return False
            
            if data['roe'] < self.criteria['min_roe']:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_valuation_score(self, data: Dict) -> float:
        """Calculate comprehensive valuation score with forward-looking metrics
        
        Enhanced for long-term value investing focus:
        - Forward P/E gets higher weight than trailing P/E
        - PEG ratio rewards growth at reasonable price
        - Margin trend rewards improving profitability
        - Margin of Safety only contributes positively when actually undervalued
        """
        scores = []
        weights = []
        sector = data.get('sector', 'Unknown')
        sector_data = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])
        
        # === FORWARD-LOOKING METRICS (40% total) ===
        
        # Forward P/E Score (15%) - CRITICAL for long-term value
        forward_pe = data.get('forward_pe', 0)
        trailing_pe = data.get('pe_ratio', 0)
        sector_pe = sector_data['avg_pe']
        
        if 0 < forward_pe <= 100:
            # Compare forward P/E to sector average
            fpe_ratio_to_sector = forward_pe / sector_pe
            if fpe_ratio_to_sector < 0.5:
                fpe_score = 100
            elif fpe_ratio_to_sector < 0.75:
                fpe_score = 90
            elif fpe_ratio_to_sector < 1.0:
                fpe_score = 75
            elif fpe_ratio_to_sector < 1.25:
                fpe_score = 55
            else:
                fpe_score = 35
            
            # Bonus if forward P/E < trailing P/E (earnings expected to grow)
            if trailing_pe > 0 and forward_pe < trailing_pe * 0.9:
                fpe_score = min(100, fpe_score + 10)
            
            scores.append(fpe_score)
            weights.append(0.15)
        elif 0 < trailing_pe <= 100:
            # Fallback to trailing P/E if no forward P/E available
            pe_ratio_to_sector = trailing_pe / sector_pe
            if pe_ratio_to_sector < 0.5:
                pe_score = 90
            elif pe_ratio_to_sector < 0.75:
                pe_score = 75
            elif pe_ratio_to_sector < 1.0:
                pe_score = 60
            elif pe_ratio_to_sector < 1.25:
                pe_score = 45
            else:
                pe_score = 30
            scores.append(pe_score)
            weights.append(0.15)
        
        # PEG Ratio Score (15%) - Growth at Reasonable Price
        peg = data.get('peg_ratio', 0)
        if 0 < peg <= 5:
            if peg < 0.5:
                peg_score = 100  # Extremely undervalued for growth
            elif peg < 1.0:
                peg_score = 90   # Classic value + growth
            elif peg < 1.5:
                peg_score = 70   # Fair value for growth
            elif peg < 2.0:
                peg_score = 50   # Slightly expensive
            else:
                peg_score = 30   # Expensive relative to growth
            scores.append(peg_score)
            weights.append(0.15)
        
        # Earnings Growth Score (10%) - Forward-looking momentum
        earnings_growth = data.get('earnings_growth', 0)
        revenue_growth = data.get('revenue_growth', 0)
        if earnings_growth != 0 or revenue_growth != 0:
            # Average of earnings and revenue growth
            avg_growth = (earnings_growth + revenue_growth) / 2
            if avg_growth > 25:
                growth_score = 100
            elif avg_growth > 15:
                growth_score = 85
            elif avg_growth > 8:
                growth_score = 70
            elif avg_growth > 0:
                growth_score = 55
            else:
                growth_score = 35
            scores.append(growth_score)
            weights.append(0.10)
        
        # === VALUATION METRICS (25% total) ===
        
        # P/B Score (10%)
        pb = data['pb_ratio']
        sector_pb = sector_data['avg_pb']
        if 0 < pb <= 30:
            pb_ratio_to_sector = pb / sector_pb
            if pb_ratio_to_sector < 0.5:
                pb_score = 100
            elif pb_ratio_to_sector < 0.8:
                pb_score = 80
            elif pb_ratio_to_sector < 1.2:
                pb_score = 60
            else:
                pb_score = 40
            scores.append(pb_score)
            weights.append(0.10)
        
        # Margin of Safety Score (15%) - STRICT: only positive MoS contributes
        mos = data.get('margin_of_safety', 0)
        if mos > 50:
            mos_score = 100
        elif mos > 30:
            mos_score = 90
        elif mos > 15:
            mos_score = 75
        elif mos > 5:
            mos_score = 60
        elif mos > 0:
            mos_score = 50   # Small positive MoS - neutral
        elif mos > -10:
            mos_score = 35   # Slightly overvalued - penalty
        elif mos > -25:
            mos_score = 20   # Overvalued - significant penalty
        else:
            mos_score = 10   # Heavily overvalued - major penalty
        scores.append(mos_score)
        weights.append(0.15)
        
        # === CASH FLOW & PROFITABILITY (25% total) ===
        
        # FCF Yield Score (12%)
        fcf_yield = data['fcf_yield']
        if fcf_yield >= 10:
            fcf_score = 100
        elif fcf_yield >= 6:
            fcf_score = 85
        elif fcf_yield >= 3:
            fcf_score = 70
        elif fcf_yield >= 0:
            fcf_score = 50
        else:
            fcf_score = 30
        scores.append(fcf_score)
        weights.append(0.12)
        
        # Profitability Score (13%) - Uses ROE and profit margin
        roe = data['roe']
        profit_margin = data.get('profit_margin', 0)
        sector_roe = sector_data['avg_roe']
        
        # ROE component
        if roe > sector_roe * 1.5:
            roe_component = 50
        elif roe > sector_roe:
            roe_component = 40
        elif roe > sector_roe * 0.7:
            roe_component = 30
        elif roe > 0:
            roe_component = 20
        else:
            roe_component = 10
        
        # Profit margin component - bonus for high margins
        if profit_margin > 20:
            margin_component = 50
        elif profit_margin > 15:
            margin_component = 40
        elif profit_margin > 10:
            margin_component = 30
        elif profit_margin > 5:
            margin_component = 20
        else:
            margin_component = 10
        
        prof_score = roe_component + margin_component
        scores.append(prof_score)
        weights.append(0.13)
        
        # === FINANCIAL HEALTH (10% total) ===
        
        # Financial Health Score (10%)
        de = data['debt_equity']
        cr = data['current_ratio']
        health = 50
        if de < 0.3:
            health += 30
        elif de < 0.7:
            health += 15
        elif de > 1.5:
            health -= 15
        if cr > 2.0:
            health += 20
        elif cr > 1.5:
            health += 10
        elif cr < 1.0:
            health -= 20
        health = max(0, min(100, health))
        scores.append(health)
        weights.append(0.10)
        
        if not scores:
            return 50.0
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0
    
    def _calculate_garp_score(self, data: Dict) -> float:
        """
        Calculate GARP (Growth At Reasonable Price) score
        TRANSPARENT replacement for RF Ensemble
        
        RF Ensemble was found to use P/E at 99.93% importance, making it
        essentially just a P/E filter. This explicit GARP scoring is:
        1. More transparent (no black-box model)
        2. Forward-looking (uses Forward P/E)
        3. Growth-adjusted (uses PEG ratio)
        
        Scoring components:
        - Forward P/E: 50% (lower is better, < 15 = excellent, > 30 = poor)
        - PEG Ratio: 50% (lower is better, < 1 = excellent, 1-2 = good, > 3 = poor)
        
        Returns 0-1 normalized score (higher = better value)
        """
        try:
            # Get forward-looking metrics
            forward_pe = data.get('forward_pe', 0)
            trailing_pe = data.get('pe_ratio', 20)
            peg_ratio = data.get('peg_ratio', 2.0)
            earnings_growth = data.get('earnings_growth', 10)
            
            # Use forward P/E if available, else trailing
            effective_pe = forward_pe if 0 < forward_pe < 200 else trailing_pe
            
            # === Forward P/E Score (50%) ===
            if effective_pe <= 0:
                pe_score = 0.3  # Negative earnings, neutral-low
            elif effective_pe < 10:
                pe_score = 1.0
            elif effective_pe < 15:
                pe_score = 1.0 - (effective_pe - 10) * 0.04  # 1.0 to 0.8
            elif effective_pe < 25:
                pe_score = 0.8 - (effective_pe - 15) * 0.03  # 0.8 to 0.5
            elif effective_pe < 40:
                pe_score = 0.5 - (effective_pe - 25) * 0.02  # 0.5 to 0.2
            else:
                pe_score = max(0.2 - (effective_pe - 40) * 0.005, 0)
            
            # === PEG Ratio Score (50%) ===
            if peg_ratio <= 0 or peg_ratio > 50:  # Invalid PEG
                if earnings_growth > 20:
                    peg_score = 0.7
                elif earnings_growth > 10:
                    peg_score = 0.5
                else:
                    peg_score = 0.3
            elif peg_ratio < 0.5:
                peg_score = 0.7  # Too cheap might signal problems
            elif peg_ratio < 1.0:
                peg_score = 1.0  # Ideal GARP zone
            elif peg_ratio < 2.0:
                peg_score = 1.0 - (peg_ratio - 1.0) * 0.3  # 1.0 to 0.7
            elif peg_ratio < 3.0:
                peg_score = 0.7 - (peg_ratio - 2.0) * 0.3  # 0.7 to 0.4
            else:
                peg_score = max(0.4 - (peg_ratio - 3.0) * 0.1, 0)
            
            # Combine 50/50
            garp_score = (pe_score * 0.5) + (peg_score * 0.5)
            
            return garp_score
            
        except Exception as e:
            self.logger.debug(f"GARP score error: {e}")
            return 0.5
    
    def _calculate_consensus_score(self, data: Dict, lstm_score: float, garp_score: float) -> Dict:
        """Calculate consensus score using multi-model weighted voting
        
        ARCHITECTURE (Jan 2025): LSTM 50% + GARP 25% + Risk 25%
        RF Ensemble deprecated (99% P/E), replaced with transparent GARP.
        
        CRITICAL FIX: Heavily penalize negative margin of safety
        - Stocks with MoS < 0 (overvalued) get severely reduced consensus scores
        """
        if not self.consensus_scorer:
            return {'consensus_score': lstm_score * 100, 'confidence': 0.5}
        
        try:
            mos = data.get('margin_of_safety', 0)
            
            # === LSTM/Fair Value Score (50% weight) ===
            # STRICT normalization: Only positive MoS contributes meaningfully
            if mos > 0:
                # Positive MoS: maps 0-100% MoS to 0.5-1.0 score
                lstm_normalized = 0.5 + min(mos / 100, 0.5)
            else:
                # Negative MoS: maps 0 to -50% MoS to 0.25-0 score (heavy penalty)
                lstm_normalized = max(0.25 + (mos / 100), 0)
            
            # === GARP Score (25% weight) - Replaces RF ===
            # Transparent Forward P/E + PEG scoring
            garp_score = self._calculate_garp_score(data)
            
            # === Risk Score (25% weight) ===
            # Lower beta/volatility = safer
            beta = data.get('beta', 1.0)
            vol = data.get('volatility', 30)
            risk_score = 1 - min((beta + vol/100) / 2, 1)
            
            # Calculate consensus with NEW weights
            model_scores = {
                'lstm_dcf': lstm_normalized,
                'garp_score': garp_score,  # Replaces rf_risk_sentiment
                'risk_score': risk_score
            }
            
            consensus_result = self.consensus_scorer.calculate_consensus(model_scores)
            
            # === PENALTY MULTIPLIER for overvalued stocks ===
            # Apply additional penalty to keep overvalued stocks out of top lists
            penalty_multiplier = 1.0
            if mos < -20:
                penalty_multiplier = 0.6  # 40% penalty for heavily overvalued
            elif mos < -10:
                penalty_multiplier = 0.75  # 25% penalty for moderately overvalued
            elif mos < 0:
                penalty_multiplier = 0.9  # 10% penalty for slightly overvalued
            
            final_consensus = consensus_result['consensus_score'] * penalty_multiplier
            
            # Scale consensus score to 0-100 to match valuation_score
            return {
                'consensus_score': final_consensus * 100,
                'confidence': consensus_result['confidence'],
                'mos_penalty_applied': penalty_multiplier < 1.0
            }
            
        except Exception as e:
            self.logger.debug(f"Consensus score error: {e}")
            return {'consensus_score': lstm_score * 100, 'confidence': 0.5}

    def _get_recommendation(self, score: float, data: Dict) -> str:
        mos = data.get('margin_of_safety', 0)
        analyst_upside = data.get('analyst_upside', 0)
        
        if score >= 80 and mos > 20:
            return "STRONG BUY"
        elif score >= 70 or (score >= 65 and analyst_upside > 20):
            return "BUY"
        elif score >= 60:
            return "ACCUMULATE"
        elif score >= 50:
            return "HOLD"
        elif score >= 40:
            return "REDUCE"
        else:
            return "SELL"
    
    def _get_assessment(self, score: float) -> str:
        if score >= 80:
            return "Significantly Undervalued"
        elif score >= 70:
            return "Undervalued"
        elif score >= 60:
            return "Slightly Undervalued"
        elif score >= 50:
            return "Fairly Valued"
        elif score >= 40:
            return "Slightly Overvalued"
        else:
            return "Overvalued"
    
    def _get_risk_level(self, data: Dict) -> str:
        risk = 0
        if data['beta'] > 1.3:
            risk += 2
        elif data['beta'] > 1.0:
            risk += 1
        if data['debt_equity'] > 1.0:
            risk += 2
        elif data['debt_equity'] > 0.5:
            risk += 1
        if data['volatility'] > 40:
            risk += 2
        elif data['volatility'] > 25:
            risk += 1
        
        if risk <= 2:
            return "LOW"
        elif risk <= 4:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_educational_insights(self, data: Dict) -> Dict:
        """
        Generate educational explanations for each metric.
        Designed for retail investors learning fundamental analysis.
        
        Returns dict with metric explanations in plain language.
        """
        insights = {}
        
        # P/E Ratio explanation
        pe = data.get('pe_ratio', 0)
        sector_pe = data.get('sector_avg_pe', 20)
        if pe > 0:
            if pe < sector_pe * 0.7:
                insights['pe_ratio'] = {
                    'value': pe,
                    'interpretation': 'Potentially Undervalued',
                    'explanation': f"The P/E ratio of {pe:.1f} is significantly below the sector average of {sector_pe:.0f}. This means you're paying less per dollar of earnings compared to similar companies. However, low P/E can also indicate growth concerns."
                }
            elif pe > sector_pe * 1.3:
                insights['pe_ratio'] = {
                    'value': pe,
                    'interpretation': 'Premium Valuation',
                    'explanation': f"The P/E ratio of {pe:.1f} is above the sector average of {sector_pe:.0f}. Investors are paying a premium, usually expecting higher future growth. Be cautious if growth doesn't materialize."
                }
            else:
                insights['pe_ratio'] = {
                    'value': pe,
                    'interpretation': 'Fairly Valued',
                    'explanation': f"The P/E ratio of {pe:.1f} is in line with the sector average of {sector_pe:.0f}. The stock appears fairly priced relative to earnings."
                }
        
        # Margin of Safety explanation
        mos = data.get('margin_of_safety', 0)
        if mos > 20:
            insights['margin_of_safety'] = {
                'value': mos,
                'interpretation': 'Significant Upside',
                'explanation': f"Our fair value estimate suggests {mos:.0f}% potential upside. This 'margin of safety' provides a cushion if our estimates are slightly off. Warren Buffett looks for stocks trading below intrinsic value."
            }
        elif mos > 0:
            insights['margin_of_safety'] = {
                'value': mos,
                'interpretation': 'Modest Upside',
                'explanation': f"The stock trades {mos:.0f}% below our estimated fair value. Some upside exists, but less room for error in our analysis."
            }
        else:
            insights['margin_of_safety'] = {
                'value': mos,
                'interpretation': 'Limited Upside or Overvalued',
                'explanation': f"The stock trades near or above fair value. Consider if growth prospects justify current prices."
            }
        
        # Debt/Equity explanation
        de = data.get('debt_equity', 0)
        if de < 0.3:
            insights['debt_equity'] = {
                'value': de,
                'interpretation': 'Conservative Balance Sheet',
                'explanation': f"D/E ratio of {de:.2f} indicates low debt usage. The company is financially conservative, reducing bankruptcy risk but possibly missing growth opportunities."
            }
        elif de > 1.0:
            insights['debt_equity'] = {
                'value': de,
                'interpretation': 'High Leverage',
                'explanation': f"D/E ratio of {de:.2f} means significant debt. Higher debt amplifies both gains and losses. Interest payments may strain cash flow during downturns."
            }
        else:
            insights['debt_equity'] = {
                'value': de,
                'interpretation': 'Moderate Debt',
                'explanation': f"D/E ratio of {de:.2f} represents balanced capital structure, using some debt for growth while maintaining financial flexibility."
            }
        
        # Beta/Risk explanation
        beta = data.get('beta', 1.0)
        if beta < 0.8:
            insights['beta'] = {
                'value': beta,
                'interpretation': 'Low Volatility (Defensive)',
                'explanation': f"Beta of {beta:.2f} means the stock moves less than the market. Good for risk-averse investors seeking stability. May underperform in bull markets."
            }
        elif beta > 1.3:
            insights['beta'] = {
                'value': beta,
                'interpretation': 'High Volatility (Aggressive)',
                'explanation': f"Beta of {beta:.2f} means the stock is more volatile than the market. Larger potential gains but also larger potential losses."
            }
        else:
            insights['beta'] = {
                'value': beta,
                'interpretation': 'Market-Level Volatility',
                'explanation': f"Beta of {beta:.2f} indicates the stock moves roughly with the market. Normal risk profile."
            }
        
        # ROE explanation
        roe = data.get('roe', 0)
        if roe > 20:
            insights['roe'] = {
                'value': roe,
                'interpretation': 'Excellent Capital Efficiency',
                'explanation': f"ROE of {roe:.1f}% means the company generates strong returns on shareholder equity. High-quality businesses often maintain ROE above 15%."
            }
        elif roe > 10:
            insights['roe'] = {
                'value': roe,
                'interpretation': 'Adequate Returns',
                'explanation': f"ROE of {roe:.1f}% shows acceptable profitability. The company earns reasonable returns on invested capital."
            }
        else:
            insights['roe'] = {
                'value': roe,
                'interpretation': 'Weak Returns',
                'explanation': f"ROE of {roe:.1f}% suggests poor capital efficiency. Investigate if this is temporary or a structural issue."
            }
        
        # Overall investment thesis
        score = data.get('effective_score', data.get('valuation_score', 50))
        risk_level = data.get('risk_level', 'MEDIUM')
        
        if score >= 70 and risk_level == 'LOW':
            insights['overall'] = {
                'thesis': 'Attractive Low-Risk Opportunity',
                'explanation': 'This stock combines value characteristics with defensive risk metrics. Suitable for conservative investors seeking undervalued quality companies.'
            }
        elif score >= 70:
            insights['overall'] = {
                'thesis': 'High-Value with Risk',
                'explanation': 'Strong value metrics but elevated risk. Consider position sizing and portfolio diversification.'
            }
        elif score >= 50:
            insights['overall'] = {
                'thesis': 'Fair Value - Neutral',
                'explanation': 'The stock appears fairly priced. May be suitable for investors with specific sector views or dividend income needs.'
            }
        else:
            insights['overall'] = {
                'thesis': 'Caution Advised',
                'explanation': 'Valuation metrics suggest limited upside or overvaluation. Consider alternatives unless you have strong conviction.'
            }
        
        return insights

    def scan_market(self, progress_callback=None, use_cache: bool = True) -> pd.DataFrame:
        """Scan entire market with enhanced analysis and dynamic sector benchmarks"""
        # Check cache
        if use_cache and self.cache_enabled:
            cached = self._check_cache()
            if cached is not None:
                self.logger.info(f"Using cached results: {len(cached)} stocks")
                return cached
        
        self.logger.info(f"Scanning {len(self.tickers)} stocks...")
        start_time = datetime.now()
        
        raw_results = []  # First pass: collect raw data
        total = len(self.tickers)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_stock_data, ticker): ticker 
                for ticker in self.tickers
            }
            
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                ticker = future_to_ticker[future]
                
                if progress_callback:
                    progress_callback(i, total, ticker)
                
                try:
                    data = future.result()
                    if data and self._apply_filters(data):
                        raw_results.append(data)
                except Exception as e:
                    self.logger.debug(f"Error processing {ticker}: {e}")
        
        # Calculate dynamic sector benchmarks from scanned universe
        if self.use_dynamic_benchmarks and raw_results:
            self._dynamic_benchmarks = self.calculate_dynamic_sector_benchmarks(raw_results)
            self.update_global_benchmarks()
        
        # Second pass: apply scoring with updated benchmarks
        results = []
        for data in raw_results:
            sector = data.get('sector', 'Unknown')
            sector_data = self.get_sector_benchmark(sector)
            
            # Base valuation score (traditional metrics)
            data['valuation_score'] = self._calculate_valuation_score(data)
            
            # GARP score (transparent Forward P/E + PEG scoring)
            data['garp_score'] = self._calculate_garp_score(data)
            
            # Consensus scoring (LSTM 50% + GARP 25% + Risk 25%)
            if self.enable_consensus and self.consensus_scorer:
                consensus_result = self._calculate_consensus_score(
                    data,
                    lstm_score=data.get('margin_of_safety', 0),
                    garp_score=data.get('garp_score', 0.5)
                )
                data['consensus_score'] = consensus_result.get('consensus_score', data['valuation_score'])
                data['consensus_confidence'] = consensus_result.get('confidence', 0.5)
                # Use consensus score as primary if available
                effective_score = data['consensus_score']
            else:
                data['consensus_score'] = None
                data['consensus_confidence'] = None
                effective_score = data['valuation_score']
            
            data['effective_score'] = effective_score
            data['assessment'] = self._get_assessment(effective_score)
            data['recommendation'] = self._get_recommendation(effective_score, data)
            data['risk_level'] = self._get_risk_level(data)
            data['justification'] = self._generate_justification(data, effective_score, sector_data)
            
            # Add sector benchmark info for transparency
            data['sector_avg_pe'] = sector_data.get('avg_pe')
            data['sector_avg_pb'] = sector_data.get('avg_pb')
            data['benchmark_source'] = sector_data.get('source', 'default')
            
            results.append(data)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Scan complete: {len(results)}/{total} stocks in {elapsed:.1f}s")
        
        # Cache results
        self._save_cache(results)
        
        df = pd.DataFrame(results)
        if not df.empty:
            # Sort by effective_score (consensus if available, else valuation_score)
            sort_col = 'effective_score' if 'effective_score' in df.columns else 'valuation_score'
            df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        return df
    
    def _check_cache(self) -> Optional[pd.DataFrame]:
        cache_file = self.cache_dir / "enhanced_results.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.cache_expiry_hours):
                return None
            return pd.DataFrame(cached['results'])
        except Exception:
            return None
    
    def _save_cache(self, results: List[Dict]):
        if not self.cache_enabled:
            return
        try:
            cache_file = self.cache_dir / "enhanced_results.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(results),
                    'results': results
                }, f, default=str)
        except Exception as e:
            self.logger.debug(f"Cache save error: {e}")
    
    def get_top_undervalued(
        self, 
        n: int = 10, 
        rescan: bool = False, 
        include_ai_insights: bool = False,
        profile_id: Optional[str] = None,
        portfolio_value: Optional[float] = None
    ) -> List[Dict]:
        """Get top N undervalued stocks with comprehensive analysis
        
        ENHANCED: Now filters for TRUE undervalued stocks (positive margin of safety)
        For quality stocks regardless of valuation, use get_top_quality()
        
        Args:
            n: Number of stocks to return
            rescan: Force fresh market scan
            include_ai_insights: Include AI-generated insights
            profile_id: Optional risk profile for position sizing
            portfolio_value: Optional portfolio value for position sizing
        """
        df = self.scan_market(use_cache=not rescan)
        
        if df.empty:
            return []
        
        # CRITICAL FIX: Filter for actually undervalued stocks (MoS > 0)
        undervalued_df = df[df['margin_of_safety'] > 0].copy()
        
        # Sort by margin of safety for TRUE undervalued list
        if not undervalued_df.empty:
            undervalued_df = undervalued_df.sort_values('margin_of_safety', ascending=False)
            top_stocks = undervalued_df.head(n)
        else:
            # Fallback: if no undervalued stocks, return top by score with warning
            self.logger.warning("No stocks with positive margin of safety found")
            top_stocks = df.head(n)
        
        results = self._format_stock_results(
            top_stocks, 
            list_category='UNDERVALUED',
            profile_id=profile_id,
            portfolio_value=portfolio_value
        )
        return results
    
    def get_top_quality(
        self, 
        n: int = 10, 
        rescan: bool = False,
        profile_id: Optional[str] = None,
        portfolio_value: Optional[float] = None
    ) -> List[Dict]:
        """Get top N quality stocks by overall score (regardless of undervaluation)
        
        This list focuses on company quality metrics:
        - Strong financials
        - Good profitability
        - Low risk
        May include fairly valued or even slightly overvalued quality companies.
        """
        df = self.scan_market(use_cache=not rescan)
        
        if df.empty:
            return []
        
        # Sort by effective_score (quality-focused)
        top_stocks = df.sort_values('effective_score', ascending=False).head(n)
        results = self._format_stock_results(
            top_stocks, 
            list_category='QUALITY',
            profile_id=profile_id,
            portfolio_value=portfolio_value
        )
        return results
    
    def get_top_growth(
        self, 
        n: int = 10, 
        rescan: bool = False,
        profile_id: Optional[str] = None,
        portfolio_value: Optional[float] = None
    ) -> List[Dict]:
        """Get top N growth stocks with reasonable valuations (GARP approach)
        
        Filters for:
        - PEG < 2 (growth at reasonable price)
        - Earnings growth > 10%
        - Revenue growth > 5%
        """
        df = self.scan_market(use_cache=not rescan)
        
        if df.empty:
            return []
        
        # Filter for growth characteristics
        growth_df = df[
            (df['peg_ratio'] > 0) & (df['peg_ratio'] < 2.0) &
            (df['earnings_growth'] > 10) &
            (df['revenue_growth'] > 5)
        ].copy()
        
        if not growth_df.empty:
            # Sort by PEG ratio (lower is better for GARP)
            growth_df = growth_df.sort_values('peg_ratio', ascending=True)
            top_stocks = growth_df.head(n)
        else:
            # Fallback: top by earnings growth
            growth_fallback = df[df['earnings_growth'] > 0].sort_values('earnings_growth', ascending=False)
            top_stocks = growth_fallback.head(n) if not growth_fallback.empty else df.head(n)
        
        results = self._format_stock_results(
            top_stocks, 
            list_category='GROWTH',
            profile_id=profile_id,
            portfolio_value=portfolio_value
        )
        return results
    
    def get_categorized_watchlist(
        self, 
        n_per_category: int = 10, 
        rescan: bool = False,
        profile_id: Optional[str] = None,
        portfolio_value: Optional[float] = None
    ) -> Dict:
        """Get watchlist with separate categories for different investment styles
        
        Returns three distinct lists:
        1. UNDERVALUED: Stocks trading below fair value (MoS > 0), sorted by MoS
        2. QUALITY: Best overall scores regardless of valuation
        3. GROWTH: GARP stocks (good growth at reasonable PEG ratios)
        """
        return {
            'undervalued': self.get_top_undervalued(
                n=n_per_category, rescan=rescan, 
                profile_id=profile_id, portfolio_value=portfolio_value
            ),
            'quality': self.get_top_quality(
                n=n_per_category, rescan=False,
                profile_id=profile_id, portfolio_value=portfolio_value
            ),
            'growth': self.get_top_growth(
                n=n_per_category, rescan=False,
                profile_id=profile_id, portfolio_value=portfolio_value
            ),
            'categories_explained': {
                'undervalued': 'Stocks with fair value ABOVE current price (positive margin of safety). True value opportunities.',
                'quality': 'Highest quality companies by combined metrics. May be fairly valued but excellent businesses.',
                'growth': 'Growth At Reasonable Price (GARP) - high growth with PEG < 2.0'
            }
        }
    
    def _format_stock_results(
        self, 
        top_stocks: pd.DataFrame, 
        list_category: str = 'GENERAL',
        profile_id: Optional[str] = None,
        portfolio_value: Optional[float] = None
    ) -> List[Dict]:
        """Format stock data for API response with enhanced justification
        
        Args:
            top_stocks: DataFrame of stocks to format
            list_category: Category label (UNDERVALUED, QUALITY, GROWTH, GENERAL)
            profile_id: Optional risk profile ID for position sizing
            portfolio_value: Optional portfolio value for position sizing (default $50,000)
        """
        results = []
        
        # Default portfolio value for position sizing
        default_portfolio = portfolio_value or 50000.0
        
        for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
            # Cap margin of safety display at 100% to avoid unrealistic expectations
            mos_raw = row['margin_of_safety']
            mos_display = min(mos_raw, 100.0) if mos_raw > 0 else mos_raw
            
            # Determine valuation status for clearer communication
            if mos_raw > 20:
                valuation_status = 'SIGNIFICANTLY_UNDERVALUED'
            elif mos_raw > 5:
                valuation_status = 'MODERATELY_UNDERVALUED'
            elif mos_raw > 0:
                valuation_status = 'SLIGHTLY_UNDERVALUED'
            elif mos_raw > -10:
                valuation_status = 'FAIRLY_VALUED'
            elif mos_raw > -25:
                valuation_status = 'SLIGHTLY_OVERVALUED'
            else:
                valuation_status = 'OVERVALUED'
            
            # Forward-looking metrics summary
            forward_pe = row.get('forward_pe', 0)
            trailing_pe = row.get('pe_ratio', 0)
            forward_metrics = {
                'forward_pe': round(forward_pe, 2) if forward_pe else None,
                'trailing_pe': round(trailing_pe, 2) if trailing_pe else None,
                'pe_trend': 'IMPROVING' if forward_pe and trailing_pe and forward_pe < trailing_pe * 0.95 else (
                    'DECLINING' if forward_pe and trailing_pe and forward_pe > trailing_pe * 1.05 else 'STABLE'
                ),
                'peg_ratio': round(row['peg_ratio'], 2) if row.get('peg_ratio') else None,
                'earnings_growth': round(row['earnings_growth'], 2),
                'revenue_growth': round(row['revenue_growth'], 2),
            }

            result = {
                'rank': rank,
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'sector': row['sector'],
                'industry': row['industry'],
                'list_category': list_category,
                'valuation_status': valuation_status,
                
                # Price & Valuation
                'current_price': row['current_price'],
                'fair_value': row['fair_value'],
                'traditional_fair_value': row['traditional_fair_value'],
                'lstm_fair_value': row['lstm_fair_value'],
                'lstm_predicted_growth': row['lstm_predicted_growth'],
                'fair_value_method': row['fair_value_method'],
                'margin_of_safety': mos_display,  # Capped at 100%
                'margin_of_safety_raw': mos_raw,  # Actual calculated value
                
                # Forward-Looking Metrics (NEW)
                'forward_metrics': forward_metrics,
                
                # Scores
                'valuation_score': row['valuation_score'],
                'garp_score': row.get('garp_score'),  # Transparent Forward P/E + PEG scoring
                'consensus_score': row.get('consensus_score'),
                'consensus_confidence': row.get('consensus_confidence'),
                'effective_score': row.get('effective_score', row['valuation_score']),
                'assessment': row['assessment'],
                'recommendation': row['recommendation'],
                'risk_level': row['risk_level'],
                'justification': row['justification'],
                
                # Valuation Metrics
                'pe_ratio': round(row['pe_ratio'], 2),
                'forward_pe': round(row['forward_pe'], 2) if row.get('forward_pe') else None,
                'sector_avg_pe': row['sector_avg_pe'],
                'pe_vs_sector': round(((row['pe_ratio'] / row['sector_avg_pe']) - 1) * 100, 1) if row['sector_avg_pe'] > 0 and row['pe_ratio'] > 0 else None,
                'pb_ratio': round(row['pb_ratio'], 2),
                'peg_ratio': round(row['peg_ratio'], 2) if row['peg_ratio'] else None,
                'ev_ebitda': round(row['ev_ebitda'], 2) if row['ev_ebitda'] else None,
                
                # Financial Health
                'roe': round(row['roe'], 2),
                'roa': round(row['roa'], 2),
                'debt_equity': round(row['debt_equity'], 2),
                'current_ratio': round(row['current_ratio'], 2),
                
                # Profitability
                'profit_margin': round(row['profit_margin'], 2),
                'gross_margin': round(row['gross_margin'], 2),
                'fcf_yield': round(row['fcf_yield'], 2),
                
                # Growth
                'revenue_growth': round(row['revenue_growth'], 2),
                'earnings_growth': round(row['earnings_growth'], 2),
                
                # Performance
                'market_cap_formatted': row['market_cap_formatted'],
                'pct_from_52w_high': row['pct_from_52w_high'],
                'ytd_return': row['ytd_return'],
                'analyst_target': row['analyst_target'],
                'analyst_upside': row['analyst_upside'],
                
                # Risk
                'beta': row['beta'],
                'volatility': row['volatility'],
                'dividend_yield': round(row['dividend_yield'], 2),
            }
            
            # Add position sizing if profile provided
            if profile_id or portfolio_value:
                position_sizing = self.risk_capacity_service.calculate_position_sizing(
                    stock_data=result,
                    portfolio_value=default_portfolio,
                    profile_id=profile_id
                )
                result['position_sizing'] = {
                    'max_position_pct': position_sizing['max_position_pct'],
                    'max_position_value': position_sizing['max_position_value'],
                    'max_shares': position_sizing['max_shares'],
                    'risk_factors': position_sizing['risk_factors'],
                    'methodology': position_sizing['sizing_methodology'],
                    'confidence': position_sizing['confidence']
                }
            
            # Add educational insights if enabled
            if self.enable_education_mode:
                result['educational_insights'] = self._generate_educational_insights(result)
            
            results.append(result)
        
        return results
    
    def get_comprehensive_watchlist_json(
        self, 
        n: int = 10, 
        rescan: bool = False,
        profile_id: Optional[str] = None,
        portfolio_value: Optional[float] = None
    ) -> Dict:
        """Get comprehensive JSON output for frontend with categorized lists
        
        Returns three separate lists:
        - undervalued: TRUE undervalued stocks (MoS > 0)
        - quality: Top quality scores (any valuation)
        - growth: GARP stocks (high growth, reasonable PEG)
        
        Args:
            n: Number of stocks per category
            rescan: Force fresh market scan
            profile_id: Optional risk profile for position sizing
            portfolio_value: Optional portfolio value for position sizing
        """
        # Get categorized watchlists with position sizing
        categorized = self.get_categorized_watchlist(
            n_per_category=n, 
            rescan=rescan,
            profile_id=profile_id,
            portfolio_value=portfolio_value
        )
        
        # Primary list is undervalued stocks
        top_stocks = categorized.get('undervalued', [])
        
        # Calculate summary stats
        if top_stocks:
            avg_score = np.mean([s['valuation_score'] for s in top_stocks])
            avg_mos = np.mean([s['margin_of_safety'] for s in top_stocks if s['margin_of_safety']])
            sectors = {}
            for s in top_stocks:
                sectors[s['sector']] = sectors.get(s['sector'], 0) + 1
            
            # Count truly undervalued vs overvalued
            undervalued_count = sum(1 for s in top_stocks if s['margin_of_safety'] > 0)
            overvalued_count = len(top_stocks) - undervalued_count
        else:
            avg_score = 0
            avg_mos = 0
            sectors = {}
            undervalued_count = 0
            overvalued_count = 0
        
        # Count dynamic vs default benchmark usage
        dynamic_sectors = sum(1 for s, b in SECTOR_BENCHMARKS.items() if b.get('source') == 'dynamic')
        total_sectors = len(SECTOR_BENCHMARKS)
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'lstm_enabled': self.lstm_model is not None,
                'lstm_model': 'LSTM-DCF Enhanced' if self.lstm_model and self.lstm_model.lstm.input_size == 16 else 'LSTM-DCF Final' if self.lstm_model else None,
                'garp_scoring': True,  # Transparent Forward P/E + PEG scoring
                'consensus_enabled': self.enable_consensus and self.consensus_scorer is not None,
                'consensus_weights': {'lstm_dcf': 0.50, 'garp_score': 0.25, 'risk_score': 0.25},
                'education_mode': self.enable_education_mode,
                'valuation_method': 'Multi-Model Consensus (LSTM 50% + GARP 25% + Risk 25%)' if self.enable_consensus and self.consensus_scorer else 'Hybrid LSTM-DCF + Traditional DCF',
            },
            'scan_params': {
                'universe_size': len(self.tickers),
                'min_market_cap': self.criteria['min_market_cap'],
                'max_pe_ratio': self.criteria['max_pe_ratio'],
                'max_debt_equity': self.criteria['max_debt_equity'],
            },
            'benchmark_info': {
                'use_dynamic_benchmarks': self.use_dynamic_benchmarks,
                'dynamic_sectors': dynamic_sectors,
                'total_sectors': total_sectors,
                'benchmark_coverage': f"{dynamic_sectors}/{total_sectors} sectors using dynamic benchmarks"
            },
            'summary': {
                'results_count': len(top_stocks),
                'undervalued_count': undervalued_count,
                'overvalued_count': overvalued_count,
                'avg_valuation_score': round(avg_score, 1),
                'avg_margin_of_safety': round(avg_mos, 1),
                'sector_distribution': sectors,
                'scoring_methodology': {
                    'forward_pe_weight': '15% (vs sector, bonus if Forward PE < Trailing)',
                    'peg_weight': '15% (Growth At Reasonable Price)',
                    'earnings_growth_weight': '10% (forward momentum)',
                    'pb_weight': '10% (sector-adjusted)',
                    'margin_of_safety_weight': '15% (STRICT: penalizes negative MoS)',
                    'fcf_yield_weight': '12%',
                    'profitability_weight': '13% (ROE + profit margin)',
                    'financial_health_weight': '10%',
                    'consensus_weights': 'LSTM 40% + RF 30% + P/E 20% + Risk 10% (with MoS penalty multiplier)' if self.enable_consensus else 'N/A',
                    'note': 'Forward-looking metrics now prioritized for long-term value investing'
                }
            },
            'lists_explanation': categorized.get('categories_explained', {}),
            'sector_benchmarks': SECTOR_BENCHMARKS,
            # Primary watchlist is undervalued stocks
            'watchlist': top_stocks,
            # Additional categorized lists
            'quality_list': categorized.get('quality', []),
            'growth_list': categorized.get('growth', []),
            # Profile info if provided
            'profile_info': {
                'profile_id': profile_id,
                'portfolio_value': portfolio_value,
                'position_sizing_enabled': bool(profile_id or portfolio_value)
            } if profile_id or portfolio_value else None
        }
    
    def get_filtered_watchlist_for_profile(
        self,
        profile_id: str,
        n: int = 10,
        rescan: bool = False,
        portfolio_value: Optional[float] = None
    ) -> Dict:
        """
        Get watchlist filtered and annotated for a specific risk profile
        
        Returns only stocks that match the profile's:
        - Beta range
        - MoS threshold
        
        Also adds suitability metadata to each stock.
        """
        # Get all stocks first
        all_stocks = self.get_top_undervalued(
            n=50,  # Get more to filter down
            rescan=rescan,
            profile_id=profile_id,
            portfolio_value=portfolio_value
        )
        
        # Filter for profile suitability
        filtered = self.risk_capacity_service.filter_stocks_for_profile(all_stocks, profile_id)
        
        # Return top N filtered
        return {
            'status': 'success',
            'profile_id': profile_id,
            'filtered_count': len(filtered),
            'total_scanned': len(all_stocks),
            'suitable_stocks': filtered[:n]
        }
    
    def get_annotated_watchlist(
        self,
        profile_id: str,
        n: int = 10,
        rescan: bool = False,
        portfolio_value: Optional[float] = None
    ) -> Dict:
        """
        Get full watchlist with suitability annotations for a profile
        
        Returns ALL stocks but marks each with suitability info.
        Frontend can use this to highlight/dim stocks.
        """
        # Get stocks
        stocks = self.get_top_undervalued(
            n=n,
            rescan=rescan,
            profile_id=profile_id,
            portfolio_value=portfolio_value
        )
        
        # Annotate with suitability
        annotated = self.risk_capacity_service.annotate_stocks_with_suitability(stocks, profile_id)
        
        # Get profile info
        profile = self.risk_capacity_service.get_profile(profile_id)
        
        return {
            'status': 'success',
            'profile_id': profile_id,
            'profile_summary': {
                'overall_profile': profile.overall_profile if profile else None,
                'risk_capacity': profile.risk_capacity_score if profile else None,
                'emotional_buffer': profile.emotional_buffer_factor if profile else None,
            } if profile else None,
            'stocks': annotated,
            'suitable_count': sum(1 for s in annotated if s.get('profile_suitability', {}).get('suitable', False))
        }
    
    def get_chart_data(self, ticker: str) -> Dict:
        """Get chart/visualization data for a specific stock"""
        try:
            stock = yf.Ticker(ticker)
            
            # Price history
            hist_1y = stock.history(period="1y", interval="1d")
            hist_5y = stock.history(period="5y", interval="1wk")
            
            if hist_1y.empty:
                return {'error': f'No data for {ticker}'}
            
            # Format for charts
            price_1y = [
                {'date': d.strftime('%Y-%m-%d'), 'price': round(p, 2), 'volume': int(v)}
                for d, p, v in zip(hist_1y.index, hist_1y['Close'], hist_1y['Volume'])
            ]
            
            price_5y = [
                {'date': d.strftime('%Y-%m-%d'), 'price': round(p, 2)}
                for d, p in zip(hist_5y.index, hist_5y['Close'])
            ] if not hist_5y.empty else []
            
            # Technical indicators
            hist_1y['MA50'] = hist_1y['Close'].rolling(50).mean()
            hist_1y['MA200'] = hist_1y['Close'].rolling(200).mean()
            
            ma_data = [
                {
                    'date': d.strftime('%Y-%m-%d'),
                    'price': round(p, 2),
                    'ma50': round(ma50, 2) if pd.notna(ma50) else None,
                    'ma200': round(ma200, 2) if pd.notna(ma200) else None
                }
                for d, p, ma50, ma200 in zip(
                    hist_1y.index[-120:], 
                    hist_1y['Close'][-120:],
                    hist_1y['MA50'][-120:],
                    hist_1y['MA200'][-120:]
                )
            ]
            
            # Get valuation data
            info = stock.info
            
            return {
                'ticker': ticker,
                'company_name': info.get('shortName', ticker),
                'charts': {
                    'price_1y': price_1y,
                    'price_5y': price_5y,
                    'technical': ma_data,
                },
                'metrics': {
                    'current_price': info.get('currentPrice', 0),
                    'high_52w': info.get('fiftyTwoWeekHigh', 0),
                    'low_52w': info.get('fiftyTwoWeekLow', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'market_cap': info.get('marketCap', 0),
                },
                'fundamental_chart': {
                    'pe_history': None,  # Would need quarterly data
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'earnings_growth': info.get('earningsGrowth', 0),
                }
            }
            
        except Exception as e:
            return {'error': str(e)}


# Convenience functions
def screen_top_undervalued(n: int = 10, rescan: bool = False) -> List[Dict]:
    """Quick convenience function to get top undervalued stocks"""
    screener = StockScreener()
    return screener.get_top_undervalued(n=n, rescan=rescan)


# Backward compatibility alias
def screen_top_undervalued_enhanced(n: int = 10, rescan: bool = False) -> List[Dict]:
    """Alias for screen_top_undervalued (backward compatibility)"""
    return screen_top_undervalued(n=n, rescan=rescan)
