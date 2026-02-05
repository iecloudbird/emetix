"""
Enhanced Valuation Agent with LSTM-DCF Integration
Extends the original ValuationAgent with advanced ML capabilities
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import torch
import pandas as pd
import numpy as np

from config.settings import MODELS_DIR
from config.logging_config import get_logger
from src.analysis import ValuationAnalyzer, GrowthScreener
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.ensemble.consensus_scorer import ConsensusScorer
from src.data.processors.time_series_processor import TimeSeriesProcessor
from src.data.processors.lstm_v2_processor import LSTMV2Processor, create_v2_processor
from src.data.fetchers import YFinanceFetcher
from src.utils.llm_provider import get_llm
import yfinance as yf

logger = get_logger(__name__)


class EnhancedValuationAgent:
    """
    Enhanced Valuation Agent with LSTM-DCF model
    Combines traditional valuation with deep learning forecasts
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Enhanced Valuation Agent
        """
        self.logger = logger
        # Use default model tier for valuation analysis
        self.llm = get_llm(model_tier="default", temperature=0)
        self.valuation_analyzer = ValuationAnalyzer()
        self.growth_screener = GrowthScreener()
        self.fetcher = YFinanceFetcher()
        self.consensus_scorer = ConsensusScorer()
        
        # Load ML models first (before creating processors)
        self._load_ml_models()
        
        # Initialize appropriate processor based on model version
        if getattr(self, 'lstm_model_version', 'v1') == 'v2':
            # v2 uses quarterly fundamentals, not daily prices
            self.v2_processor = create_v2_processor(self.lstm_metadata)
            self.time_series_processor = None  # Not used for v2
            self.logger.info(f"✓ Using V2 processor (quarterly fundamentals)")
        else:
            # v1 uses daily price data
            model_seq_length = getattr(self, 'lstm_sequence_length', 60)
            self.time_series_processor = TimeSeriesProcessor(sequence_length=model_seq_length)
            self.v2_processor = None
            self.logger.info(f"✓ Using V1 processor (daily prices, seq_len={model_seq_length})")
        
        # Setup tools
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _load_ml_models(self):
        """Load trained LSTM-DCF model using from_checkpoint for v1/v2 support"""
        lstm_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
        if lstm_path.exists():
            try:
                # Use from_checkpoint for automatic version detection
                self.lstm_model, self.lstm_metadata = LSTMDCFModel.from_checkpoint(str(lstm_path))
                self.lstm_model_version = self.lstm_metadata.get('model_version', 'v1')
                self.lstm_sequence_length = self.lstm_metadata.get('sequence_length', 8)
                self.lstm_feature_scaler = self.lstm_metadata.get('feature_scaler') or self.lstm_metadata.get('scaler')
                self.lstm_target_scaler = self.lstm_metadata.get('target_scaler')
                self.logger.info(f"✓ LSTM-DCF {self.lstm_model_version} model loaded (seq_len={self.lstm_sequence_length})")
            except Exception as e:
                self.logger.warning(f"Could not load LSTM-DCF model: {e}")
                self.lstm_model = None
                self.lstm_metadata = None
                self.lstm_model_version = None
        else:
            self.logger.warning(f"LSTM-DCF model not found: {lstm_path}")
            self.lstm_model = None
            self.lstm_metadata = None
            self.lstm_model_version = None
    
    def _setup_tools(self) -> list:
        """Setup tools including traditional and ML-powered valuation"""
        
        def comprehensive_valuation_tool(ticker: str) -> str:
            """Perform comprehensive valuation analysis"""
            try:
                result = self.valuation_analyzer.analyze_stock(ticker)
                if 'error' in result:
                    return f"Error: {result['error']}"
                
                return f"""
Comprehensive Valuation for {ticker}:

Current Price: ${result['current_price']:.2f}
Fair Value: ${result['fair_value_estimate']:.2f}
Valuation Score: {result['valuation_score']}/100
Assessment: {result['assessment']}
Risk Level: {result['risk_level']}
Recommendation: {result['recommendation']}

Key Metrics:
- P/E: {result['key_metrics']['pe_ratio']:.2f}
- P/B: {result['key_metrics']['pb_ratio']:.2f}
- PEG: {result['key_metrics']['peg_ratio']:.2f}
- Debt/Equity: {result['key_metrics']['debt_equity']:.2f}
- ROE: {result['key_metrics']['roe']:.1f}%
- FCF Yield: {result['key_metrics']['fcf_yield']:.1f}%

Component Scores: {result['component_scores']}
"""
            except Exception as e:
                return f"Error: {str(e)}"
        
        def lstm_dcf_valuation_tool(ticker: str) -> str:
            """Perform LSTM-DCF hybrid valuation with time-series forecasting"""
            if not self.lstm_model:
                return "Error: LSTM-DCF model not loaded. Please train the model first."
            
            try:
                # Get stock info
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                
                # Handle v2 model (quarterly fundamentals)
                if self.lstm_model_version == 'v2' and self.v2_processor is not None:
                    self.logger.info(f"Using LSTM-DCF v2 for {ticker}...")
                    
                    # Prepare v2 inference sequence
                    result = self.v2_processor.prepare_inference_sequence(
                        ticker, 
                        feature_scaler=self.lstm_feature_scaler
                    )
                    
                    if result is None:
                        return f"Error: Could not fetch quarterly data for {ticker}. Need {self.lstm_sequence_length} quarters of financial data."
                    
                    input_tensor, metadata = result
                    
                    # Run model prediction
                    self.lstm_model.eval()
                    with torch.no_grad():
                        prediction = self.lstm_model(input_tensor)
                    
                    # Interpret prediction
                    interpretation = self.v2_processor.interpret_prediction(
                        prediction,
                        target_scaler=self.lstm_target_scaler,
                        current_price=current_price
                    )
                    
                    revenue_growth = interpretation['revenue_growth_forecast']
                    fcf_growth = interpretation['fcf_growth_forecast']
                    upside = interpretation['upside_percent']
                    fair_value = interpretation['implied_fair_value']
                    
                    # Determine recommendation
                    if upside > 15:
                        assessment = "UNDERVALUED"
                        recommendation = "BUY"
                    elif upside > -10:
                        assessment = "FAIRLY VALUED"
                        recommendation = "HOLD"
                    else:
                        assessment = "OVERVALUED"
                        recommendation = "SELL"
                    
                    return f"""
LSTM-DCF v2 Valuation for {ticker}:

Current Price: ${current_price:.2f}
Implied Fair Value: ${fair_value:.2f}
Valuation Gap: {upside:+.2f}%
Assessment: {assessment}

Growth Forecasts (Next Year):
- Revenue Growth: {revenue_growth:+.1f}%
- FCF Growth: {fcf_growth:+.1f}%

Model Details:
- Quarters Analyzed: {metadata['quarters_used']}
- Latest Quarter: {metadata['latest_quarter']}
- Model Version: v2 (fundamentals-based)

Recommendation: {recommendation}
Confidence: {"High" if abs(upside) > 20 else "Medium"}
Note: Predictions based on quarterly financial trends
"""
                
                # Handle v1 model (daily prices) - fallback
                else:
                    self.logger.info(f"Using LSTM-DCF v1 for {ticker}...")
                    
                    if self.time_series_processor is None:
                        return "Error: V1 processor not initialized for v1 model"
                    
                    ts_data = self.time_series_processor.fetch_sequential_data(ticker, period='5y')
                    
                    if ts_data is None or ts_data.empty:
                        return f"Error: Could not fetch time-series data for {ticker}"
                    
                    X, _ = self.time_series_processor.create_sequences(ts_data, target_col='close')
                    if len(X) == 0:
                        return f"Error: Insufficient historical data for {ticker}"
                    
                    last_seq = torch.tensor(X[-1:], dtype=torch.float32)
                    shares = info.get('sharesOutstanding', 1e9)
                    
                    fcff_forecasts = self.lstm_model.forecast_fcff(
                        last_seq, 
                        periods=10, 
                        scaler=self.time_series_processor.scaler,
                        fcff_feature_idx=-1,
                        use_per_share=True
                    )
                    
                    dcf_result = self.lstm_model.dcf_valuation(fcff_forecasts, 1.0, current_price)
                    fair_value = dcf_result.get('calibrated_fair_value', dcf_result['fair_value'])
                    gap = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
                    is_undervalued = gap > 10
                    
                    return f"""
LSTM-DCF v1 Valuation for {ticker}:

Current Price: ${current_price:.2f}
Fair Value (DCF): ${fair_value:.2f}
Valuation Gap: {gap:+.2f}%
Assessment: {"UNDERVALUED" if is_undervalued else "FAIRLY VALUED" if abs(gap) < 10 else "OVERVALUED"}

DCF Components:
- Present Value of FCFF: ${dcf_result['pv_fcff']:.2f}
- Terminal Value (PV): ${dcf_result['pv_terminal_value']:.2f}

10-Year FCFF Forecast:
{' → '.join([f"${f:.2f}" for f in fcff_forecasts[:5]])}...

Recommendation: {"BUY" if gap > 15 else "HOLD" if gap > -10 else "SELL"}
Note: Model trained on proxy FCFF (v1)
"""
            except Exception as e:
                self.logger.error(f"LSTM-DCF valuation error: {e}", exc_info=True)
                return f"Error performing LSTM-DCF valuation: {str(e)}"
        
        def consensus_valuation_tool(ticker: str) -> str:
            """Generate consensus valuation from all models"""
            try:
                # Get traditional valuation
                trad_result = self.valuation_analyzer.analyze_stock(ticker)
                trad_score = trad_result.get('valuation_score', 50) / 100 if 'error' not in trad_result else 0.5
                
                # Get LSTM-DCF score
                lstm_score = 0.5
                if self.lstm_model:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                        
                        if self.lstm_model_version == 'v2' and self.v2_processor is not None:
                            # v2: Use quarterly fundamentals
                            result = self.v2_processor.prepare_inference_sequence(
                                ticker, feature_scaler=self.lstm_feature_scaler
                            )
                            if result is not None:
                                input_tensor, _ = result
                                self.lstm_model.eval()
                                with torch.no_grad():
                                    prediction = self.lstm_model(input_tensor)
                                
                                interpretation = self.v2_processor.interpret_prediction(
                                    prediction,
                                    target_scaler=self.lstm_target_scaler,
                                    current_price=current_price
                                )
                                
                                # Convert growth forecasts to score (0-1)
                                avg_growth = (interpretation['revenue_growth_forecast'] + 
                                            interpretation['fcf_growth_forecast']) / 2
                                # Strong positive growth = high score
                                if avg_growth > 30:
                                    lstm_score = 0.85
                                elif avg_growth > 15:
                                    lstm_score = 0.7
                                elif avg_growth > 0:
                                    lstm_score = 0.5 + (avg_growth / 30) * 0.2
                                elif avg_growth > -15:
                                    lstm_score = 0.35 + (avg_growth + 15) / 30 * 0.15
                                else:
                                    lstm_score = 0.3
                        else:
                            # v1: Use daily price data
                            if self.time_series_processor is not None:
                                ts_data = self.time_series_processor.fetch_sequential_data(ticker, period='5y')
                                if ts_data is not None and not ts_data.empty:
                                    X, _ = self.time_series_processor.create_sequences(ts_data, target_col='close')
                                    if len(X) > 0:
                                        last_seq = torch.tensor(X[-1:], dtype=torch.float32)
                                        fcff_forecasts = self.lstm_model.forecast_fcff(
                                            last_seq, periods=10,
                                            scaler=self.time_series_processor.scaler,
                                            fcff_feature_idx=-1, use_per_share=True
                                        )
                                        fcff_growth = (fcff_forecasts[-1] - fcff_forecasts[0]) / fcff_forecasts[0] if fcff_forecasts[0] > 0 else 0
                                        if fcff_growth > 0.5:
                                            lstm_score = 1.0
                                        elif fcff_growth > 0.2:
                                            lstm_score = 0.65 + (fcff_growth - 0.2) * 1.17
                                        elif fcff_growth > 0:
                                            lstm_score = 0.5 + (fcff_growth / 0.2) * 0.15
                                        elif fcff_growth > -0.1:
                                            lstm_score = 0.5 + (fcff_growth / 0.1) * 0.2
                                        else:
                                            lstm_score = 0.3
                    except Exception as e:
                        self.logger.warning(f"LSTM scoring failed: {e}")
                
                # Calculate consensus (LSTM-DCF + Traditional)
                # Weights: LSTM-DCF 50%, Traditional 40%, Risk placeholder 10%
                model_scores = {
                    'lstm_dcf': lstm_score,
                    'linear_valuation': trad_score,
                    'risk_classifier': 0.6  # Placeholder - would integrate actual risk model
                }
                
                consensus = self.consensus_scorer.calculate_consensus(model_scores)
                breakdown = self.consensus_scorer.get_model_breakdown(model_scores)
                
                # Format output
                result = f"""
Multi-Model Consensus Valuation for {ticker}:

Consensus Score: {consensus['consensus_score']:.4f}
Confidence Level: {consensus['confidence']:.2%}
Models Agreement: {consensus['num_models']}/3 models active

Final Assessment: {"UNDERVALUED" if consensus['is_undervalued'] else "NOT UNDERVALUED"}

Model Breakdown:
"""
                for model, details in breakdown.items():
                    result += f"  {model}: {details['raw_score']:.4f} × {details['weight']:.2f} = {details['weighted_contribution']:.4f}\n"
                
                result += f"\nRecommendation: "
                if consensus['is_undervalued'] and consensus['confidence'] > 0.75:
                    result += "STRONG BUY"
                elif consensus['is_undervalued']:
                    result += "BUY"
                elif consensus['consensus_score'] > 0.45:
                    result += "HOLD"
                else:
                    result += "AVOID"
                
                return result
                
            except Exception as e:
                self.logger.error(f"Consensus valuation error: {e}", exc_info=True)
                return f"Error generating consensus: {str(e)}"
        
        def stock_comparison_tool(tickers_str: str) -> str:
            """Compare stocks across multiple dimensions"""
            try:
                tickers = [t.strip().upper() for t in tickers_str.split(',')]
                comparison = self.valuation_analyzer.compare_stocks(tickers)
                
                if comparison.empty:
                    return "No valid data for comparison"
                
                result = "Stock Comparison:\n\n"
                for _, row in comparison.iterrows():
                    result += f"{row['ticker']}: Score {row['valuation_score']}/100, "
                    result += f"${row['current_price']:.2f}, "
                    result += f"{row['assessment']}, {row['recommendation']}\n"
                
                best = comparison.iloc[0]
                result += f"\nTop Pick: {best['ticker']} (Score: {best['valuation_score']}/100)"
                
                return result
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Return all tools
        tools = [
            Tool(
                name="ComprehensiveValuation",
                func=comprehensive_valuation_tool,
                description="Traditional comprehensive valuation with P/E, P/B, PEG ratios, financial health. Use for standard valuation analysis."
            ),
            Tool(
                name="LSTM_DCF_Valuation",
                func=lstm_dcf_valuation_tool,
                description="Advanced LSTM-DCF hybrid valuation using deep learning to forecast future cash flows. Use for sophisticated fair value estimation with time-series analysis."
            ),
            Tool(
                name="ConsensusValuation",
                func=consensus_valuation_tool,
                description="Multi-model consensus combining LSTM-DCF and traditional models. Use for robust valuation with confidence scoring."
            ),
            Tool(
                name="StockComparison",
                func=stock_comparison_tool,
                description="Compare multiple stocks side-by-side. Input: comma-separated tickers (e.g., 'AAPL,MSFT,GOOGL')."
            ),
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the agent using langgraph"""
        # System prompt for the agent
        system_prompt = """You are an Enhanced Valuation Agent with access to advanced ML models for stock valuation.

You have access to tools that can:
1. Run LSTM-DCF deep learning model for fair value estimation
2. Calculate consensus from multiple models
3. Perform traditional valuation analysis
4. Compare multiple stocks

When analyzing a stock:
1. Use ComprehensiveValuation for traditional metrics
2. Use LSTM_DCF_Valuation for ML-powered fair value
3. Use ConsensusValuation for combined assessment

Provide ML-powered valuation insights with confidence levels."""
        
        # Create agent using langgraph prebuilt
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt
        )
        
        return agent
    
    def analyze(self, query: str) -> str:
        """
        Analyze a stock or answer valuation questions
        
        Args:
            query: Natural language query about stock valuation
            
        Returns:
            Analysis result as string
        """
        try:
            # New langgraph API uses messages format
            result = self.agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            return result["messages"][-1].content if result.get("messages") else "No response"
        except Exception as e:
            self.logger.error(f"Agent analysis error: {e}", exc_info=True)
            return f"Error during analysis: {str(e)}"
