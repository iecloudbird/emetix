"""
Enhanced Valuation Agent with LSTM-DCF and RF Ensemble Integration
Extends the original ValuationAgent with advanced ML capabilities
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain import hub
import os
import torch
import pandas as pd

from config.settings import GROQ_API_KEY, MODELS_DIR
from config.logging_config import get_logger
from src.analysis import ValuationAnalyzer, GrowthScreener
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.models.ensemble.consensus_scorer import ConsensusScorer
from src.data.processors.time_series_processor import TimeSeriesProcessor
from src.data.fetchers import YFinanceFetcher
import yfinance as yf

logger = get_logger(__name__)


class EnhancedValuationAgent:
    """
    Enhanced Valuation Agent with LSTM-DCF and RF Ensemble models
    Combines traditional valuation with deep learning forecasts
    """
    
    def __init__(self, api_key: str = GROQ_API_KEY):
        """
        Initialize the Enhanced Valuation Agent
        
        Args:
            api_key: Groq API key for LLM
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        self.logger = logger
        # Use llama-3.3-70b-versatile - llama3-8b-8192 has been decommissioned
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.valuation_analyzer = ValuationAnalyzer()
        self.growth_screener = GrowthScreener()
        self.time_series_processor = TimeSeriesProcessor()
        self.fetcher = YFinanceFetcher()
        self.consensus_scorer = ConsensusScorer()
        
        # Load ML models
        self._load_ml_models()
        
        # Setup tools
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _load_ml_models(self):
        """Load trained LSTM-DCF and RF Ensemble models"""
        # Load LSTM-DCF
        lstm_path = MODELS_DIR / "lstm_dcf_final.pth"
        if lstm_path.exists():
            try:
                self.lstm_model = LSTMDCFModel(input_size=12, hidden_size=128, num_layers=3)
                self.lstm_model.load_model(str(lstm_path))
                self.lstm_model.eval()
                self.logger.info("✓ LSTM-DCF model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load LSTM-DCF model: {e}")
                self.lstm_model = None
        else:
            self.logger.warning(f"LSTM-DCF model not found: {lstm_path}")
            self.lstm_model = None
        
        # Load RF Ensemble
        rf_path = MODELS_DIR / "rf_ensemble.pkl"
        if rf_path.exists():
            try:
                self.rf_model = RFEnsembleModel()
                self.rf_model.load(str(rf_path))
                self.logger.info("✓ RF Ensemble model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load RF Ensemble model: {e}")
                self.rf_model = None
        else:
            self.logger.warning(f"RF Ensemble model not found: {rf_path}")
            self.rf_model = None
    
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
                # Fetch time-series data
                self.logger.info(f"Fetching time-series data for {ticker}...")
                ts_data = self.time_series_processor.fetch_sequential_data(ticker, period='5y')
                
                if ts_data is None or ts_data.empty:
                    return f"Error: Could not fetch time-series data for {ticker}"
                
                # Create sequences
                X, _ = self.time_series_processor.create_sequences(ts_data, target_col='close')
                if len(X) == 0:
                    return f"Error: Insufficient historical data for {ticker} (need at least 60 periods)"
                
                # Get last sequence for prediction
                last_seq = torch.tensor(X[-1:], dtype=torch.float32)
                
                # Forecast FCFF
                fcff_forecasts = self.lstm_model.forecast_fcff(last_seq, periods=10)
                
                # Get stock info
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                shares = info.get('sharesOutstanding', 1e9)
                
                # Calculate DCF valuation
                dcf_result = self.lstm_model.dcf_valuation(fcff_forecasts, shares)
                fair_value = dcf_result['fair_value']
                
                # Calculate valuation gap
                gap = ((fair_value - current_price) / current_price) * 100 if current_price > 0 else 0
                is_undervalued = gap > 10  # More than 10% undervalued
                
                # Format output
                return f"""
LSTM-DCF Hybrid Valuation for {ticker}:

Current Price: ${current_price:.2f}
Fair Value (DCF): ${fair_value:.2f}
Valuation Gap: {gap:+.2f}%
Assessment: {"UNDERVALUED" if is_undervalued else "FAIRLY VALUED" if abs(gap) < 10 else "OVERVALUED"}

DCF Components:
- Enterprise Value: ${dcf_result['enterprise_value']/1e9:.2f}B
- Terminal Value: ${dcf_result['terminal_value']/1e9:.2f}B
- PV of Terminal Value: ${dcf_result['pv_terminal_value']/1e9:.2f}B

10-Year FCFF Forecast (LSTM):
{' → '.join([f"${f/1e9:.2f}B" for f in fcff_forecasts[:5]])}...

Recommendation: {"BUY" if gap > 15 else "HOLD" if gap > -10 else "SELL"}
Confidence: {"High" if abs(gap) > 20 else "Medium"}
"""
            except Exception as e:
                self.logger.error(f"LSTM-DCF valuation error: {e}", exc_info=True)
                return f"Error performing LSTM-DCF valuation: {str(e)}"
        
        def rf_multi_metric_analysis_tool(ticker: str) -> str:
            """Perform Random Forest multi-metric analysis"""
            if not self.rf_model:
                return "Error: RF Ensemble model not loaded. Please train the model first."
            
            try:
                # Fetch stock data
                stock_data = self.fetcher.fetch_stock_data(ticker)
                
                if stock_data is None or (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
                    return f"Error: Could not fetch data for {ticker}"
                
                # Prepare features
                X = self.rf_model.prepare_features(stock_data)
                
                # Get prediction
                result = self.rf_model.predict_score(X)
                
                # Get feature importance
                importance = self.rf_model.get_feature_importance().head(5)
                
                # Format output
                return f"""
Random Forest Multi-Metric Analysis for {ticker}:

Ensemble Score: {result['ensemble_score']:.4f}
Undervalued Probability: {result['undervalued_probability']:.2%}
Classification: {"UNDERVALUED" if result['is_undervalued'] else "NOT UNDERVALUED"}

Regression Prediction: {result['regression_prediction']:.4f}
Classification Confidence: {result['classification_confidence']:.2%}

Top 5 Most Important Features:
{chr(10).join([f"  {row['feature']}: {row['importance']:.4f}" for _, row in importance.iterrows()])}

Recommendation: {"BUY" if result['is_undervalued'] and result['undervalued_probability'] > 0.7 else "HOLD" if result['ensemble_score'] > 0.5 else "AVOID"}
"""
            except Exception as e:
                self.logger.error(f"RF analysis error: {e}", exc_info=True)
                return f"Error performing RF analysis: {str(e)}"
        
        def consensus_valuation_tool(ticker: str) -> str:
            """Generate consensus valuation from all models"""
            try:
                # Get traditional valuation
                trad_result = self.valuation_analyzer.analyze_stock(ticker)
                trad_score = trad_result.get('valuation_score', 50) / 100 if 'error' not in trad_result else 0.5
                
                # Get RF score
                rf_score = 0.5
                if self.rf_model:
                    try:
                        stock_data = self.fetcher.fetch_stock_data(ticker)
                        if stock_data is not None:
                            X = self.rf_model.prepare_features(stock_data)
                            rf_result = self.rf_model.predict_score(X)
                            rf_score = rf_result['ensemble_score']
                    except:
                        pass
                
                # Get LSTM-DCF score (normalized valuation gap)
                lstm_score = 0.5
                if self.lstm_model:
                    try:
                        ts_data = self.time_series_processor.fetch_sequential_data(ticker, period='5y')
                        if ts_data is not None and not ts_data.empty:
                            X, _ = self.time_series_processor.create_sequences(ts_data, target_col='close')
                            if len(X) > 0:
                                last_seq = torch.tensor(X[-1:], dtype=torch.float32)
                                fcff_forecasts = self.lstm_model.forecast_fcff(last_seq, periods=10)
                                
                                stock = yf.Ticker(ticker)
                                info = stock.info
                                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                                shares = info.get('sharesOutstanding', 1e9)
                                
                                dcf_result = self.lstm_model.dcf_valuation(fcff_forecasts, shares)
                                fair_value = dcf_result['fair_value']
                                
                                if current_price > 0:
                                    gap = ((fair_value - current_price) / current_price) * 100
                                    # Normalize gap to 0-1 scale (-20% to +20% maps to 0 to 1)
                                    lstm_score = max(0, min(1, (gap + 20) / 40))
                    except:
                        pass
                
                # Calculate consensus
                model_scores = {
                    'lstm_dcf': lstm_score,
                    'rf_ensemble': rf_score,
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
Models Agreement: {consensus['num_models']}/4 models active

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
                name="RF_MultiMetric_Analysis",
                func=rf_multi_metric_analysis_tool,
                description="Random Forest ensemble analysis combining multiple valuation metrics. Use for machine learning-based undervalued stock detection with feature importance."
            ),
            Tool(
                name="ConsensusValuation",
                func=consensus_valuation_tool,
                description="Multi-model consensus combining LSTM-DCF, Random Forest, and traditional models. Use for the most robust valuation with confidence scoring."
            ),
            Tool(
                name="StockComparison",
                func=stock_comparison_tool,
                description="Compare multiple stocks side-by-side. Input: comma-separated tickers (e.g., 'AAPL,MSFT,GOOGL')."
            ),
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor"""
        # Use react prompt from hub
        prompt = hub.pull("hwchase17/react")
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def analyze(self, query: str) -> str:
        """
        Analyze a stock or answer valuation questions
        
        Args:
            query: Natural language query about stock valuation
            
        Returns:
            Analysis result as string
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            return result['output']
        except Exception as e:
            self.logger.error(f"Agent analysis error: {e}", exc_info=True)
            return f"Error during analysis: {str(e)}"
