"""
Watchlist Manager Agent - Intelligent watchlist with dynamic scoring
Part of Multi-Agent Stock Analysis System
Uses LSTM-DCF for ML-powered predictions + transparent GARP scoring
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config.settings import MODELS_DIR
from config.logging_config import get_logger
from src.utils.llm_provider import get_llm

# Import ML models
from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.data.processors.time_series_processor import TimeSeriesProcessor

logger = get_logger(__name__)


class WatchlistManagerAgent:
    """
    Specialized agent for managing intelligent watchlist
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Watchlist Manager Agent with ML models
        """
        self.logger = logger
        # Use default model tier for watchlist management
        self.llm = get_llm(model_tier="default", temperature=0)
        
        # Load ML models
        self.ml_models_available = False
        try:
            # Load LSTM-DCF model using from_checkpoint for v1/v2 support
            lstm_path = MODELS_DIR / "lstm_dcf_enhanced.pth"
            if lstm_path.exists():
                self.lstm_model, self.lstm_metadata = LSTMDCFModel.from_checkpoint(str(lstm_path))
                self.lstm_model_version = self.lstm_metadata.get('model_version', 'v1')
                self.lstm_sequence_length = self.lstm_metadata.get('sequence_length', 8)
                self.lstm_feature_scaler = self.lstm_metadata.get('feature_scaler') or self.lstm_metadata.get('scaler')
                self.lstm_target_scaler = self.lstm_metadata.get('target_scaler')
            else:
                raise FileNotFoundError(f"Model not found: {lstm_path}")
            
            # Time series processor for LSTM
            self.ts_processor = TimeSeriesProcessor()
            
            self.ml_models_available = True
            self.logger.info(f"✅ LSTM-DCF {self.lstm_model_version} model loaded (seq_len={self.lstm_sequence_length})")
        except Exception as e:
            self.logger.warning(f"⚠️ ML models not available: {str(e)}")
            self.logger.info("Falling back to traditional scoring only")
        
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
        
        # Updated weights for ML-enhanced scoring (LSTM-DCF + GARP)
        if self.ml_models_available:
            self.default_weights = {
                # Traditional factors (50% total)
                'growth': 0.15,        # 15% - Revenue/earnings growth
                'sentiment': 0.12,     # 12% - Market sentiment
                'valuation': 0.13,     # 13% - P/E, PEG ratios (GARP)
                'risk': 0.10,          # 10% - Beta, volatility, debt
                # ML predictions (50% total)
                'lstm_dcf': 0.50       # 50% - LSTM-DCF fair value
            }
            self.logger.info("📊 Using LSTM-DCF + GARP scoring weights")
        else:
            # Fallback to traditional weights
            self.default_weights = {
                'growth': 0.30,        # 30% - Revenue/earnings growth
                'sentiment': 0.25,     # 25% - Market sentiment
                'valuation': 0.20,     # 20% - P/E, PEG ratios
                'risk': 0.15,          # 15% - Beta, volatility, debt
                'macro': 0.10          # 10% - Macroeconomic factors
            }
        
        # Contrarian multiplier settings
        self.contrarian_config = {
            'sentiment_threshold': 0.4,      # Sentiment < 0.4 = suppressed
            'valuation_threshold': 0.7,      # Valuation score > 0.7 = undervalued
            'bonus_multiplier': 0.2,         # 20% bonus for contrarian plays
            'max_bonus': 0.15                # Cap bonus at 15 points
        }
    
    def _setup_tools(self) -> List[Tool]:
        """Setup watchlist management tools"""
        
        def calculate_composite_score_tool(scores_json: str) -> str:
            """
            Calculate weighted composite score
            
            Args:
                scores_json: JSON string with format:
                    '{"growth": 0.8, "sentiment": 0.3, "valuation": 0.9, "risk": 0.7, "macro": 0.6}'
            """
            try:
                import json
                scores = json.loads(scores_json)
                
                # Apply default weights
                composite_score = (
                    scores.get('growth', 0.5) * 0.30 +
                    scores.get('sentiment', 0.5) * 0.25 +
                    scores.get('valuation', 0.5) * 0.20 +
                    scores.get('risk', 0.5) * 0.15 +
                    scores.get('macro', 0.5) * 0.10
                )
                
                # Apply contrarian bonus if applicable
                contrarian_bonus = 0
                if scores.get('sentiment', 0.5) < 0.4 and scores.get('valuation', 0.5) > 0.7:
                    # Suppressed sentiment + strong valuation = contrarian opportunity
                    contrarian_bonus = max(
                        0,
                        (0.7 - scores.get('sentiment', 0.5)) * scores.get('valuation', 0.5) * 0.2
                    )
                    contrarian_bonus = min(contrarian_bonus, 0.15)  # Cap at 15%
                
                final_score = composite_score + contrarian_bonus
                final_score = min(final_score, 1.0)  # Cap at 1.0
                
                # Generate signal
                if final_score >= 0.75:
                    signal = "STRONG BUY"
                elif final_score >= 0.65:
                    signal = "BUY"
                elif final_score >= 0.45:
                    signal = "HOLD"
                elif final_score >= 0.35:
                    signal = "WEAK HOLD"
                else:
                    signal = "SELL"
                
                data = {
                    'composite_score': round(final_score * 100, 1),  # 0-100 scale
                    'contrarian_bonus': round(contrarian_bonus * 100, 1),
                    'signal': signal,
                    'component_scores': scores,
                    'interpretation': self._interpret_score(final_score, contrarian_bonus > 0)
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating composite score: {str(e)}"
        
        def calculate_ml_enhanced_score_tool(ticker: str, scores_json: str) -> str:
            """
            Calculate ML-enhanced composite score with LSTM-DCF and RF predictions
            
            Args:
                ticker: Stock ticker symbol
                scores_json: JSON string with traditional scores:
                    '{"growth": 0.8, "sentiment": 0.3, "valuation": 0.9, "risk": 0.7}'
            """
            try:
                import json
                from src.data.fetchers import YFinanceFetcher
                
                scores = json.loads(scores_json)
                
                # Traditional component (55%)
                traditional_score = (
                    scores.get('growth', 0.5) * 0.18 +
                    scores.get('sentiment', 0.5) * 0.15 +
                    scores.get('valuation', 0.5) * 0.12 +
                    scores.get('risk', 0.5) * 0.10
                )
                
                ml_scores = {}
                
                # ML component (45%) - only if models available
                if self.ml_models_available:
                    try:
                        # LSTM-DCF prediction (25%)
                        ts_data = self.ts_processor.fetch_sequential_data(ticker, period='5y')
                        if ts_data is not None and len(ts_data) > 60:
                            X, _ = self.ts_processor.create_sequences(ts_data)
                            last_seq = torch.tensor(X[-1:], dtype=torch.float32)
                            
                            # Get current price and shares
                            fetcher = YFinanceFetcher()
                            stock_data = fetcher.fetch_stock_data(ticker)
                            
                            if stock_data is not None and not (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
                                # Extract scalar values from DataFrame
                                def safe_extract(df, col, default=0):
                                    """Extract scalar value from DataFrame column"""
                                    if isinstance(df, dict):
                                        return df.get(col, default)
                                    elif isinstance(df, pd.DataFrame):
                                        val = df.get(col)
                                        if val is not None:
                                            if isinstance(val, pd.Series) and not val.empty:
                                                return val.iloc[0]
                                            elif not isinstance(val, pd.Series):
                                                return val
                                    return default
                                
                                current_price = safe_extract(stock_data, 'current_price', 0)
                                shares = safe_extract(stock_data, 'shares_outstanding', 0)
                                
                                if current_price > 0 and shares > 0:
                                    result = self.lstm_model.predict_stock_value(last_seq, current_price, shares)
                                    
                                    # Convert valuation gap to score (0-1)
                                    gap = result['fair_value_gap']
                                    # Gap > 0 means undervalued (good), < 0 means overvalued (bad)
                                    # Normalize: gap of +20% = 1.0, -20% = 0.0
                                    lstm_score = 0.5 + (gap / 40)  # Scale gap to 0-1 range
                                    lstm_score = max(0, min(1, lstm_score))  # Clamp to [0, 1]
                                    
                                    ml_scores['lstm_dcf'] = lstm_score
                                    ml_scores['lstm_fair_value'] = result['fair_value']
                                    ml_scores['lstm_gap'] = gap
                                    self.logger.info(f"LSTM-DCF: {ticker} fair value ${result['fair_value']:.2f}, gap {gap:+.1f}%")
                    
                    except Exception as e:
                        self.logger.warning(f"ML prediction failed for {ticker}: {str(e)}")
                
                # Calculate final score
                if ml_scores:
                    # LSTM-DCF weighted at 50% of total score
                    ml_component = ml_scores.get('lstm_dcf', 0.5) * 0.50
                    composite_score = traditional_score + ml_component
                    scoring_method = "LSTM-DCF Enhanced"
                else:
                    # Fallback to traditional + macro if ML unavailable
                    composite_score = traditional_score + scores.get('macro', 0.5) * 0.10
                    scoring_method = "Traditional"
                
                # Apply contrarian bonus
                contrarian_bonus = 0
                if scores.get('sentiment', 0.5) < 0.4 and scores.get('valuation', 0.5) > 0.7:
                    contrarian_bonus = max(
                        0,
                        (0.7 - scores.get('sentiment', 0.5)) * scores.get('valuation', 0.5) * 0.2
                    )
                    contrarian_bonus = min(contrarian_bonus, 0.15)
                
                final_score = composite_score + contrarian_bonus
                final_score = min(final_score, 1.0)
                
                # Generate signal
                if final_score >= 0.75:
                    signal = "STRONG BUY"
                elif final_score >= 0.65:
                    signal = "BUY"
                elif final_score >= 0.45:
                    signal = "HOLD"
                elif final_score >= 0.35:
                    signal = "WEAK HOLD"
                else:
                    signal = "SELL"
                
                # Add ML confirmation flag
                ml_confirmed = False
                if ml_scores:
                    lstm_dcf_score = ml_scores.get('lstm_dcf', 0.5)
                    if lstm_dcf_score > 0.6 and final_score >= 0.65:
                        ml_confirmed = True
                        signal = signal + " (ML-Confirmed)"
                
                data = {
                    'ticker': ticker,
                    'composite_score': round(final_score * 100, 1),
                    'traditional_score': round(traditional_score * 100, 1),
                    'ml_score': round((ml_component if ml_scores else 0) * 100, 1),
                    'contrarian_bonus': round(contrarian_bonus * 100, 1),
                    'signal': signal,
                    'scoring_method': scoring_method,
                    'ml_confirmed': ml_confirmed,
                    'component_scores': scores,
                    'ml_predictions': ml_scores,
                    'interpretation': self._interpret_score(final_score, contrarian_bonus > 0)
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error calculating ML-enhanced score: {str(e)}"
        
        def rank_watchlist_tool(watchlist_json: str) -> str:
            """
            Rank stocks in watchlist by composite score
            
            Args:
                watchlist_json: JSON string with format:
                    '[{"ticker": "AAPL", "composite_score": 78.5}, {"ticker": "MSFT", "composite_score": 85.2}]'
            """
            try:
                import json
                watchlist = json.loads(watchlist_json)
                
                # Sort by composite score (descending)
                ranked = sorted(watchlist, key=lambda x: x.get('composite_score', 0), reverse=True)
                
                # Add rankings
                for i, stock in enumerate(ranked, start=1):
                    stock['rank'] = i
                    stock['percentile'] = round((1 - (i-1)/len(ranked)) * 100, 1)
                
                data = {
                    'ranked_watchlist': ranked,
                    'total_stocks': len(ranked),
                    'top_pick': ranked[0] if ranked else None,
                    'average_score': round(np.mean([s.get('composite_score', 0) for s in ranked]), 1) if ranked else 0
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error ranking watchlist: {str(e)}"
        
        def detect_contrarian_opportunities_tool(watchlist_json: str) -> str:
            """
            Detect contrarian opportunities in watchlist
            
            Args:
                watchlist_json: JSON with ticker, sentiment_score, valuation_score
            """
            try:
                import json
                watchlist = json.loads(watchlist_json)
                
                contrarian_opportunities = []
                
                for stock in watchlist:
                    ticker = stock.get('ticker', 'UNKNOWN')
                    sentiment = stock.get('sentiment_score', 0.5)
                    valuation = stock.get('valuation_score', 0.5)
                    
                    # Contrarian logic: negative sentiment + strong fundamentals
                    if sentiment < 0.4 and valuation > 0.7:
                        opportunity_strength = (0.7 - sentiment) * valuation
                        
                        contrarian_opportunities.append({
                            'ticker': ticker,
                            'opportunity_strength': round(opportunity_strength, 2),
                            'sentiment_score': sentiment,
                            'valuation_score': valuation,
                            'reason': f'Suppressed sentiment ({sentiment:.2f}) with strong fundamentals ({valuation:.2f})',
                            'risk_level': 'MEDIUM' if valuation > 0.8 else 'MODERATE',
                            'time_horizon': 'LONG_TERM'  # Mean reversion takes time
                        })
                
                # Sort by opportunity strength
                contrarian_opportunities.sort(key=lambda x: x['opportunity_strength'], reverse=True)
                
                data = {
                    'contrarian_opportunities': contrarian_opportunities,
                    'total_opportunities': len(contrarian_opportunities),
                    'top_contrarian_pick': contrarian_opportunities[0] if contrarian_opportunities else None
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error detecting contrarian opportunities: {str(e)}"
        
        def generate_alerts_tool(watchlist_json: str) -> str:
            """
            Generate alerts for significant changes
            
            Args:
                watchlist_json: JSON with ticker, score_change_pct, price_change_pct, etc.
            """
            try:
                import json
                watchlist = json.loads(watchlist_json)
                
                alerts = []
                
                for stock in watchlist:
                    ticker = stock.get('ticker', 'UNKNOWN')
                    score_change = stock.get('score_change_pct', 0)
                    price_change = stock.get('price_change_pct', 0)
                    composite_score = stock.get('composite_score', 50)
                    
                    # Alert conditions
                    if score_change > 15:
                        alerts.append({
                            'ticker': ticker,
                            'type': 'SCORE_SURGE',
                            'message': f'{ticker} score increased {score_change:.1f}% - potential buy signal',
                            'priority': 'HIGH'
                        })
                    elif score_change < -15:
                        alerts.append({
                            'ticker': ticker,
                            'type': 'SCORE_DROP',
                            'message': f'{ticker} score decreased {score_change:.1f}% - review position',
                            'priority': 'MEDIUM'
                        })
                    
                    if composite_score >= 75 and price_change < -5:
                        alerts.append({
                            'ticker': ticker,
                            'type': 'BUY_DIP',
                            'message': f'{ticker} high score ({composite_score}) with price dip ({price_change:.1f}%) - buying opportunity',
                            'priority': 'HIGH'
                        })
                    
                    if composite_score <= 40 and price_change > 10:
                        alerts.append({
                            'ticker': ticker,
                            'type': 'OVERHEATED',
                            'message': f'{ticker} low score ({composite_score}) with price surge ({price_change:.1f}%) - consider selling',
                            'priority': 'HIGH'
                        })
                
                data = {
                    'alerts': alerts,
                    'total_alerts': len(alerts),
                    'high_priority_count': sum(1 for a in alerts if a['priority'] == 'HIGH')
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error generating alerts: {str(e)}"
        
        tools = [
            Tool(
                name="CalculateCompositeScore",
                func=calculate_composite_score_tool,
                description="Calculate weighted composite score from component scores (growth, sentiment, valuation, risk, macro). Applies contrarian bonus for suppressed-yet-undervalued stocks. Returns 0-100 score with buy/hold/sell signal. [Traditional scoring only]"
            ),
            Tool(
                name="CalculateMLEnhancedScore",
                func=calculate_ml_enhanced_score_tool,
                description="Calculate ML-enhanced composite score using LSTM-DCF (50%) combined with traditional factors (50%). Includes fair value estimation, expected returns, and ML confirmation flags. Use this for advanced analysis with deep learning models. Returns detailed score breakdown with ML predictions."
            ),
            Tool(
                name="RankWatchlist",
                func=rank_watchlist_tool,
                description="Rank stocks in watchlist by composite score. Returns sorted list with rankings, percentiles, and top pick recommendation."
            ),
            Tool(
                name="DetectContrarianOpportunities",
                func=detect_contrarian_opportunities_tool,
                description="Detect contrarian opportunities (negative sentiment + strong fundamentals). Returns ranked list of suppressed stocks with mean-reversion potential for long-term gains."
            ),
            Tool(
                name="GenerateAlerts",
                func=generate_alerts_tool,
                description="Generate automated alerts for significant changes (score surges/drops, buy dips, overheated signals). Returns prioritized alert list."
            )
        ]
        
        return tools
    
    def _interpret_score(self, score: float, is_contrarian: bool) -> str:
        """Interpret composite score"""
        if score >= 0.75:
            base = "Strong buy opportunity"
        elif score >= 0.65:
            base = "Buy opportunity"
        elif score >= 0.45:
            base = "Hold position"
        else:
            base = "Weak/sell candidate"
        
        if is_contrarian:
            return f"{base} (Contrarian play - suppressed by negative sentiment)"
        return base
    
    def _create_agent(self):
        """Create the LangChain agent using langgraph"""
        try:
            # System prompt for the agent
            system_prompt = """You are a Watchlist Manager Agent specialized in building and ranking investment watchlists.

You have access to tools that can:
1. Calculate composite scores from multiple factors (growth, sentiment, valuation, risk, macro)
2. Detect contrarian opportunities (negative sentiment + strong fundamentals)
3. Generate investment signals based on scoring

When building a watchlist:
1. Use CompositeScore tool to calculate weighted scores
2. Use ContrarianDetector to find hidden opportunities
3. Use SignalGenerator to create actionable signals

Rank stocks from best to worst opportunity based on composite scores.
Highlight contrarian plays where sentiment diverges from fundamentals."""
            
            # Create agent using langgraph prebuilt
            agent = create_react_agent(
                self.llm,
                self.tools,
                prompt=system_prompt
            )
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Error creating Watchlist Manager Agent: {str(e)}")
            raise
    
    def build_watchlist(self, stocks_data: List[Dict]) -> Dict:
        """
        Build intelligent watchlist from stock data
        
        Args:
            stocks_data: List of dicts with ticker and component scores
            
        Returns:
            Ranked watchlist with signals
        """
        try:
            import json
            
            # Calculate composite scores
            scored_stocks = []
            for stock in stocks_data:
                scores_json = json.dumps({
                    'growth': stock.get('growth_score', 0.5),
                    'sentiment': stock.get('sentiment_score', 0.5),
                    'valuation': stock.get('valuation_score', 0.5),
                    'risk': stock.get('risk_score', 0.5),
                    'macro': stock.get('macro_score', 0.5)
                })
                
                # This would call the tool through agent
                # For direct usage, we'll calculate inline
                composite_score = (
                    stock.get('growth_score', 0.5) * 0.30 +
                    stock.get('sentiment_score', 0.5) * 0.25 +
                    stock.get('valuation_score', 0.5) * 0.20 +
                    stock.get('risk_score', 0.5) * 0.15 +
                    stock.get('macro_score', 0.5) * 0.10
                )
                
                # Contrarian bonus
                if stock.get('sentiment_score', 0.5) < 0.4 and stock.get('valuation_score', 0.5) > 0.7:
                    bonus = (0.7 - stock.get('sentiment_score', 0.5)) * stock.get('valuation_score', 0.5) * 0.2
                    composite_score += min(bonus, 0.15)
                
                scored_stocks.append({
                    'ticker': stock['ticker'],
                    'composite_score': round(composite_score * 100, 1),
                    'component_scores': {
                        'growth': stock.get('growth_score', 0.5),
                        'sentiment': stock.get('sentiment_score', 0.5),
                        'valuation': stock.get('valuation_score', 0.5),
                        'risk': stock.get('risk_score', 0.5),
                        'macro': stock.get('macro_score', 0.5)
                    }
                })
            
            # Rank
            watchlist_json = json.dumps(scored_stocks)
            query = f"Rank this watchlist and identify top opportunities: {watchlist_json}"
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'watchlist': result['output'],
                'agent': 'WatchlistManagerAgent',
                'model': 'llama-3.3-70b-versatile'
            }
            
        except Exception as e:
            self.logger.error(f"Error building watchlist: {str(e)}")
            return {'error': str(e)}
    
    def scan_for_contrarian_plays(self, stocks_data: List[Dict]) -> Dict:
        """
        Scan watchlist for contrarian opportunities
        
        Args:
            stocks_data: List with ticker, sentiment_score, valuation_score
            
        Returns:
            Contrarian opportunities list
        """
        try:
            import json
            
            watchlist_json = json.dumps(stocks_data)
            query = f"Detect contrarian opportunities in this watchlist: {watchlist_json}. Focus on stocks with negative sentiment but strong fundamentals for long-term mean-reversion gains."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'contrarian_analysis': result['output'],
                'agent': 'WatchlistManagerAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error scanning for contrarian plays: {str(e)}")
            return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    print("=== Watchlist Manager Agent Test ===\n")
    
    # Initialize agent
    agent = WatchlistManagerAgent()
    
    # Test watchlist building
    sample_stocks = [
        {'ticker': 'AAPL', 'growth_score': 0.7, 'sentiment_score': 0.6, 'valuation_score': 0.5, 'risk_score': 0.8, 'macro_score': 0.6},
        {'ticker': 'TSLA', 'growth_score': 0.9, 'sentiment_score': 0.3, 'valuation_score': 0.8, 'risk_score': 0.4, 'macro_score': 0.5},  # Contrarian
        {'ticker': 'MSFT', 'growth_score': 0.8, 'sentiment_score': 0.7, 'valuation_score': 0.6, 'risk_score': 0.9, 'macro_score': 0.7}
    ]
    
    print("Building watchlist...")
    watchlist = agent.build_watchlist(sample_stocks)
    print(f"\nWatchlist:\n{watchlist['watchlist']}")
    
    # Test contrarian detection
    print("\n\nScanning for contrarian plays...")
    contrarian = agent.scan_for_contrarian_plays([
        {'ticker': 'TSLA', 'sentiment_score': 0.3, 'valuation_score': 0.8},
        {'ticker': 'PFE', 'sentiment_score': 0.35, 'valuation_score': 0.75}
    ])
    print(f"\nContrarian Opportunities:\n{contrarian['contrarian_analysis']}")
