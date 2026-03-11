"""
Enhanced Stock Screener API Routes
Comprehensive endpoints with LSTM-DCF, sector comparisons, AI insights, and chart data
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

router = APIRouter(prefix="/screener", tags=["Stock Screener"])

# Lazy-loaded instances (keyed by configuration)
_screeners = {}
_executor = ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger(__name__)


def get_screener(
    enable_consensus: bool = False, 
    enable_education: bool = False,
    use_full_universe: bool = False,
    max_universe_tickers: int = None
):
    """Lazy load the stock screener with specified features"""
    global _screeners
    
    from src.analysis.stock_screener import StockScreener
    
    # Create unique key for this configuration
    config_key = f"consensus={enable_consensus}_full={use_full_universe}_max={max_universe_tickers}"
    
    if config_key not in _screeners:
        _screeners[config_key] = StockScreener(
            use_extended_universe=True,
            use_full_universe=use_full_universe,
            max_universe_tickers=max_universe_tickers,
            enable_lstm=True,
            enable_consensus=enable_consensus,
            enable_education_mode=enable_education,
            enable_ai_insights=False
        )
        logger.info(f"Created screener: {config_key}")
    
    screener = _screeners[config_key]
    if enable_education != screener.enable_education_mode:
        screener.enable_education_mode = enable_education
    
    return screener


# ============================================================================
# REMOVED DEPRECATED ENDPOINTS (Phase 3 Cleanup - Jan 2025)
# ============================================================================
# The following endpoints have been removed and replaced by /api/pipeline/:
#
# - GET /watchlist -> Use /api/pipeline/qualified
# - GET /watchlist/simple -> Use /api/pipeline/qualified
# - GET /watchlist/categorized -> Use /api/pipeline/classified
# - GET /watchlist/for-profile/{profile_id} -> Use /api/pipeline/qualified?profile_id=X
#
# See /api/pipeline/ for the new 4-pillar scoring system.
# ============================================================================


@router.get("/stock/{ticker}")
async def get_stock_analysis(ticker: str):
    """
    Get detailed analysis for a specific stock
    
    Returns:
    - Fair value (LSTM-DCF & Traditional)
    - Sector comparison
    - All fundamental metrics
    - Justification
    """
    try:
        screener = get_screener()
        ticker = ticker.upper()
        
        # Fetch stock data
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            _executor,
            lambda: screener._fetch_stock_data(ticker)
        )
        
        if not data:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found or no data available")
        
        # Calculate valuation
        from src.analysis.stock_screener import SECTOR_BENCHMARKS
        sector = data.get('sector', 'Unknown')
        sector_data = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])
        
        data['valuation_score'] = screener._calculate_valuation_score(data)
        data['assessment'] = screener._get_assessment(data['valuation_score'])
        data['recommendation'] = screener._get_recommendation(data['valuation_score'], data)
        data['risk_level'] = screener._get_risk_level(data)
        data['justification'] = screener._generate_justification(data, data['valuation_score'], sector_data)
        
        return {
            'status': 'success',
            'data': data,
            'sector_benchmarks': sector_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts/{ticker}")
async def get_chart_data(ticker: str):
    """
    Get chart/visualization data for frontend graphs
    
    Returns:
    - 1-year price history (daily)
    - 5-year price history (weekly)
    - Technical indicators (MA50, MA200)
    - Key metrics for overlays
    """
    try:
        screener = get_screener()
        ticker = ticker.upper()
        
        loop = asyncio.get_event_loop()
        chart_data = await loop.run_in_executor(
            _executor,
            lambda: screener.get_chart_data(ticker)
        )
        
        if 'error' in chart_data:
            raise HTTPException(status_code=404, detail=chart_data['error'])
        
        return {
            'status': 'success',
            'data': chart_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sectors")
async def get_sector_benchmarks():
    """
    Get sector benchmark data for comparison
    
    Returns average P/E, P/B, ROE, and margins for each sector.
    After a market scan, these may be dynamically calculated from the scanned universe.
    """
    screener = get_screener()
    current_benchmarks = screener.get_current_benchmarks()
    
    return {
        'status': 'success',
        'description': 'Industry sector averages for valuation benchmarking',
        'metadata': current_benchmarks['metadata'],
        'benchmarks': current_benchmarks['benchmarks']
    }


@router.get("/sectors/{sector}")
async def get_sector_stocks(
    sector: str,
    n: int = Query(default=10, ge=1, le=30)
):
    """Get top undervalued stocks from a specific sector"""
    try:
        screener = get_screener()
        
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            _executor,
            lambda: screener.scan_market()
        )
        
        if df.empty:
            return {'status': 'success', 'sector': sector, 'stocks': []}
        
        # Filter by sector (case-insensitive)
        sector_df = df[df['sector'].str.lower() == sector.lower()]
        
        if sector_df.empty:
            return {
                'status': 'success',
                'sector': sector,
                'message': 'No stocks found in this sector',
                'available_sectors': df['sector'].unique().tolist()
            }
        
        top = sector_df.head(n)
        stocks = []
        for _, row in top.iterrows():
            stocks.append({
                'ticker': row['ticker'],
                'company': row['company_name'],
                'price': row['current_price'],
                'fair_value': row['fair_value'],
                'margin_of_safety': row['margin_of_safety'],
                'score': row['valuation_score'],
                'recommendation': row['recommendation'],
                'pe_ratio': round(row['pe_ratio'], 2),
                'roe': round(row['roe'], 2),
            })
        
        from src.analysis.stock_screener import SECTOR_BENCHMARKS
        sector_bench = SECTOR_BENCHMARKS.get(sector.title(), SECTOR_BENCHMARKS['Unknown'])
        
        return {
            'status': 'success',
            'sector': sector.title(),
            'benchmark': sector_bench,
            'stocks_count': len(stocks),
            'stocks': stocks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_stocks(
    tickers: str = Query(..., description="Comma-separated tickers (e.g., AAPL,MSFT,GOOGL)")
):
    """
    Compare multiple stocks side-by-side
    
    Returns valuation metrics and fair values for comparison
    """
    try:
        screener = get_screener()
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        if len(ticker_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 stocks for comparison")
        
        results = []
        for ticker in ticker_list:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                _executor,
                lambda t=ticker: screener._fetch_stock_data(t)
            )
            
            if data:
                from src.analysis.stock_screener import SECTOR_BENCHMARKS
                sector = data.get('sector', 'Unknown')
                sector_data = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])
                
                data['valuation_score'] = screener._calculate_valuation_score(data)
                data['recommendation'] = screener._get_recommendation(data['valuation_score'], data)
                
                results.append({
                    'ticker': ticker,
                    'company': data['company_name'],
                    'sector': sector,
                    'price': data['current_price'],
                    'fair_value': data['fair_value'],
                    'margin_of_safety': data['margin_of_safety'],
                    'score': data['valuation_score'],
                    'recommendation': data['recommendation'],
                    'pe_ratio': round(data['pe_ratio'], 2),
                    'sector_pe': sector_data['avg_pe'],
                    'pb_ratio': round(data['pb_ratio'], 2),
                    'roe': round(data['roe'], 2),
                    'debt_equity': round(data['debt_equity'], 2),
                    'fcf_yield': round(data['fcf_yield'], 2),
                    'dividend_yield': round(data['dividend_yield'], 2),
                })
            else:
                results.append({'ticker': ticker, 'error': 'No data available'})
        
        return {
            'status': 'success',
            'comparison': results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_market_summary():
    """
    Get market screening summary statistics
    """
    try:
        screener = get_screener()
        
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            _executor,
            lambda: screener.scan_market()
        )
        
        if df.empty:
            return {'status': 'success', 'message': 'No data available'}
        
        # Calculate summary stats
        summary = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_screened': len(screener.tickers),
            'passed_filters': len(df),
            'lstm_enabled': screener.lstm_model is not None,
            'statistics': {
                'avg_valuation_score': round(df['valuation_score'].mean(), 1),
                'avg_margin_of_safety': round(df['margin_of_safety'].mean(), 1),
                'avg_pe': round(df[df['pe_ratio'] > 0]['pe_ratio'].mean(), 1),
                'avg_roe': round(df['roe'].mean(), 1),
                'avg_debt_equity': round(df['debt_equity'].mean(), 2),
            },
            'distribution': {
                'strong_buy': len(df[df['recommendation'] == 'STRONG BUY']),
                'buy': len(df[df['recommendation'] == 'BUY']),
                'accumulate': len(df[df['recommendation'] == 'ACCUMULATE']),
                'hold': len(df[df['recommendation'] == 'HOLD']),
                'reduce': len(df[df['recommendation'] == 'REDUCE']),
                'sell': len(df[df['recommendation'] == 'SELL']),
            },
            'sector_breakdown': df.groupby('sector').size().to_dict(),
            'risk_distribution': df.groupby('risk_level').size().to_dict(),
            'top_5': df.head(5)[['ticker', 'valuation_score', 'margin_of_safety', 'recommendation']].to_dict('records'),
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REMOVED DEPRECATED ENDPOINTS (Phase 3 Cleanup - Jan 2025)
# ============================================================================
# - GET /universe -> Use /api/pipeline/summary
# - POST /scan -> Use /api/pipeline/trigger-scan
# ============================================================================


@router.get("/methodology")
async def get_methodology():
    """
    Explain the screening and scoring methodology
    """
    return {
        'status': 'success',
        'methodology': {
            'overview': 'Multi-Model Consensus scoring for undervalued stock identification',
            'architecture_note': 'Revised Jan 2025 - RF Ensemble deprecated (99% P/E), replaced with transparent GARP scoring',
            'fair_value_models': {
                'lstm_dcf': {
                    'description': 'Deep learning model predicting FCF growth rate',
                    'formula': 'FV = FCF × (1 + predicted_growth) / (WACC - terminal_growth)',
                    'inputs': '16 quarterly financial features (revenue, capex, margins, etc.)',
                    'training': 'Trained on S&P 500 quarterly financial statements',
                    'weight_in_consensus': '50%'
                },
                'garp_score': {
                    'description': 'Transparent Growth At Reasonable Price scoring',
                    'inputs': 'Forward P/E (50%) + PEG Ratio (50%)',
                    'purpose': 'Forward-looking valuation without black-box',
                    'weight_in_consensus': '25%',
                    'note': 'Replaces RF Ensemble which was 99% P/E'
                },
                'traditional_dcf': {
                    'description': 'Sector-adjusted P/E and P/B multiple valuation',
                    'formula': 'FV = 0.6 × (EPS × SectorPE) + 0.3 × (BookValue × SectorPB) + 0.1 × CurrentPrice',
                }
            },
            'consensus_scoring': {
                'description': 'Multi-model weighted voting for robust recommendations (Revised Jan 2025)',
                'weights': {
                    'lstm_dcf': '50% - Primary fair value signal',
                    'garp_score': '25% - Forward P/E + PEG (transparent)',
                    'risk_score': '25% - Beta + volatility safety filter'
                },
                'confidence': 'Based on model agreement (1 - standard deviation of scores)',
                'deprecated': 'RF Ensemble removed (was 30%, only used P/E at 99.93%)'
            },
            'valuation_score': {
                'pe_score': '20% weight - P/E ratio vs sector average',
                'pb_score': '15% weight - P/B ratio vs sector average',
                'margin_of_safety': '20% weight - Upside to fair value',
                'fcf_yield': '15% weight - Free cash flow yield',
                'financial_health': '15% weight - Debt/equity, current ratio',
                'profitability': '15% weight - ROE vs sector average',
            },
            'filters': {
                'market_cap': '≥ $1B (liquid, institutional-quality)',
                'pe_ratio': '0 < P/E ≤ 50 (profitable, reasonable valuation)',
                'debt_equity': '≤ 3.0 (manageable debt)',
                'volume': '≥ 100K daily average (liquid)',
            },
            'recommendations': {
                'STRONG BUY': 'Score ≥ 80 AND Margin of Safety > 20%',
                'BUY': 'Score ≥ 70 OR (Score ≥ 65 AND Analyst Upside > 20%)',
                'ACCUMULATE': 'Score 60-69',
                'HOLD': 'Score 50-59',
                'REDUCE': 'Score 40-49',
                'SELL': 'Score < 40',
            }
        }
    }


@router.get("/education/{ticker}")
async def get_educational_analysis(
    ticker: str,
    detail_level: str = Query(default="standard", description="Level of detail: basic, standard, comprehensive")
):
    """
    Get educational analysis for a specific stock.
    
    Designed for retail investors learning fundamental analysis.
    Provides plain-language explanations of all metrics.
    
    Detail levels:
    - basic: Key metrics with simple explanations
    - standard: Full metric breakdown with interpretations
    - comprehensive: Deep dive with investment thesis
    """
    try:
        screener = get_screener(enable_consensus=True, enable_education=True)
        ticker = ticker.upper()
        
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            _executor,
            lambda: screener._fetch_stock_data(ticker)
        )
        
        if not data:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
        
        # Calculate scores
        from src.analysis.stock_screener import SECTOR_BENCHMARKS
        sector = data.get('sector', 'Unknown')
        sector_data = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS['Unknown'])
        
        data['valuation_score'] = screener._calculate_valuation_score(data)
        data['effective_score'] = data['valuation_score']
        data['risk_level'] = screener._get_risk_level(data)
        data['sector_avg_pe'] = sector_data.get('avg_pe', 20)
        
        # Generate educational insights
        insights = screener._generate_educational_insights(data)
        
        # Build response based on detail level
        if detail_level == "basic":
            return {
                'status': 'success',
                'ticker': ticker,
                'company': data['company_name'],
                'current_price': data['current_price'],
                'fair_value': data['fair_value'],
                'overall_thesis': insights.get('overall', {}),
                'key_takeaway': f"{data['company_name']} appears {insights.get('overall', {}).get('thesis', 'fairly valued')}"
            }
        
        elif detail_level == "comprehensive":
            return {
                'status': 'success',
                'ticker': ticker,
                'company': data['company_name'],
                'sector': sector,
                'price_data': {
                    'current_price': data['current_price'],
                    'fair_value': data['fair_value'],
                    'margin_of_safety': data['margin_of_safety'],
                    'analyst_target': data.get('analyst_target')
                },
                'educational_insights': insights,
                'raw_metrics': {
                    'pe_ratio': data['pe_ratio'],
                    'pb_ratio': data['pb_ratio'],
                    'roe': data['roe'],
                    'debt_equity': data['debt_equity'],
                    'beta': data['beta'],
                    'fcf_yield': data['fcf_yield'],
                    'revenue_growth': data['revenue_growth']
                },
                'sector_comparison': {
                    'sector': sector,
                    'sector_avg_pe': sector_data['avg_pe'],
                    'sector_avg_roe': sector_data['avg_roe'],
                    'vs_sector': 'Below average P/E' if data['pe_ratio'] < sector_data['avg_pe'] else 'Above average P/E'
                },
                'learning_resources': {
                    'pe_ratio': 'Price-to-Earnings ratio measures how much investors pay per $1 of earnings. Lower is generally better value.',
                    'margin_of_safety': 'The difference between market price and estimated fair value. Provides cushion against estimation errors.',
                    'beta': 'Measures stock volatility relative to the market. Beta < 1 means less volatile than market.',
                    'roe': 'Return on Equity shows how efficiently a company uses shareholder capital to generate profits.'
                }
            }
        
        else:  # standard
            return {
                'status': 'success',
                'ticker': ticker,
                'company': data['company_name'],
                'sector': sector,
                'price_data': {
                    'current_price': data['current_price'],
                    'fair_value': data['fair_value'],
                    'margin_of_safety': data['margin_of_safety']
                },
                'educational_insights': insights,
                'recommendation': screener._get_recommendation(data['valuation_score'], data),
                'risk_level': data['risk_level']
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_model_status():
    """
    Get status of all ML models used in the screener
    """
    try:
        screener = get_screener(enable_consensus=True)
        
        return {
            'status': 'success',
            'architecture_version': 'Jan 2025 - RF deprecated, GARP scoring active',
            'models': {
                'lstm_dcf': {
                    'loaded': screener.lstm_model is not None,
                    'type': 'LSTM-DCF Enhanced' if screener.lstm_model and screener.lstm_model.lstm.input_size == 16 else 'LSTM-DCF Final' if screener.lstm_model else None,
                    'features': screener.lstm_model.lstm.input_size if screener.lstm_model else 0,
                    'weight': '50%',
                    'purpose': 'Fair value estimation via FCF growth prediction'
                },
                'garp_score': {
                    'active': True,
                    'weight': '25%',
                    'components': ['Forward P/E (50%)', 'PEG Ratio (50%)'],
                    'purpose': 'Transparent forward-looking valuation'
                },
                'risk_score': {
                    'active': True,
                    'weight': '25%',
                    'components': ['Beta', 'Volatility'],
                    'purpose': 'Low-risk stock filtering'
                },
                'rf_ensemble': {
                    'status': 'DEPRECATED (Jan 2025)',
                    'reason': 'P/E ratio = 99.93% importance, no multi-factor value',
                    'replacement': 'garp_score (transparent Forward P/E + PEG)'
                },
                'consensus_scorer': {
                    'loaded': screener.consensus_scorer is not None,
                    'weights': screener.consensus_scorer.weights if screener.consensus_scorer else None,
                    'purpose': 'Multi-model weighted voting + MoS penalty'
                }
            },
            'features': {
                'education_mode': screener.enable_education_mode,
                'dynamic_benchmarks': screener.use_dynamic_benchmarks,
                'ai_insights': screener.enable_ai_insights
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Multi-Agent Screening Endpoints
_supervisor_agent = None


def get_supervisor_agent():
    """Lazy load the SupervisorAgent for multi-agent orchestration"""
    global _supervisor_agent
    
    if _supervisor_agent is None:
        try:
            from src.agents.supervisor_agent import SupervisorAgent
            from config.settings import GROQ_API_KEY
            
            if not GROQ_API_KEY:
                return None
            
            _supervisor_agent = SupervisorAgent(api_key=GROQ_API_KEY)
        except Exception as e:
            # Log error but don't crash - agents are optional
            import logging
            logging.warning(f"Failed to initialize SupervisorAgent: {e}")
            return None
    
    return _supervisor_agent


@router.post("/agent/analyze")
async def agent_analyze_stock(ticker: str = Query(..., description="Stock ticker to analyze")):
    """
    Multi-Agent Stock Analysis
    
    Uses the SupervisorAgent to orchestrate:
    - DataFetcherAgent: Fetch comprehensive stock data
    - SentimentAnalyzerAgent: Analyze news and market sentiment
    - FundamentalsAnalyzerAgent: Deep fundamental analysis
    - EnhancedValuationAgent: ML-powered valuation (LSTM-DCF + RF + Consensus)
    
    Returns comprehensive analysis with investment recommendation.
    
    Requires: GROQ_API_KEY environment variable
    """
    try:
        supervisor = get_supervisor_agent()
        
        if supervisor is None:
            raise HTTPException(
                status_code=503,
                detail="Multi-agent system unavailable. Ensure GROQ_API_KEY is set."
            )
        
        # Run blocking agent operation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: supervisor.orchestrate_stock_analysis(ticker)
        )
        
        return {
            'status': 'success',
            'ticker': ticker,
            'analysis': result,
            'agent_type': 'supervisor',
            'sub_agents': ['data_fetcher', 'sentiment', 'fundamentals', 'enhanced_valuation']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/build-watchlist")
async def agent_build_watchlist(
    tickers: List[str] = Query(..., description="List of tickers to analyze"),
    include_sentiment: bool = Query(default=True, description="Include sentiment analysis"),
    include_contrarian: bool = Query(default=False, description="Detect contrarian opportunities")
):
    """
    Multi-Agent Watchlist Builder
    
    Uses the SupervisorAgent to:
    1. Fetch data for all tickers
    2. Score each stock on growth, sentiment, valuation, risk
    3. Detect contrarian opportunities (undervalued + negative sentiment)
    4. Rank and return intelligent watchlist
    
    Requires: GROQ_API_KEY environment variable
    """
    try:
        supervisor = get_supervisor_agent()
        
        if supervisor is None:
            raise HTTPException(
                status_code=503,
                detail="Multi-agent system unavailable. Ensure GROQ_API_KEY is set."
            )
        
        # Build watchlist using agent
        loop = asyncio.get_event_loop()
        
        if include_contrarian:
            result = await loop.run_in_executor(
                _executor,
                lambda: supervisor.find_contrarian_opportunities(tickers)
            )
        else:
            # Regular watchlist build
            result = await loop.run_in_executor(
                _executor,
                lambda: supervisor.build_intelligent_watchlist(tickers)
            )
        
        return {
            'status': 'success',
            'tickers_analyzed': len(tickers),
            'mode': 'contrarian' if include_contrarian else 'standard',
            'watchlist': result,
            'agent_type': 'supervisor'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/status")
async def get_agent_status():
    """
    Check status of multi-agent system
    """
    try:
        supervisor = get_supervisor_agent()
        
        if supervisor is None:
            return {
                'status': 'unavailable',
                'reason': 'GROQ_API_KEY not set or agent initialization failed',
                'fallback': 'Use /screener/watchlist with consensus=true for ML-based screening'
            }
        
        return {
            'status': 'available',
            'agents': {
                'supervisor': True,
                'data_fetcher': supervisor.data_fetcher is not None,
                'sentiment_analyzer': supervisor.sentiment_analyzer is not None,
                'fundamentals_analyzer': supervisor.fundamentals_analyzer is not None,
                'watchlist_manager': supervisor.watchlist_manager is not None,
                'enhanced_valuation': supervisor.enhanced_valuation is not None
            },
            'model': 'llama-3.3-70b-versatile',
            'capabilities': [
                'orchestrate_stock_analysis',
                'build_intelligent_watchlist',
                'find_contrarian_opportunities',
                'ml_powered_valuation'
            ]
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e)
        }


@router.get("/universe/info")
async def get_universe_info():
    """
    Get information about the available stock universe.
    
    Returns:
    - Curated universe: 214 S&P 500 + extended tickers (default)
    - Full universe: ~5,700+ US-traded common stocks from NASDAQ
    """
    try:
        from src.data.fetchers.ticker_universe import TickerUniverseFetcher
        
        # Get full universe count
        fetcher = TickerUniverseFetcher()
        full_universe = fetcher.get_all_us_tickers()
        
        # Get curated universe
        from src.analysis.stock_screener import StockScreener
        curated = StockScreener.SP500_TICKERS + StockScreener.EXTENDED_TICKERS
        curated = list(set(curated))
        
        return {
            'curated_universe': {
                'count': len(curated),
                'description': 'S&P 500 + Extended high-quality stocks',
                'sample': curated[:10]
            },
            'full_universe': {
                'count': len(full_universe),
                'description': 'All US-traded common stocks (NASDAQ + NYSE + AMEX)',
                'sample': full_universe[:10],
                'note': 'Full scan takes 10-30 minutes. Use max_tickers to limit.'
            },
            'api_usage': {
                'default': '/screener/watchlist (uses curated 214)',
                'full_scan': '/screener/watchlist?full_universe=true',
                'limited_scan': '/screener/watchlist?full_universe=true&max_tickers=500'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STOCK INFOGRAPHIC DATA (for visual summary cards)
# ============================================================================

@router.get("/infographic/{ticker}")
async def get_stock_infographic(ticker: str):
    """
    Return structured data for analyst-style stock infographic cards.

    Includes: company overview, annual + quarterly revenue with CAGR,
    net income, EPS trend, margin trends, valuation multiple compression,
    stock performance, beat-down detection, and major holder breakdown.
    All data sourced from Yahoo Finance — no LLM calls.
    """
    import yfinance as yf
    import numpy as np

    ticker = ticker.upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # ----- Company overview -----
        overview = {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
        }

        # ----- Annual revenue + CAGR (bar chart data) -----
        annual_revenue = []
        try:
            inc_annual = stock.income_stmt
            if inc_annual is not None and not inc_annual.empty:
                for col in reversed(list(inc_annual.columns)[:7]):
                    try:
                        year_label = str(col.year) if hasattr(col, "year") else str(col)[:4]
                    except Exception:
                        year_label = str(col)
                    rev = inc_annual.at["Total Revenue", col] if "Total Revenue" in inc_annual.index else None
                    if rev is not None and not (isinstance(rev, float) and np.isnan(rev)):
                        annual_revenue.append({"year": year_label, "revenue": float(rev)})
            # Add YoY growth %
            for i in range(1, len(annual_revenue)):
                prev = annual_revenue[i - 1]["revenue"]
                if prev and prev != 0:
                    annual_revenue[i]["yoy_growth"] = round(
                        (annual_revenue[i]["revenue"] - prev) / abs(prev) * 100, 1
                    )
        except Exception as e:
            logger.warning(f"[{ticker}] Annual revenue error: {e}")

        # Revenue CAGR (reuse shared utility)
        revenue_cagr = None
        try:
            from src.utils.financial_signals import calculate_revenue_cagr
            revenue_cagr = calculate_revenue_cagr(ticker, years=5)
        except Exception:
            pass

        # ----- Quarterly financials -----
        quarterly_revenue = []
        quarterly_net_income = []
        margin_trends = []

        try:
            inc = stock.quarterly_income_stmt
            if inc is not None and not inc.empty:
                for col in reversed(list(inc.columns)[:8]):
                    try:
                        period = f"{col.year}-Q{(col.month - 1) // 3 + 1}"
                    except Exception:
                        period = str(col)

                    rev = inc.at["Total Revenue", col] if "Total Revenue" in inc.index else None
                    ni = inc.at["Net Income", col] if "Net Income" in inc.index else None
                    gp = inc.at["Gross Profit", col] if "Gross Profit" in inc.index else None
                    op_inc = inc.at["Operating Income", col] if "Operating Income" in inc.index else None

                    if rev is not None and not (isinstance(rev, float) and np.isnan(rev)):
                        entry = {"period": period, "revenue": float(rev)}
                        quarterly_revenue.append(entry)
                    if ni is not None and not (isinstance(ni, float) and np.isnan(ni)):
                        quarterly_net_income.append({"period": period, "net_income": float(ni)})
                    if rev and float(rev) != 0:
                        margins = {"period": period}
                        if op_inc and not (isinstance(op_inc, float) and np.isnan(op_inc)):
                            margins["operating_margin"] = round(float(op_inc) / float(rev) * 100, 2)
                        if gp and not (isinstance(gp, float) and np.isnan(gp)):
                            margins["gross_margin"] = round(float(gp) / float(rev) * 100, 2)
                        if ni and not (isinstance(ni, float) and np.isnan(ni)):
                            margins["net_margin"] = round(float(ni) / float(rev) * 100, 2)
                        margin_trends.append(margins)
        except Exception as e:
            logger.warning(f"[{ticker}] Quarterly financials error: {e}")

        # YoY revenue growth for quarterly entries
        for i, entry in enumerate(quarterly_revenue):
            if i >= 4 and quarterly_revenue[i - 4]["revenue"]:
                prev = quarterly_revenue[i - 4]["revenue"]
                entry["yoy_growth"] = round((entry["revenue"] - prev) / abs(prev) * 100, 2)

        # ----- EPS trend -----
        eps_trend = []
        try:
            if info.get("trailingEps") is not None:
                eps_trend.append({"label": "Trailing EPS", "value": info["trailingEps"]})
            if info.get("forwardEps") is not None:
                eps_trend.append({"label": "Forward EPS", "value": info["forwardEps"]})
        except Exception:
            pass

        # ----- Valuation multiples + compression -----
        multiples = {}
        try:
            ps = info.get("priceToSalesTrailing12Months")
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            pfcf = info.get("priceToFreeCashflows") if "priceToFreeCashflows" in info else None
            ev_rev = info.get("enterpriseToRevenue")
            ev_ebitda = info.get("enterpriseToEbitda")

            if ps is not None:
                multiples["ps"] = {"current": round(ps, 2), "label": "P/S"}
            if pe is not None and pe > 0:
                multiples["pe"] = {"current": round(pe, 2), "label": "P/E"}
            if pb is not None and pb > 0:
                multiples["pb"] = {"current": round(pb, 2), "label": "P/B"}
            if pfcf is not None and pfcf > 0:
                multiples["pfcf"] = {"current": round(pfcf, 2), "label": "P/FCF"}
            if ev_rev is not None and ev_rev > 0:
                multiples["ev_rev"] = {"current": round(ev_rev, 2), "label": "EV/Rev"}
            if ev_ebitda is not None and ev_ebitda > 0:
                multiples["ev_ebitda"] = {"current": round(ev_ebitda, 2), "label": "EV/EBITDA"}

            # Estimate peak multiples from 52-week high ratio
            price_52h = info.get("fiftyTwoWeekHigh")
            price_now = overview["current_price"]
            if price_52h and price_now and price_now > 0:
                peak_ratio = price_52h / price_now
                for key in multiples:
                    multiples[key]["peak_est"] = round(multiples[key]["current"] * peak_ratio, 2)
                    curr = multiples[key]["current"]
                    peak = multiples[key]["peak_est"]
                    if peak > 0:
                        multiples[key]["compression_pct"] = round((peak - curr) / peak * 100, 1)
        except Exception as e:
            logger.warning(f"[{ticker}] Multiples error: {e}")

        # ----- Stock performance -----
        performance = {}
        try:
            hist = stock.history(period="1y")
            if not hist.empty and len(hist) >= 2:
                close = hist["Close"]
                current = float(close.iloc[-1])
                n = len(close)
                performance["1d"] = round((current - float(close.iloc[-2])) / float(close.iloc[-2]) * 100, 2) if n >= 2 else None
                performance["1w"] = round((current - float(close.iloc[-min(5, n)])) / float(close.iloc[-min(5, n)]) * 100, 2) if n >= 5 else None
                performance["1m"] = round((current - float(close.iloc[-min(22, n)])) / float(close.iloc[-min(22, n)]) * 100, 2) if n >= 22 else None
                performance["3m"] = round((current - float(close.iloc[-min(63, n)])) / float(close.iloc[-min(63, n)]) * 100, 2) if n >= 63 else None
                performance["6m"] = round((current - float(close.iloc[-min(126, n)])) / float(close.iloc[-min(126, n)]) * 100, 2) if n >= 126 else None
                performance["1y"] = round((current - float(close.iloc[0])) / float(close.iloc[0]) * 100, 2)
                performance["52w_high"] = float(close.max())
                performance["52w_low"] = float(close.min())
                performance["pct_from_52w_high"] = round((current - performance["52w_high"]) / performance["52w_high"] * 100, 1)
        except Exception as e:
            logger.warning(f"[{ticker}] Performance calc error: {e}")

        # ----- Beat-down detection -----
        beat_down = {"is_beat_down": False, "signals": []}
        try:
            pct_from_high = performance.get("pct_from_52w_high", 0)
            gross_margin = info.get("grossMargins", 0) or 0
            cagr_val = revenue_cagr["cagr"] if revenue_cagr else 0

            if pct_from_high <= -15:
                beat_down["signals"].append(f"Down {abs(pct_from_high):.0f}% from 52-week high")
            if cagr_val >= 15:
                beat_down["signals"].append(f"Revenue CAGR {cagr_val:.0f}% (strong growth)")
            if gross_margin >= 0.50:
                beat_down["signals"].append(f"Gross margin {gross_margin * 100:.0f}% (scalable)")

            # Check multiple compression
            avg_compression = 0
            comp_count = 0
            for m in multiples.values():
                if "compression_pct" in m and m["compression_pct"] > 0:
                    avg_compression += m["compression_pct"]
                    comp_count += 1
            if comp_count > 0:
                avg_compression /= comp_count
                if avg_compression >= 20:
                    beat_down["signals"].append(f"Avg multiple compression {avg_compression:.0f}%")

            # Beat-down = price dropped significantly + business still strong
            if (pct_from_high <= -15 and
                (cagr_val >= 15 or gross_margin >= 0.50) and
                len(beat_down["signals"]) >= 3):
                beat_down["is_beat_down"] = True
        except Exception:
            pass

        # ----- Major holders breakdown -----
        holders = {}
        try:
            from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
            fetcher = YFinanceFetcher()
            major = fetcher.fetch_major_holders(ticker)
            if major:
                holders = major.get("breakdown", {})
        except Exception:
            pass

        return {
            "status": "success",
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "overview": overview,
            "annual_revenue": annual_revenue,
            "revenue_cagr": revenue_cagr,
            "quarterly_revenue": quarterly_revenue,
            "quarterly_net_income": quarterly_net_income,
            "eps_trend": eps_trend,
            "margin_trends": margin_trends,
            "multiples": multiples,
            "performance": performance,
            "beat_down": beat_down,
            "major_holders": holders,
        }

    except Exception as e:
        logger.error(f"Infographic data error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BEAT-DOWN SCANNER
# ============================================================================

@router.get("/beat-down-scan")
async def beat_down_scan(
    tickers: str = Query(
        default="",
        description="Comma-separated tickers to scan. If empty, scans curated universe.",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
):
    """
    Scan stocks for beat-down opportunities — growth stocks with
    significant price drops but strong underlying fundamentals.

    Criteria:
    - Price >= 15% below 52-week high
    - Revenue CAGR >= 15% or Gross margin >= 50%
    - Multiple compression >= 20% from peak

    Returns ranked list of beat-down candidates.
    """
    import yfinance as yf

    # Determine ticker list
    if tickers.strip():
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        # Use curated universe subset
        try:
            from src.analysis.stock_screener import StockScreener
            ticker_list = (StockScreener.SP500_TICKERS + StockScreener.EXTENDED_TICKERS)[:200]
        except Exception:
            raise HTTPException(status_code=500, detail="Cannot load ticker universe")

    results = []

    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            high_52 = info.get("fiftyTwoWeekHigh")
            if not price or not high_52 or high_52 <= 0:
                continue

            pct_from_high = (price - high_52) / high_52 * 100
            if pct_from_high > -15:
                continue  # Not beat-down enough

            gross_margin = info.get("grossMargins", 0) or 0
            rev_growth = info.get("revenueGrowth", 0) or 0

            # Quick CAGR from available data
            cagr = None
            try:
                from src.utils.financial_signals import calculate_revenue_cagr
                cagr_data = calculate_revenue_cagr(ticker, years=3)
                if cagr_data:
                    cagr = cagr_data["cagr"]
            except Exception:
                pass

            # Must have strong business
            if not (gross_margin >= 0.50 or (cagr and cagr >= 15) or rev_growth >= 0.15):
                continue

            signals = []
            if pct_from_high <= -15:
                signals.append(f"Down {abs(pct_from_high):.0f}% from 52W high")
            if cagr and cagr >= 15:
                signals.append(f"CAGR {cagr:.0f}%")
            if gross_margin >= 0.50:
                signals.append(f"Gross margin {gross_margin * 100:.0f}%")
            if rev_growth >= 0.15:
                signals.append(f"Rev growth {rev_growth * 100:.0f}%")

            results.append({
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "sector": info.get("sector", ""),
                "current_price": round(price, 2),
                "pct_from_52w_high": round(pct_from_high, 1),
                "gross_margin": round(gross_margin * 100, 1) if gross_margin else None,
                "revenue_cagr": cagr,
                "revenue_growth": round(rev_growth * 100, 1) if rev_growth else None,
                "signals": signals,
                "signal_count": len(signals),
            })
        except Exception as e:
            logger.debug(f"Beat-down scan skip {ticker}: {e}")
            continue

    # Sort by signal strength (most signals first, then biggest drop)
    results.sort(key=lambda x: (-x["signal_count"], x["pct_from_52w_high"]))

    return {
        "status": "success",
        "generated_at": datetime.now().isoformat(),
        "scanned": len(ticker_list),
        "found": len(results[:limit]),
        "beat_down_stocks": results[:limit],
    }
