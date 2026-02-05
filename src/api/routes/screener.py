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
