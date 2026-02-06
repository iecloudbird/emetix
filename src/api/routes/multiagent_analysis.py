"""
Multi-Agent Analysis API Routes

Provides deep AI analysis using the full multi-agent system:
- SupervisorAgent (orchestrator)
- DataFetcherAgent (stock data)
- SentimentAnalyzerAgent (news sentiment)
- FundamentalsAnalyzerAgent (financial metrics)
- EnhancedValuationAgent (LSTM-DCF + ML)
- WatchlistManagerAgent (risk scoring)

This is a SEPARATE endpoint from the basic LLM analysis,
designed for deeper, more comprehensive insights.
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import json

router = APIRouter(prefix="/multiagent", tags=["Multi-Agent Analysis"])

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=3)


# =============================================================================
# REAL SCORE COMPUTATION (data-driven, NOT from LLM text)
# =============================================================================

def _compute_sentiment_score(ticker: str) -> float:
    """
    Compute a real sentiment score (0-1) from actual market data.
    Combines: news sentiment (40%), analyst ratings (30%), price momentum (30%).
    """
    scores = []
    weights = []
    
    # --- 1. News Sentiment (40% weight) ---
    try:
        from src.data.fetchers.news_sentiment_fetcher import NewsSentimentFetcher
        fetcher = NewsSentimentFetcher()
        news_result = fetcher.fetch_all_news(ticker)
        news_score = news_result.get("sentiment_score", 0.5)
        scores.append(news_score)
        weights.append(0.4)
        logger.info(f"[{ticker}] News sentiment score: {news_score:.2f}")
    except Exception as e:
        logger.warning(f"[{ticker}] News sentiment failed: {e}")
    
    # --- 2. Analyst Ratings (30% weight) ---
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations is not None and not recommendations.empty:
            recent = recommendations.tail(20)
            rec_counts = recent['To Grade'].value_counts()
            
            buy_kw = ['buy', 'outperform', 'overweight', 'positive', 'strong buy']
            hold_kw = ['hold', 'neutral', 'equal', 'market perform', 'sector perform']
            sell_kw = ['sell', 'underperform', 'underweight', 'negative', 'reduce']
            
            buy_count = sum(
                rec_counts.get(g, 0) for g in rec_counts.index
                if any(k in g.lower() for k in buy_kw)
            )
            hold_count = sum(
                rec_counts.get(g, 0) for g in rec_counts.index
                if any(k in g.lower() for k in hold_kw)
            )
            sell_count = sum(
                rec_counts.get(g, 0) for g in rec_counts.index
                if any(k in g.lower() for k in sell_kw)
            )
            
            total = buy_count + hold_count + sell_count
            if total > 0:
                analyst_score = (buy_count + 0.5 * hold_count) / total
            else:
                analyst_score = 0.5
            
            scores.append(analyst_score)
            weights.append(0.3)
            logger.info(f"[{ticker}] Analyst sentiment score: {analyst_score:.2f} ({buy_count}B/{hold_count}H/{sell_count}S)")
    except Exception as e:
        logger.warning(f"[{ticker}] Analyst ratings failed: {e}")
    
    # --- 3. Price Momentum (30% weight) ---
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if not hist.empty and len(hist) >= 5:
            # 1-month return
            month_ago_idx = max(0, len(hist) - 22)
            month_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[month_ago_idx]) / hist['Close'].iloc[month_ago_idx]
            
            # Convert return to 0-1 score: -20% → 0.1, 0% → 0.5, +20% → 0.9
            momentum_score = max(0.05, min(0.95, 0.5 + month_return * 2.0))
            
            scores.append(momentum_score)
            weights.append(0.3)
            logger.info(f"[{ticker}] Momentum score: {momentum_score:.2f} (1mo return: {month_return:.1%})")
    except Exception as e:
        logger.warning(f"[{ticker}] Momentum calc failed: {e}")
    
    # --- Weighted average ---
    if scores and weights:
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(weighted_score, 2)
    
    return 0.5  # True fallback only if ALL sources failed


def _compute_fundamentals_scores(ticker: str) -> dict:
    """
    Compute real quality/growth/value scores (0-1) from actual financial data.
    
    Returns:
        {"quality_score": float, "growth_score": float, "value_score": float}
    """
    import yfinance as yf
    
    quality_score = 0.5
    growth_score = 0.5
    value_score = 0.5
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # --- Quality Score (profitability + financial health) ---
        quality_components = []
        
        # ROE: >20% excellent, >15% good, >10% avg, <10% poor
        roe = info.get('returnOnEquity')
        if roe is not None:
            roe_pct = roe * 100
            roe_s = min(1.0, max(0.0, roe_pct / 25.0))  # 25% ROE → 1.0
            quality_components.append(roe_s)
        
        # Profit margin: >20% excellent, >10% good
        margin = info.get('profitMargins')
        if margin is not None:
            margin_pct = margin * 100
            margin_s = min(1.0, max(0.0, margin_pct / 25.0))
            quality_components.append(margin_s)
        
        # Debt/Equity: <0.5 strong, <1.0 good, >2.0 weak
        de = info.get('debtToEquity')
        if de is not None:
            de_ratio = de / 100  # yfinance returns as percentage
            de_s = max(0.0, min(1.0, 1.0 - de_ratio / 3.0))  # 0 D/E → 1.0, 3.0 D/E → 0.0
            quality_components.append(de_s)
        
        # Current ratio: >2 strong, >1.5 good, <1 weak
        cr = info.get('currentRatio')
        if cr is not None:
            cr_s = min(1.0, max(0.0, cr / 2.5))  # 2.5 → 1.0
            quality_components.append(cr_s)
        
        if quality_components:
            quality_score = round(sum(quality_components) / len(quality_components), 2)
        
        # --- Growth Score (revenue + earnings growth) ---
        growth_components = []
        
        # Revenue growth
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None:
            rg_pct = rev_growth * 100
            rg_s = min(1.0, max(0.0, (rg_pct + 5) / 30.0))  # -5%→0, 25%→1.0
            growth_components.append(rg_s)
        
        # Earnings growth
        earn_growth = info.get('earningsGrowth')
        if earn_growth is not None:
            eg_pct = earn_growth * 100
            eg_s = min(1.0, max(0.0, (eg_pct + 10) / 40.0))  # -10%→0, 30%→1.0
            growth_components.append(eg_s)
        
        # Revenue growth quarterly
        rev_q = info.get('revenueQuarterlyGrowth')
        if rev_q is not None:
            rq_s = min(1.0, max(0.0, (rev_q * 100 + 5) / 30.0))
            growth_components.append(rq_s)
        
        if growth_components:
            growth_score = round(sum(growth_components) / len(growth_components), 2)
        
        # --- Value Score (valuation attractiveness) ---
        value_components = []
        
        # P/E: <15 undervalued, 15-25 fair, >25 expensive
        pe = info.get('trailingPE')
        if pe is not None and pe > 0:
            pe_s = max(0.0, min(1.0, 1.0 - (pe - 10) / 30.0))  # PE 10→1.0, PE 40→0.0
            value_components.append(pe_s)
        
        # PEG: <1 undervalued, 1-1.5 fair, >2 expensive
        peg = info.get('pegRatio')
        if peg is not None and peg > 0:
            peg_s = max(0.0, min(1.0, 1.0 - (peg - 0.5) / 2.0))  # PEG 0.5→1.0, PEG 2.5→0.0
            value_components.append(peg_s)
        
        # P/B: <1.5 undervalued, <3 fair, >5 expensive
        pb = info.get('priceToBook')
        if pb is not None and pb > 0:
            pb_s = max(0.0, min(1.0, 1.0 - (pb - 1) / 8.0))  # P/B 1→1.0, P/B 9→0.0
            value_components.append(pb_s)
        
        # EV/EBITDA: <10 cheap, 10-15 fair, >20 expensive
        ev_ebitda = info.get('enterpriseToEbitda')
        if ev_ebitda is not None and ev_ebitda > 0:
            eve_s = max(0.0, min(1.0, 1.0 - (ev_ebitda - 5) / 20.0))  # 5→1.0, 25→0.0
            value_components.append(eve_s)
        
        if value_components:
            value_score = round(sum(value_components) / len(value_components), 2)
        
        logger.info(f"[{ticker}] Fundamentals scores: quality={quality_score}, growth={growth_score}, value={value_score}")
        
    except Exception as e:
        logger.error(f"[{ticker}] Failed to compute fundamentals scores: {e}")
    
    return {
        "quality_score": quality_score,
        "growth_score": growth_score,
        "value_score": value_score
    }


def _extract_text_from_langchain_content(content: Any) -> str:
    """
    Extract plain text from LangChain message content.
    
    LangChain can return content in various formats:
    - str: Plain text
    - list: List of content blocks like [{"type": "text", "text": "...", "extras": {...}}]
    - dict: Object with "text" field
    
    This function normalizes all formats to plain text.
    """
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        # Handle list of content blocks
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                # Extract text from {type: "text", text: "..."} format
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
                elif "text" in block:
                    text_parts.append(str(block["text"]))
                elif "content" in block:
                    text_parts.append(str(block["content"]))
        return "\n".join(text_parts) if text_parts else str(content)
    
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _extract_text_from_langchain_content(content["content"])
        return str(content)
    
    return str(content)


def get_supervisor_agent():
    """Lazy import SupervisorAgent to avoid circular imports."""
    try:
        from src.agents.supervisor_agent import SupervisorAgent
        return SupervisorAgent()
    except Exception as e:
        logger.error(f"Failed to initialize SupervisorAgent: {e}")
        return None


def get_sentiment_agent():
    """Lazy import SentimentAnalyzerAgent."""
    try:
        from src.agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
        return SentimentAnalyzerAgent()
    except Exception as e:
        logger.error(f"Failed to initialize SentimentAnalyzerAgent: {e}")
        return None


def get_fundamentals_agent():
    """Lazy import FundamentalsAnalyzerAgent."""
    try:
        from src.agents.fundamentals_analyzer_agent import FundamentalsAnalyzerAgent
        return FundamentalsAnalyzerAgent()
    except Exception as e:
        logger.error(f"Failed to initialize FundamentalsAnalyzerAgent: {e}")
        return None


def get_enhanced_valuation_agent():
    """Lazy import EnhancedValuationAgent."""
    try:
        from src.agents.enhanced_valuation_agent import EnhancedValuationAgent
        return EnhancedValuationAgent()
    except Exception as e:
        logger.error(f"Failed to initialize EnhancedValuationAgent: {e}")
        return None


def get_llm():
    """Lazy import LLM provider."""
    from src.utils.llm_provider import get_llm as llm_factory
    return llm_factory(model_tier="large", temperature=0.2)


# =============================================================================
# MULTI-AGENT ANALYSIS ENDPOINTS
# =============================================================================

@router.get("/stock/{ticker}")
async def get_multiagent_analysis(
    ticker: str,
    include_sentiment: bool = Query(default=True, description="Include sentiment analysis from news"),
    include_fundamentals: bool = Query(default=True, description="Include fundamental analysis"),
    include_ml_valuation: bool = Query(default=True, description="Include LSTM-DCF ML valuation"),
    deep_analysis: bool = Query(default=False, description="Full orchestrated analysis (slower)"),
    deep_synthesis: bool = Query(default=False, description="Generate comprehensive diagnostic reasoning instead of brief summary")
):
    """
    Generate comprehensive multi-agent analysis for a stock.
    
    This endpoint leverages the full multi-agent system for deeper insights:
    - Sentiment: News analysis and market mood
    - Fundamentals: Quality, growth, value metrics
    - ML Valuation: LSTM-DCF fair value estimation
    
    Set deep_analysis=true for full SupervisorAgent orchestration (slowest but most comprehensive).
    Set deep_synthesis=true for detailed diagnostic reasoning (5-7 paragraphs) vs brief 3-4 sentence summary.
    """
    ticker = ticker.upper()
    
    try:
        result = {
            "status": "success",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "multi_agent_deep" if deep_analysis else "multi_agent",
            "agents_used": [],
            "sections": {}
        }
        
        if deep_analysis:
            # Full orchestrated analysis via SupervisorAgent
            supervisor = get_supervisor_agent()
            if supervisor:
                try:
                    loop = asyncio.get_event_loop()
                    orchestrated = await loop.run_in_executor(
                        _executor,
                        lambda: supervisor.analyze(f"Provide comprehensive analysis for {ticker}")
                    )
                    result["sections"]["orchestrated_analysis"] = {
                        "available": True,
                        "content": orchestrated,
                        "agent": "SupervisorAgent"
                    }
                    result["agents_used"].append("SupervisorAgent")
                except Exception as e:
                    logger.error(f"SupervisorAgent error for {ticker}: {e}")
                    result["sections"]["orchestrated_analysis"] = {
                        "available": False,
                        "error": str(e)
                    }
            else:
                result["sections"]["orchestrated_analysis"] = {
                    "available": False,
                    "error": "SupervisorAgent not available"
                }
        else:
            # Individual agent analysis (faster, modular)
            
            if include_sentiment:
                sentiment_result = await _get_sentiment_analysis(ticker)
                result["sections"]["sentiment"] = sentiment_result
                if sentiment_result.get("available"):
                    result["agents_used"].append("SentimentAnalyzerAgent")
            
            if include_fundamentals:
                fundamentals_result = await _get_fundamentals_analysis(ticker)
                result["sections"]["fundamentals"] = fundamentals_result
                if fundamentals_result.get("available"):
                    result["agents_used"].append("FundamentalsAnalyzerAgent")
            
            if include_ml_valuation:
                ml_result = await _get_ml_valuation(ticker)
                result["sections"]["ml_valuation"] = ml_result
                if ml_result.get("available"):
                    result["agents_used"].append("EnhancedValuationAgent")
            
            # Generate synthesis if we have multiple sections
            if len(result["agents_used"]) >= 2:
                synthesis = await _generate_synthesis(ticker, result["sections"], deep_mode=deep_synthesis)
                result["sections"]["synthesis"] = synthesis
        
        return result
        
    except Exception as e:
        logger.error(f"Multi-agent analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{ticker}/sentiment")
async def get_sentiment_only(ticker: str):
    """
    Get sentiment analysis only using SentimentAnalyzerAgent.
    Analyzes news, social media mentions, and market mood.
    """
    ticker = ticker.upper()
    
    try:
        result = await _get_sentiment_analysis(ticker)
        return {
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sentiment": result
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{ticker}/fundamentals")
async def get_fundamentals_only(ticker: str):
    """
    Get fundamental analysis only using FundamentalsAnalyzerAgent.
    Analyzes financial statements, ratios, and quality metrics.
    """
    ticker = ticker.upper()
    
    try:
        result = await _get_fundamentals_analysis(ticker)
        return {
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fundamentals": result
        }
    except Exception as e:
        logger.error(f"Fundamentals analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{ticker}/ml-valuation")
async def get_ml_valuation_only(ticker: str):
    """
    Get ML-powered valuation using EnhancedValuationAgent.
    Uses LSTM-DCF model for fair value estimation with consensus scoring.
    """
    ticker = ticker.upper()
    
    try:
        result = await _get_ml_valuation(ticker)
        return {
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ml_valuation": result
        }
    except Exception as e:
        logger.error(f"ML valuation error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/analyze")
async def analyze_watchlist(
    tickers: List[str],
    find_contrarian: bool = Query(default=False, description="Find contrarian opportunities")
):
    """
    Analyze a list of stocks using the multi-agent system.
    Optionally find contrarian opportunities (sentiment suppressed but fundamentally strong).
    """
    if not tickers or len(tickers) == 0:
        raise HTTPException(status_code=400, detail="No tickers provided")
    
    if len(tickers) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 tickers allowed per request")
    
    tickers = [t.upper() for t in tickers]
    
    try:
        result = {
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tickers_analyzed": len(tickers),
            "stocks": []
        }
        
        supervisor = get_supervisor_agent()
        
        for ticker in tickers:
            try:
                stock_result = {
                    "ticker": ticker,
                    "analysis": {}
                }
                
                # Quick sentiment + fundamentals for each
                sentiment = await _get_sentiment_analysis(ticker)
                fundamentals = await _get_fundamentals_analysis(ticker)
                
                stock_result["analysis"]["sentiment"] = sentiment
                stock_result["analysis"]["fundamentals"] = fundamentals
                
                # Flag contrarian opportunities
                if find_contrarian and sentiment.get("available") and fundamentals.get("available"):
                    sentiment_score = sentiment.get("sentiment_score", 0.5)
                    quality_score = fundamentals.get("quality_score", 0.5)
                    
                    # Contrarian = negative sentiment but strong fundamentals
                    if sentiment_score < 0.4 and quality_score > 0.6:
                        stock_result["contrarian_flag"] = True
                        stock_result["contrarian_reason"] = "Sentiment suppressed but fundamentally strong"
                    else:
                        stock_result["contrarian_flag"] = False
                
                result["stocks"].append(stock_result)
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                result["stocks"].append({
                    "ticker": ticker,
                    "error": str(e)
                })
        
        # Sort by contrarian flag if requested
        if find_contrarian:
            result["stocks"] = sorted(
                result["stocks"],
                key=lambda x: x.get("contrarian_flag", False),
                reverse=True
            )
            result["contrarian_opportunities"] = [
                s["ticker"] for s in result["stocks"] if s.get("contrarian_flag")
            ]
        
        return result
        
    except Exception as e:
        logger.error(f"Watchlist analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _get_sentiment_analysis(ticker: str) -> Dict[str, Any]:
    """Get sentiment analysis from SentimentAnalyzerAgent + compute real score from data."""
    try:
        agent = get_sentiment_agent()
        if not agent:
            return {"available": False, "error": "SentimentAnalyzerAgent not available"}
        
        loop = asyncio.get_event_loop()
        
        # Run LLM agent analysis AND real score computation in parallel
        agent_future = loop.run_in_executor(
            _executor,
            lambda: agent.analyze_comprehensive_sentiment(ticker)
        )
        score_future = loop.run_in_executor(
            None,
            lambda: _compute_sentiment_score(ticker)
        )
        
        result, computed_score = await asyncio.gather(agent_future, score_future)
        
        # Parse LLM text
        if isinstance(result, dict):
            raw_analysis = result.get("sentiment_analysis", "")
            sentiment_analysis = _extract_text_from_langchain_content(raw_analysis)
        else:
            sentiment_analysis = _extract_text_from_langchain_content(result)
        
        # Use the REAL computed score (not from LLM text)
        sentiment_score = computed_score
        
        return {
            "available": True,
            "agent": "SentimentAnalyzerAgent",
            "analysis": sentiment_analysis,
            "sentiment_score": sentiment_score,
            "sentiment_label": _score_to_label(sentiment_score)
        }
        
    except Exception as e:
        logger.error(f"Sentiment agent error for {ticker}: {e}")
        return {
            "available": False,
            "error": str(e)
        }


async def _get_fundamentals_analysis(ticker: str) -> Dict[str, Any]:
    """Get fundamental analysis from FundamentalsAnalyzerAgent + compute real scores from data."""
    try:
        agent = get_fundamentals_agent()
        if not agent:
            return {"available": False, "error": "FundamentalsAnalyzerAgent not available"}
        
        loop = asyncio.get_event_loop()
        
        # Run LLM agent analysis AND real score computation in parallel
        agent_future = loop.run_in_executor(
            _executor,
            lambda: agent.analyze_comprehensive_fundamentals(ticker)
        )
        score_future = loop.run_in_executor(
            None,
            lambda: _compute_fundamentals_scores(ticker)
        )
        
        result, computed_scores = await asyncio.gather(agent_future, score_future)
        
        # Parse LLM text
        if isinstance(result, dict):
            raw_analysis = result.get("fundamental_analysis", "")
            fundamental_analysis = _extract_text_from_langchain_content(raw_analysis)
        else:
            fundamental_analysis = _extract_text_from_langchain_content(result)
        
        # Use REAL computed scores (not from LLM text)
        quality_score = computed_scores["quality_score"]
        growth_score = computed_scores["growth_score"]
        value_score = computed_scores["value_score"]
        
        return {
            "available": True,
            "agent": "FundamentalsAnalyzerAgent",
            "analysis": fundamental_analysis,
            "quality_score": quality_score,
            "growth_score": growth_score,
            "value_score": value_score
        }
        
    except Exception as e:
        logger.error(f"Fundamentals agent error for {ticker}: {e}")
        return {
            "available": False,
            "error": str(e)
        }


async def _get_ml_valuation(ticker: str) -> Dict[str, Any]:
    """Get ML-powered valuation from EnhancedValuationAgent."""
    try:
        agent = get_enhanced_valuation_agent()
        if not agent:
            return {"available": False, "error": "EnhancedValuationAgent not available"}
        
        loop = asyncio.get_event_loop()
        query = f"Calculate the LSTM-DCF fair value and consensus score for {ticker}. Include margin of safety."
        result = await loop.run_in_executor(
            _executor,
            lambda: agent.analyze(query)
        )
        
        # Extract text from LangChain response format
        analysis_text = _extract_text_from_langchain_content(result)
        
        # Parse the result for structured data
        fair_value = None
        consensus_score = None
        margin_of_safety = None
        
        import re
        
        # Look for fair value
        fv_match = re.search(r'fair value[:\s]*\$?([\d.]+)', analysis_text, re.IGNORECASE)
        if fv_match:
            try:
                fair_value = float(fv_match.group(1))
            except ValueError:
                pass
        
        # Look for consensus score
        cs_match = re.search(r'consensus[:\s]*([\d.]+)', analysis_text, re.IGNORECASE)
        if cs_match:
            try:
                consensus_score = float(cs_match.group(1))
            except ValueError:
                pass
        
        # Look for margin of safety
        mos_match = re.search(r'margin[:\s]*([\d.]+)%?', analysis_text, re.IGNORECASE)
        if mos_match:
            try:
                margin_of_safety = float(mos_match.group(1))
            except ValueError:
                pass
        
        return {
            "available": True,
            "agent": "EnhancedValuationAgent",
            "analysis": analysis_text,
            "extracted_metrics": {
                "fair_value": fair_value,
                "consensus_score": consensus_score,
                "margin_of_safety": margin_of_safety
            }
        }
        
    except Exception as e:
        logger.error(f"ML valuation agent error for {ticker}: {e}")
        return {
            "available": False,
            "error": str(e)
        }


async def _generate_synthesis(ticker: str, sections: Dict[str, Any], deep_mode: bool = False) -> Dict[str, Any]:
    """Generate a synthesis of all agent analyses using LLM.
    
    Args:
        ticker: Stock ticker symbol
        sections: Dict of agent analysis sections
        deep_mode: If True, generate comprehensive diagnostic reasoning
    """
    try:
        llm = get_llm()
        
        # Build context from sections
        context_parts = []
        sentiment_score = 0.5
        quality_score = 0.5
        fair_value = None
        margin_of_safety = None
        
        if "sentiment" in sections and sections["sentiment"].get("available"):
            s = sections["sentiment"]
            sentiment_score = s.get("sentiment_score", 0.5)
            context_parts.append(f"SENTIMENT ({s.get('sentiment_label', 'Neutral')}, Score: {sentiment_score:.2f}): {s.get('analysis', 'N/A')[:800]}")
        
        if "fundamentals" in sections and sections["fundamentals"].get("available"):
            f = sections["fundamentals"]
            quality_score = f.get("quality_score", 0.5)
            growth_score = f.get("growth_score", 0.5)
            value_score = f.get("value_score", 0.5)
            context_parts.append(f"FUNDAMENTALS (Quality={quality_score:.2f}, Growth={growth_score:.2f}, Value={value_score:.2f}): {f.get('analysis', 'N/A')[:800]}")
        
        if "ml_valuation" in sections and sections["ml_valuation"].get("available"):
            m = sections["ml_valuation"]
            metrics = m.get("extracted_metrics", {})
            fair_value = metrics.get("fair_value")
            margin_of_safety = metrics.get("margin_of_safety")
            context_parts.append(f"ML VALUATION (Fair Value=${fair_value or 'N/A'}, Margin of Safety={margin_of_safety or 'N/A'}%): {m.get('analysis', 'N/A')[:800]}")
        
        if not context_parts:
            return {"available": False, "error": "No agent data to synthesize"}
        
        # Detect contrarian signal
        is_contrarian = sentiment_score < 0.4 and quality_score > 0.6
        contrarian_note = ""
        if is_contrarian:
            contrarian_note = "\n\n⚠️ CONTRARIAN SIGNAL DETECTED: Market sentiment is bearish but fundamentals are strong. This could indicate an accumulation opportunity if the negative sentiment is overblown."
        
        if deep_mode:
            # Deep diagnostic reasoning for "Deep Analysis" tab
            prompt = f"""You are a senior equity analyst providing a comprehensive diagnostic for {ticker}.

Here's what our specialized AI agents discovered:

{chr(10).join(context_parts)}

Provide a DETAILED diagnostic analysis (5-7 paragraphs) covering:

## 1. WHAT'S HAPPENING
- Current market perception (sentiment) vs actual business performance (fundamentals)
- Is the stock priced correctly relative to our ML fair value estimate?

## 2. WHY IS THIS HAPPENING
- What's driving the sentiment (positive or negative news)?
- Are there fundamental reasons justifying current price levels?
- Any disconnect between market narrative and financial reality?

## 3. KEY RISKS
- What could go wrong with this investment?
- What assumptions is the ML model making that might not hold?

## 4. OPPORTUNITY ASSESSMENT
- If undervalued: What catalyst could unlock value?
- If overvalued: What might cause a correction?
- Time horizon considerations (short-term vs long-term view)

## 5. ACTIONABLE RECOMMENDATION
- Clear BUY/HOLD/AVOID with reasoning
- Position sizing consideration (high conviction vs speculative)
- What to watch for (key metrics or events){contrarian_note}

Be specific with numbers where available. Avoid generic statements. Do NOT use emojis."""
        else:
            # Quick synthesis for lightweight display
                prompt = f"""You are synthesizing a multi-agent stock analysis for {ticker}.

    Here's what our specialized AI agents found:

    {chr(10).join(context_parts)}

    Provide a concise synthesis (3-4 sentences) that:
    1. Highlights the key insight from combining these perspectives
    2. Identifies any conflicts or agreements between agents
    3. Gives a clear actionable takeaway{contrarian_note}

    Keep it conversational and avoid jargon. Do NOT use emojis."""
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Include growth_score and value_score if available
        growth = sections.get("fundamentals", {}).get("growth_score", None)
        value = sections.get("fundamentals", {}).get("value_score", None)
        
        return {
            "available": True,
            "synthesis": content,
            "agents_combined": list(sections.keys()),
            "deep_mode": deep_mode,
            "contrarian_signal": is_contrarian,
            "scores_summary": {
                "sentiment": sentiment_score,
                "quality": quality_score,
                "growth": growth,
                "value": value,
                "fair_value": fair_value,
                "margin_of_safety": margin_of_safety
            }
        }
        
    except Exception as e:
        logger.error(f"Synthesis generation error for {ticker}: {e}")
        return {
            "available": False,
            "error": str(e)
        }


def _score_to_label(score: float) -> str:
    """Convert numeric score (0-1) to sentiment label."""
    if score >= 0.7:
        return "Very Bullish"
    elif score >= 0.55:
        return "Bullish"
    elif score >= 0.45:
        return "Neutral"
    elif score >= 0.3:
        return "Bearish"
    else:
        return "Very Bearish"
