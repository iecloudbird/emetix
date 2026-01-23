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
    deep_analysis: bool = Query(default=False, description="Full orchestrated analysis (slower)")
):
    """
    Generate comprehensive multi-agent analysis for a stock.
    
    This endpoint leverages the full multi-agent system for deeper insights:
    - Sentiment: News analysis and market mood
    - Fundamentals: Quality, growth, value metrics
    - ML Valuation: LSTM-DCF fair value estimation
    
    Set deep_analysis=true for full SupervisorAgent orchestration (slowest but most comprehensive).
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
                synthesis = await _generate_synthesis(ticker, result["sections"])
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
    """Get sentiment analysis from SentimentAnalyzerAgent."""
    try:
        agent = get_sentiment_agent()
        if not agent:
            return {"available": False, "error": "SentimentAnalyzerAgent not available"}
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: agent.analyze_comprehensive_sentiment(ticker)
        )
        
        # Parse result
        if isinstance(result, dict):
            sentiment_analysis = result.get("sentiment_analysis", "")
            sentiment_score = result.get("sentiment_score", 0.5)
        else:
            sentiment_analysis = str(result)
            sentiment_score = 0.5
        
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
    """Get fundamental analysis from FundamentalsAnalyzerAgent."""
    try:
        agent = get_fundamentals_agent()
        if not agent:
            return {"available": False, "error": "FundamentalsAnalyzerAgent not available"}
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: agent.analyze_comprehensive_fundamentals(ticker)
        )
        
        # Parse result
        if isinstance(result, dict):
            fundamental_analysis = result.get("fundamental_analysis", "")
            quality_score = result.get("quality_score", 0.5)
            growth_score = result.get("growth_score", 0.5)
            value_score = result.get("value_score", 0.5)
        else:
            fundamental_analysis = str(result)
            quality_score = growth_score = value_score = 0.5
        
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
        
        # Parse the result for structured data
        fair_value = None
        consensus_score = None
        margin_of_safety = None
        
        if isinstance(result, str):
            # Try to extract numbers from the response
            import re
            
            # Look for fair value
            fv_match = re.search(r'fair value[:\s]*\$?([\d.]+)', result, re.IGNORECASE)
            if fv_match:
                fair_value = float(fv_match.group(1))
            
            # Look for consensus score
            cs_match = re.search(r'consensus[:\s]*([\d.]+)', result, re.IGNORECASE)
            if cs_match:
                consensus_score = float(cs_match.group(1))
            
            # Look for margin of safety
            mos_match = re.search(r'margin[:\s]*([\d.]+)%?', result, re.IGNORECASE)
            if mos_match:
                margin_of_safety = float(mos_match.group(1))
        
        return {
            "available": True,
            "agent": "EnhancedValuationAgent",
            "analysis": result if isinstance(result, str) else str(result),
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


async def _generate_synthesis(ticker: str, sections: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a synthesis of all agent analyses using LLM."""
    try:
        llm = get_llm()
        
        # Build context from sections
        context_parts = []
        
        if "sentiment" in sections and sections["sentiment"].get("available"):
            s = sections["sentiment"]
            context_parts.append(f"SENTIMENT ({s.get('sentiment_label', 'Neutral')}): {s.get('analysis', 'N/A')[:500]}")
        
        if "fundamentals" in sections and sections["fundamentals"].get("available"):
            f = sections["fundamentals"]
            context_parts.append(f"FUNDAMENTALS (Quality={f.get('quality_score', 'N/A')}, Growth={f.get('growth_score', 'N/A')}): {f.get('analysis', 'N/A')[:500]}")
        
        if "ml_valuation" in sections and sections["ml_valuation"].get("available"):
            m = sections["ml_valuation"]
            metrics = m.get("extracted_metrics", {})
            context_parts.append(f"ML VALUATION (FV=${metrics.get('fair_value', 'N/A')}, MoS={metrics.get('margin_of_safety', 'N/A')}%): {m.get('analysis', 'N/A')[:500]}")
        
        if not context_parts:
            return {"available": False, "error": "No agent data to synthesize"}
        
        prompt = f"""You are synthesizing a multi-agent stock analysis for {ticker}.

Here's what our specialized AI agents found:

{chr(10).join(context_parts)}

Provide a concise synthesis (3-4 sentences) that:
1. Highlights the key insight from combining these perspectives
2. Identifies any conflicts or agreements between agents
3. Gives a clear actionable takeaway

Keep it conversational and avoid jargon. Do NOT use emojis."""
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "available": True,
            "synthesis": content,
            "agents_combined": list(sections.keys())
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
