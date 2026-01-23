"""
AI Analysis API Routes

Provides AI-driven stock analysis with:
- Educational explanations
- Diagnostic insights
- Investment thesis generation
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import logging

router = APIRouter(prefix="/analysis", tags=["AI Analysis"])

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def get_llm():
    """Lazy import LLM provider"""
    from src.utils.llm_provider import get_llm as llm_factory
    return llm_factory(model_tier="default", temperature=0.3)


def get_pipeline_db():
    """Lazy import pipeline DB"""
    from src.data.pipeline_db import pipeline_db, is_pipeline_available
    if not is_pipeline_available():
        return None
    return pipeline_db


def get_live_metrics(ticker: str) -> Optional[Dict]:
    """Fetch live metrics from Yahoo Finance for detailed analysis."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        
        if not info or not info.get("regularMarketPrice"):
            return None
        
        return {
            # Valuation
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "peg_ratio": _safe_float(info.get("pegRatio")),
            "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
            "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "current_price": _safe_float(info.get("regularMarketPrice")),
            "market_cap": info.get("marketCap"),
            # Fundamentals
            "roe_pct": _safe_float(info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None),
            "roa_pct": _safe_float(info.get("returnOnAssets") * 100 if info.get("returnOnAssets") else None),
            "profit_margin_pct": _safe_float(info.get("profitMargins") * 100 if info.get("profitMargins") else None),
            "gross_margin_pct": _safe_float(info.get("grossMargins") * 100 if info.get("grossMargins") else None),
            "operating_margin_pct": _safe_float(info.get("operatingMargins") * 100 if info.get("operatingMargins") else None),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "debt_equity": _safe_float(info.get("debtToEquity")),
            "quick_ratio": _safe_float(info.get("quickRatio")),
            # Growth
            "revenue_growth_pct": _safe_float(info.get("revenueGrowth") * 100 if info.get("revenueGrowth") else None),
            "earnings_growth_pct": _safe_float(info.get("earningsGrowth") * 100 if info.get("earningsGrowth") else None),
            "earnings_quarterly_growth_pct": _safe_float(info.get("earningsQuarterlyGrowth") * 100 if info.get("earningsQuarterlyGrowth") else None),
            # Risk / Dividend
            "beta": _safe_float(info.get("beta")),
            "dividend_yield_pct": _safe_float(info.get("dividendYield") * 100 if info.get("dividendYield") else None),
            "payout_ratio_pct": _safe_float(info.get("payoutRatio") * 100 if info.get("payoutRatio") else None),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch live metrics for {ticker}: {e}")
        return None


def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float, handling None and NaN."""
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return None if math.isnan(f) else round(f, 2)
    except (ValueError, TypeError):
        return None


def get_enhanced_valuation_agent():
    """Lazy import enhanced valuation agent"""
    from src.agents.enhanced_valuation_agent import EnhancedValuationAgent
    return EnhancedValuationAgent()


# =============================================================================
# AI ANALYSIS ENDPOINTS
# =============================================================================

@router.get("/stock/{ticker}")
async def get_ai_analysis(
    ticker: str,
    include_education: bool = Query(default=True, description="Include educational content"),
    include_diagnosis: bool = Query(default=True, description="Include diagnostic analysis"),
    include_thesis: bool = Query(default=True, description="Include investment thesis"),
    use_llm: bool = Query(default=False, description="Use Gemini LLM for richer analysis (slower)")
):
    """
    Generate comprehensive AI-driven analysis for a stock.
    
    Returns:
    - Educational: What the metrics mean for beginners
    - Diagnostic: Why the stock is valued this way
    - Investment Thesis: Bull case vs Bear case
    
    Set use_llm=true for Gemini-powered natural language analysis (slower but richer).
    """
    ticker = ticker.upper()
    
    try:
        # Get stock data from pipeline if available
        pipeline_data = None
        db = get_pipeline_db()
        if db:
            pipeline_data = db.get_qualified_by_ticker(ticker)
        
        # Build analysis sections
        result = {
            "status": "success",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "llm" if use_llm else "rule_based",
            "sections": {}
        }
        
        # Use LLM for richer content if requested
        if use_llm and pipeline_data:
            llm_analysis = await _generate_llm_analysis(ticker, pipeline_data)
            if llm_analysis:
                result["sections"]["llm_summary"] = llm_analysis
        
        if include_education:
            result["sections"]["education"] = await _generate_educational_content(
                ticker, pipeline_data
            )
        
        if include_diagnosis:
            result["sections"]["diagnosis"] = await _generate_diagnostic_analysis(
                ticker, pipeline_data
            )
        
        if include_thesis:
            result["sections"]["investment_thesis"] = await _generate_investment_thesis(
                ticker, pipeline_data
            )
        
        return result
        
    except Exception as e:
        logger.error(f"AI analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/{ticker}/quick")
async def get_quick_analysis(ticker: str):
    """
    Get a quick AI summary for a stock (faster, less detailed).
    Suitable for list views and tooltips.
    """
    ticker = ticker.upper()
    
    try:
        db = get_pipeline_db()
        pipeline_data = db.get_qualified_by_ticker(ticker) if db else None
        
        if not pipeline_data:
            # Fallback to basic fetcher
            from src.data.fetchers import YFinanceFetcher
            fetcher = YFinanceFetcher()
            stock_data = fetcher.fetch_stock_data(ticker)
            if stock_data is None:
                raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
            
            return {
                "status": "success",
                "ticker": ticker,
                "summary": _generate_quick_summary_from_fetched(stock_data),
                "source": "live"
            }
        
        # Use pipeline data for richer summary
        return {
            "status": "success",
            "ticker": ticker,
            "summary": _generate_quick_summary(pipeline_data),
            "source": "pipeline"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LLM-POWERED ANALYSIS (Gemini)
# =============================================================================

async def _generate_llm_analysis(ticker: str, data: Dict) -> Optional[Dict]:
    """
    Generate rich analysis using Gemini LLM.
    Returns natural language investment analysis.
    """
    try:
        llm = get_llm()
        
        # Prepare context for LLM
        pillars = data.get("pillar_scores", {})
        composite = data.get("composite_score", 0)
        mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
        classification = data.get("classification", "Unknown")
        name = data.get("name", ticker)
        sector = data.get("sector", "Unknown")
        price = data.get("price", 0) or 0
        
        # Calculate fair value from MoS if not directly available
        fair_value = data.get("fair_value") or data.get("lstm_fair_value")
        if not fair_value and price and mos:
            fair_value = price / (1 - mos / 100) if mos < 100 else price * 2
        
        # Format fair value string
        fair_value_str = f"${fair_value:.2f}" if fair_value else "N/A"
        
        # Build prompt
        prompt = f"""You are a friendly stock analyst helping retail investors understand stocks simply.

Analyze this stock based on our ML-powered screening data:

**{ticker} - {name}**
- Sector: {sector}
- Current Price: ${price:.2f}
- Our Fair Value Estimate: {fair_value_str}
- Composite Score: {composite:.0f}/100
- Margin of Safety: {mos:.1f}%
- Classification: {classification}
- Pillar Scores: Value={pillars.get('value', 0):.0f}, Quality={pillars.get('quality', 0):.0f}, Growth={pillars.get('growth', 0):.0f}, Safety={pillars.get('safety', 0):.0f}

Provide your analysis in this exact format:

## TLDR
One sentence verdict for busy investors.

## Why {classification}?
2-3 sentences explaining why this stock got this rating in plain English.

## 💪 Key Strength
The most compelling reason to consider this stock.

## ⚠️ Main Risk
The biggest concern investors should be aware of.

## 🎯 What To Do
Actionable suggestion for the investor.

Keep it conversational and avoid jargon. Be honest about limitations."""

        # Call LLM
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "available": True,
            "generated_by": "gemini",
            "analysis": content,
            "context": {
                "ticker": ticker,
                "composite_score": composite,
                "classification": classification,
                "margin_of_safety": mos
            },
            "data_context": {
                "fair_value": data.get("fair_value") or data.get("lstm_fair_value"),
                "current_price": price,
                "consensus_score": composite,
                "margin_of_safety": mos,
                "classification": classification,
                "sector": sector
            }
        }
        
    except Exception as e:
        logger.warning(f"LLM analysis failed for {ticker}: {e}")
        return {
            "available": False,
            "error": str(e),
            "fallback": "Using rule-based analysis instead"
        }


# =============================================================================
# RULE-BASED ANALYSIS FUNCTIONS
# =============================================================================

async def _generate_educational_content(ticker: str, data: Optional[Dict]) -> Dict:
    """
    Generate educational content explaining what the metrics mean.
    Targeted at beginner investors.
    """
    if not data:
        return {
            "available": False,
            "message": f"{ticker} not in pipeline. Run Stage 2 scoring first."
        }
    
    pillars = data.get("pillar_scores", {})
    composite = data.get("composite_score", 0)
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    classification = data.get("classification", "Unknown")
    
    # Build educational sections
    education = {
        "available": True,
        "what_is_fair_value": {
            "title": "Understanding Fair Value",
            "content": f"""Fair value is an estimate of what a stock is truly worth based on its fundamentals.
            
For {ticker}, our ML model (LSTM-DCF) estimates the fair value by forecasting future cash flows.
A positive Margin of Safety ({mos:.1f}%) means the stock trades below its estimated fair value.""",
            "key_insight": "undervalued" if mos > 15 else "fairly_valued" if mos > -10 else "overvalued"
        },
        "pillar_scores_explained": {
            "title": "The 4-Pillar Scoring System",
            "pillars": {
                "value": {
                    "score": pillars.get("value", 0),
                    "meaning": "How cheap is the stock relative to its fundamentals?",
                    "what_it_measures": "Price-to-Earnings, Price-to-Book, FCF Yield, Margin of Safety"
                },
                "quality": {
                    "score": pillars.get("quality", 0),
                    "meaning": "How efficiently does the company use capital?",
                    "what_it_measures": "Return on Equity (ROE), Return on Invested Capital (ROIC), Profit Margins"
                },
                "growth": {
                    "score": pillars.get("growth", 0),
                    "meaning": "How fast is the company growing?",
                    "what_it_measures": "Revenue Growth, Earnings Growth, LSTM-predicted future growth"
                },
                "safety": {
                    "score": pillars.get("safety", 0),
                    "meaning": "How risky is this investment?",
                    "what_it_measures": "Beta (market volatility), Debt levels, Drawdown history"
                }
            }
        },
        "classification_explained": {
            "title": "What Does the Classification Mean?",
            "current": classification,
            "meanings": {
                "Buy": "Stock meets criteria for immediate purchase: high MoS (≥20%) + high composite score (≥70)",
                "Hold": "Good quality stock near fair value: monitor for better entry points",
                "Watch": "Interesting but not yet actionable: may be too expensive or have quality concerns"
            }
        },
        "risk_education": {
            "title": "Understanding Investment Risk",
            "beta_explained": f"Beta measures how much a stock moves relative to the market. Beta > 1 = more volatile than market.",
            "diversification_tip": "Never put all eggs in one basket. Consider position sizing based on your risk tolerance."
        }
    }
    
    return education


async def _generate_diagnostic_analysis(ticker: str, data: Optional[Dict]) -> Dict:
    """
    Generate diagnostic analysis explaining WHY the stock is valued this way.
    Now includes actual metrics to back up the claims.
    """
    if not data:
        return {
            "available": False,
            "message": f"{ticker} not in pipeline. Run Stage 2 scoring first."
        }
    
    pillars = data.get("pillar_scores", {})
    composite = data.get("composite_score", 0)
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    classification = data.get("classification", "Unknown")
    sector = data.get("sector", "Unknown")
    price = data.get("price", 0) or 0
    
    # Fetch live metrics for detailed context
    live_metrics = get_live_metrics(ticker)
    
    # Calculate fair value
    fair_value = price / (1 - mos / 100) if mos and mos < 100 and price else 0
    
    # Identify strengths and weaknesses
    pillar_list = [(k, v) for k, v in pillars.items() if isinstance(v, (int, float))]
    sorted_pillars = sorted(pillar_list, key=lambda x: x[1], reverse=True)
    
    strengths = [(p, s) for p, s in sorted_pillars if s >= 70]
    weaknesses = [(p, s) for p, s in sorted_pillars if s < 50]
    
    # Build diagnostic content
    diagnosis = {
        "available": True,
        "overall_assessment": {
            "composite_score": composite,
            "interpretation": _interpret_composite_score(composite),
            "margin_of_safety": mos,
            "valuation_status": "undervalued" if mos > 15 else "fairly_valued" if mos > -10 else "overvalued"
        },
        # Include actual metrics for context
        "metrics_snapshot": _build_metrics_snapshot(ticker, live_metrics, data, fair_value),
        "strengths": {
            "count": len(strengths),
            "pillars": [
                {
                    "name": p.title(),
                    "score": s,
                    "why_strong": _explain_pillar_strength(p, s, data),
                    "supporting_metrics": _get_pillar_metrics(p, live_metrics)
                }
                for p, s in strengths
            ]
        },
        "weaknesses": {
            "count": len(weaknesses),
            "pillars": [
                {
                    "name": p.title(),
                    "score": s,
                    "concern": _explain_pillar_weakness(p, s, data)
                }
                for p, s in weaknesses
            ]
        },
        "sector_context": {
            "sector": sector,
            "note": f"Scores are relative to the {sector} sector average where applicable."
        },
        "key_drivers": _identify_key_drivers(data),
        "red_flags": _identify_red_flags(data),
        "catalysts": _identify_catalysts(data)
    }
    
    return diagnosis


async def _generate_investment_thesis(ticker: str, data: Optional[Dict]) -> Dict:
    """
    Generate investment thesis with bull case and bear case.
    """
    if not data:
        return {
            "available": False,
            "message": f"{ticker} not in pipeline. Run Stage 2 scoring first."
        }
    
    pillars = data.get("pillar_scores", {})
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    classification = data.get("classification", "Unknown")
    name = data.get("name", ticker)
    
    # Build thesis
    thesis = {
        "available": True,
        "company": name,
        "ticker": ticker,
        "recommendation": classification,
        "conviction": _calculate_conviction(data),
        "bull_case": {
            "summary": _generate_bull_case_summary(data),
            "points": _generate_bull_case_points(data)
        },
        "bear_case": {
            "summary": _generate_bear_case_summary(data),
            "points": _generate_bear_case_points(data)
        },
        "base_case": {
            "expected_return": _estimate_expected_return(data),
            "timeframe": "12-18 months",
            "key_assumption": "Market recognizes fair value over time"
        },
        "action_items": _generate_action_items(data)
    }
    
    return thesis


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_quick_summary(data: Dict) -> Dict:
    """Generate quick summary from pipeline data."""
    pillars = data.get("pillar_scores", {})
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    composite = data.get("composite_score", 0)
    
    # Find best pillar
    best_pillar = max(pillars.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0) if pillars else ("unknown", 0)
    
    if mos > 30:
        headline = f"Significantly undervalued with {mos:.0f}% margin of safety"
    elif mos > 15:
        headline = f"Attractive valuation with {mos:.0f}% upside potential"
    elif mos > 0:
        headline = f"Modestly undervalued at {mos:.0f}% below fair value"
    elif mos > -10:
        headline = "Trading near fair value"
    else:
        headline = f"Currently overvalued by {abs(mos):.0f}%"
    
    return {
        "headline": headline,
        "composite_score": composite,
        "best_pillar": {"name": best_pillar[0].title(), "score": best_pillar[1]},
        "classification": data.get("classification", "Unknown"),
        "one_liner": f"Score: {composite:.0f}/100 | Best: {best_pillar[0].title()} ({best_pillar[1]:.0f})"
    }


def _generate_quick_summary_from_fetched(data: Dict) -> Dict:
    """Generate quick summary from live fetched data."""
    pe = data.get("pe_ratio", 0)
    
    return {
        "headline": "Basic metrics available (not in pipeline)",
        "pe_ratio": pe,
        "source": "live_fetch",
        "one_liner": f"P/E: {pe:.1f}" if pe else "Limited data available"
    }


def _interpret_composite_score(score: float) -> str:
    """Interpret composite score."""
    if score >= 80:
        return "Excellent - Top tier quality and value combination"
    elif score >= 70:
        return "Good - Above average on most metrics"
    elif score >= 60:
        return "Fair - Meets minimum quality threshold"
    elif score >= 50:
        return "Below Average - Some concerns present"
    else:
        return "Poor - Multiple red flags"


def _explain_pillar_strength(pillar: str, score: float, data: Dict) -> str:
    """Explain why a pillar is strong."""
    explanations = {
        "value": f"Trading at attractive valuation with potential upside",
        "quality": "Strong capital efficiency and profit margins",
        "growth": "Above-average growth trajectory",
        "safety": "Lower volatility and manageable risk profile",
        "momentum": "Positive price momentum and technical setup"
    }
    return explanations.get(pillar, f"Strong {pillar} characteristics")

def _build_metrics_snapshot(ticker: str, live_metrics: Optional[Dict], data: Dict, fair_value: float) -> Dict:
    """Build a snapshot of key metrics for display."""
    price = data.get("price", 0) or 0
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    snapshot = {
        "valuation": {
            "current_price": price,
            "fair_value_estimate": round(fair_value, 2) if fair_value else None,
            "margin_of_safety_pct": round(mos, 1) if mos else None,
        },
        "fundamentals": {},
        "growth": {},
        "risk": {}
    }
    
    if live_metrics:
        # Valuation metrics
        if live_metrics.get("pe_ratio"):
            snapshot["valuation"]["pe_ratio"] = live_metrics["pe_ratio"]
        if live_metrics.get("forward_pe"):
            snapshot["valuation"]["forward_pe"] = live_metrics["forward_pe"]
        if live_metrics.get("pb_ratio"):
            snapshot["valuation"]["pb_ratio"] = live_metrics["pb_ratio"]
        if live_metrics.get("peg_ratio"):
            snapshot["valuation"]["peg_ratio"] = live_metrics["peg_ratio"]
        
        # Fundamentals (now using _pct suffix)
        if live_metrics.get("roe_pct"):
            snapshot["fundamentals"]["roe_pct"] = live_metrics["roe_pct"]
        if live_metrics.get("roa_pct"):
            snapshot["fundamentals"]["roa_pct"] = live_metrics["roa_pct"]
        if live_metrics.get("profit_margin_pct"):
            snapshot["fundamentals"]["profit_margin_pct"] = live_metrics["profit_margin_pct"]
        if live_metrics.get("gross_margin_pct"):
            snapshot["fundamentals"]["gross_margin_pct"] = live_metrics["gross_margin_pct"]
        if live_metrics.get("current_ratio"):
            snapshot["fundamentals"]["current_ratio"] = live_metrics["current_ratio"]
        if live_metrics.get("debt_equity"):
            snapshot["fundamentals"]["debt_to_equity"] = live_metrics["debt_equity"]
        
        # Growth (now using _pct suffix)
        if live_metrics.get("revenue_growth_pct"):
            snapshot["growth"]["revenue_growth_pct"] = live_metrics["revenue_growth_pct"]
        if live_metrics.get("earnings_growth_pct"):
            snapshot["growth"]["earnings_growth_pct"] = live_metrics["earnings_growth_pct"]
        
        # Risk
        if live_metrics.get("beta"):
            snapshot["risk"]["beta"] = live_metrics["beta"]
        if live_metrics.get("dividend_yield_pct"):
            snapshot["risk"]["dividend_yield_pct"] = live_metrics["dividend_yield_pct"]
    
    return snapshot


def _get_pillar_metrics(pillar: str, live_metrics: Optional[Dict]) -> list:
    """Get supporting metrics for a pillar to back up claims."""
    if not live_metrics:
        return []
    
    metrics = []
    
    if pillar == "value":
        if live_metrics.get("pe_ratio"):
            pe = live_metrics["pe_ratio"]
            assessment = "attractive" if pe < 20 else "fair" if pe < 30 else "premium"
            metrics.append({"name": "P/E", "value": f"{pe:.1f}x", "note": f"{assessment}"})
        if live_metrics.get("pb_ratio"):
            pb = live_metrics["pb_ratio"]
            assessment = "below book" if pb < 1 else "near book" if pb < 2 else "above book"
            metrics.append({"name": "P/B", "value": f"{pb:.1f}x", "note": assessment})
        if live_metrics.get("peg_ratio"):
            peg = live_metrics["peg_ratio"]
            assessment = "attractive" if peg < 1 else "fair" if peg < 2 else "expensive"
            metrics.append({"name": "PEG", "value": f"{peg:.2f}", "note": assessment})
    
    elif pillar == "quality":
        if live_metrics.get("roe_pct"):
            roe = live_metrics["roe_pct"]
            assessment = "excellent" if roe > 20 else "good" if roe > 15 else "moderate"
            metrics.append({"name": "ROE", "value": f"{roe:.1f}%", "note": f"{assessment}"})
        if live_metrics.get("profit_margin_pct"):
            pm = live_metrics["profit_margin_pct"]
            assessment = "strong" if pm > 15 else "healthy" if pm > 8 else "thin"
            metrics.append({"name": "Margin", "value": f"{pm:.1f}%", "note": f"{assessment}"})
        if live_metrics.get("current_ratio"):
            cr = live_metrics["current_ratio"]
            assessment = "solid" if cr > 1.5 else "adequate" if cr > 1 else "tight"
            metrics.append({"name": "Current", "value": f"{cr:.2f}", "note": f"{assessment}"})
    
    elif pillar == "growth":
        if live_metrics.get("revenue_growth_pct"):
            rg = live_metrics["revenue_growth_pct"]
            assessment = "high" if rg > 20 else "solid" if rg > 10 else "modest"
            metrics.append({"name": "Rev Growth", "value": f"{rg:.1f}%", "note": assessment})
        if live_metrics.get("earnings_growth_pct"):
            eg = live_metrics["earnings_growth_pct"]
            assessment = "accelerating" if eg > 25 else "growing" if eg > 10 else "stable"
            metrics.append({"name": "EPS Growth", "value": f"{eg:.1f}%", "note": assessment})
        if live_metrics.get("peg_ratio"):
            peg = live_metrics["peg_ratio"]
            assessment = "attractive" if peg < 1 else "fair" if peg < 2 else "expensive"
            metrics.append({"name": "PEG", "value": f"{peg:.2f}", "note": f"{assessment}"})
    
    elif pillar == "safety":
        if live_metrics.get("beta"):
            beta = live_metrics["beta"]
            assessment = "low vol" if beta < 0.8 else "market-like" if beta < 1.2 else "volatile"
            metrics.append({"name": "Beta", "value": f"{beta:.2f}", "note": assessment})
        if live_metrics.get("debt_equity"):
            de = live_metrics["debt_equity"]
            assessment = "low debt" if de < 50 else "moderate" if de < 100 else "leveraged"
            metrics.append({"name": "D/E", "value": f"{de:.0f}%", "note": assessment})
        if live_metrics.get("dividend_yield_pct"):
            dy = live_metrics["dividend_yield_pct"]
            if dy and dy > 0:
                metrics.append({"name": "Div Yield", "value": f"{dy:.2f}%", "note": "income"})
    
    return metrics[:3]  # Limit to top 3 metrics per pillar

def _explain_pillar_weakness(pillar: str, score: float, data: Dict) -> str:
    """Explain pillar weakness concerns."""
    concerns = {
        "value": "May be expensive relative to fundamentals",
        "quality": "Questions about capital efficiency or profitability",
        "growth": "Growth may be slowing or below expectations",
        "safety": "Higher volatility or risk factors present",
        "momentum": "Weak technical setup or negative price trend"
    }
    return concerns.get(pillar, f"Weakness in {pillar} metrics")


def _identify_key_drivers(data: Dict) -> list:
    """Identify key value drivers."""
    drivers = []
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    pillars = data.get("pillar_scores", {})
    
    if mos > 25:
        drivers.append("Significant undervaluation provides margin of safety")
    if pillars.get("quality", 0) > 75:
        drivers.append("High-quality business with strong fundamentals")
    if pillars.get("growth", 0) > 70:
        drivers.append("Strong growth profile supports future value creation")
    if pillars.get("safety", 0) > 70:
        drivers.append("Lower risk profile suitable for conservative investors")
    
    return drivers if drivers else ["Standard metrics within normal ranges"]


def _identify_red_flags(data: Dict) -> list:
    """Identify potential red flags."""
    flags = []
    pillars = data.get("pillar_scores", {})
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    if pillars.get("quality", 100) < 40:
        flags.append("Quality concerns - verify business fundamentals")
    if pillars.get("safety", 100) < 40:
        flags.append("Higher risk profile - consider position sizing")
    if mos < -20:
        flags.append("Appears significantly overvalued")
    
    return flags if flags else []


def _identify_catalysts(data: Dict) -> list:
    """Identify potential catalysts."""
    catalysts = []
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    pillars = data.get("pillar_scores", {})
    
    if mos > 20:
        catalysts.append("Market recognition of undervaluation")
    if pillars.get("growth", 0) > 70:
        catalysts.append("Continued growth execution")
    if pillars.get("momentum", 0) > 60:
        catalysts.append("Technical breakout potential")
    
    catalysts.append("Positive earnings surprise")
    catalysts.append("Sector rotation into value/growth")
    
    return catalysts[:4]


def _calculate_conviction(data: Dict) -> str:
    """Calculate conviction level."""
    composite = data.get("composite_score", 0)
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    if composite >= 80 and mos >= 30:
        return "High"
    elif composite >= 70 and mos >= 15:
        return "Medium-High"
    elif composite >= 60:
        return "Medium"
    else:
        return "Low"


def _generate_bull_case_summary(data: Dict) -> str:
    """Generate bull case summary."""
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    name = data.get("name", data.get("ticker", "This company"))
    
    if mos > 25:
        return f"{name} is significantly undervalued and could rerate to fair value, providing substantial upside."
    elif mos > 10:
        return f"{name} trades at an attractive discount with room for multiple expansion."
    else:
        return f"{name} could benefit from continued operational improvement and market recognition."


def _generate_bull_case_points(data: Dict) -> list:
    """Generate bull case points."""
    points = []
    pillars = data.get("pillar_scores", {})
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    if mos > 15:
        points.append(f"Undervalued by {mos:.0f}% - significant upside to fair value")
    if pillars.get("quality", 0) > 65:
        points.append("Strong fundamentals support long-term value creation")
    if pillars.get("growth", 0) > 65:
        points.append("Growth trajectory could drive earnings expansion")
    if pillars.get("momentum", 0) > 60:
        points.append("Positive technical setup suggests institutional accumulation")
    
    points.append("Market may underappreciate competitive advantages")
    
    return points[:4]


def _generate_bear_case_summary(data: Dict) -> str:
    """Generate bear case summary."""
    pillars = data.get("pillar_scores", {})
    
    weak_pillars = [p for p, s in pillars.items() if isinstance(s, (int, float)) and s < 50]
    
    if weak_pillars:
        return f"Concerns around {', '.join(weak_pillars)} metrics could limit upside or increase downside risk."
    return "Macro headwinds or sector rotation could pressure the stock despite fundamentals."


def _generate_bear_case_points(data: Dict) -> list:
    """Generate bear case points."""
    points = []
    pillars = data.get("pillar_scores", {})
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    if pillars.get("safety", 100) < 60:
        points.append("Higher volatility could lead to larger drawdowns")
    if pillars.get("quality", 100) < 60:
        points.append("Business quality concerns may limit multiple expansion")
    if mos < 0:
        points.append("Currently trading above fair value estimate")
    
    points.append("Broader market correction could impact valuations")
    points.append("Competitive pressures may erode margins")
    
    return points[:4]


def _estimate_expected_return(data: Dict) -> str:
    """Estimate expected return range."""
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    if mos > 30:
        return "20-40% over base case"
    elif mos > 15:
        return "10-25% over base case"
    elif mos > 0:
        return "5-15% over base case"
    else:
        return "Limited upside at current levels"


def _generate_action_items(data: Dict) -> list:
    """Generate action items for investors."""
    classification = data.get("classification", "Watch")
    mos = data.get("margin_of_safety_pct", data.get("margin_of_safety", 0))
    
    items = []
    
    if classification == "Buy":
        items.append("Consider initiating or adding to position")
        items.append("Set target price based on fair value estimate")
        items.append("Determine position size based on risk tolerance")
    elif classification == "Hold":
        items.append("Maintain current position if held")
        items.append("Wait for better entry point if not owned")
        items.append("Monitor for changes in fundamentals")
    else:
        items.append("Add to watchlist for future consideration")
        items.append("Monitor for valuation improvement")
        items.append("Research competitive position and moat")
    
    items.append("Review quarterly earnings for execution")
    
    return items
