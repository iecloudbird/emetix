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
import math

router = APIRouter(prefix="/multiagent", tags=["Multi-Agent Analysis"])

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=3)

# =============================================================================
# FLOAT SANITISER (prevents JSON serialisation errors from NaN / inf)
# =============================================================================

def _sanitize_floats(obj):
    """Recursively replace NaN / inf float values with None so JSON serialisation succeeds."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


# =============================================================================
# ANALYSIS CACHE (reduces Gemini calls — TTL 8 hours)
# =============================================================================
CACHE_TTL_HOURS = 8


def _get_cache_db():
    """Get pipeline DB for analysis caching."""
    try:
        from src.data.pipeline_db import PipelineDBClient
        client = PipelineDBClient()
        if client.connect():
            return client.db
    except Exception as e:
        logger.warning(f"Cache DB unavailable: {e}")
    return None


def _get_cached_analysis(ticker: str, analysis_type: str) -> Optional[Dict]:
    """Check the analysis_cache collection for a fresh cached result."""
    db = _get_cache_db()
    if db is None:
        return None
    try:
        cached = db.analysis_cache.find_one(
            {"ticker": ticker, "type": analysis_type}
        )
        if cached:
            generated_at = cached.get("generated_at")
            if generated_at:
                age_hours = (
                    datetime.now(timezone.utc) - generated_at
                ).total_seconds() / 3600
                if age_hours < CACHE_TTL_HOURS:
                    logger.info(
                        f"[{ticker}] Cache hit for {analysis_type} "
                        f"(age: {age_hours:.1f}h)"
                    )
                    result = cached.get("result", {})
                    result["status"] = "cached"
                    result.pop("_id", None)
                    return result
    except Exception as e:
        logger.warning(f"Cache read failed for {ticker}: {e}")
    return None


def _save_to_cache(ticker: str, analysis_type: str, result: Dict):
    """Persist an analysis result into MongoDB analysis_cache."""
    db = _get_cache_db()
    if db is None:
        return
    try:
        cache_doc = {
            "ticker": ticker,
            "type": analysis_type,
            "result": result,
            "generated_at": datetime.now(timezone.utc),
        }
        db.analysis_cache.update_one(
            {"ticker": ticker, "type": analysis_type},
            {"$set": cache_doc},
            upsert=True,
        )
        logger.info(f"[{ticker}] Cached {analysis_type} result")
    except Exception as e:
        logger.warning(f"Cache write failed for {ticker}: {e}")


# =============================================================================
# REAL SCORE COMPUTATION (data-driven, NOT from LLM text)
# =============================================================================

def _compute_sentiment_data(ticker: str) -> dict:
    """
    Compute sentiment data (0-1 scores) from actual market data.
    Returns score + component breakdown for data-driven narrative.
    Combines: news sentiment (40%), analyst ratings (30%), price momentum (30%).
    """
    scores = []
    weights = []
    details = {
        "news_score": None, "num_articles": 0,
        "analyst_score": None, "buy_count": 0, "hold_count": 0, "sell_count": 0,
        "momentum_score": None, "month_return": None,
    }
    
    # --- 1. News Sentiment (40% weight) ---
    try:
        from src.data.fetchers.news_sentiment_fetcher import NewsSentimentFetcher
        fetcher = NewsSentimentFetcher()
        news_result = fetcher.fetch_all_news(ticker)
        news_score = news_result.get("sentiment_score", 0.5)
        scores.append(news_score)
        weights.append(0.4)
        details["news_score"] = round(news_score, 2)
        details["num_articles"] = news_result.get("total_articles", 0)
        details["sentiment_label"] = news_result.get("sentiment_label", "NEUTRAL")
        details["positive_count"] = news_result.get("positive_count", 0)
        details["negative_count"] = news_result.get("negative_count", 0)
        details["neutral_count"] = news_result.get("neutral_count", 0)
        details["company_name"] = news_result.get("company_name", ticker)
        # Capture top 5 articles for headlines display
        raw_articles = news_result.get("articles", [])[:5]
        details["top_articles"] = [
            {
                "title": a.get("title", ""),
                "source": a.get("source") or a.get("publisher", ""),
                "sentiment": a.get("sentiment", "NEUTRAL"),
                "publish_time": a.get("publish_time", ""),
            }
            for a in raw_articles if a.get("title")
        ]
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
            details["analyst_score"] = round(analyst_score, 2)
            details["buy_count"] = buy_count
            details["hold_count"] = hold_count
            details["sell_count"] = sell_count
            # Capture recent analyst actions (last 3) for narrative
            recent_actions = []
            for _, row in recent.tail(3).iterrows():
                firm = row.get("Firm", "")
                grade = row.get("To Grade", "")
                action = row.get("Action", "")
                if firm and grade:
                    recent_actions.append({"firm": str(firm), "grade": str(grade), "action": str(action)})
            details["recent_analyst_actions"] = recent_actions
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
            details["momentum_score"] = round(momentum_score, 2)
            details["month_return"] = round(month_return, 4)
            logger.info(f"[{ticker}] Momentum score: {momentum_score:.2f} (1mo return: {month_return:.1%})")
    except Exception as e:
        logger.warning(f"[{ticker}] Momentum calc failed: {e}")
    
    # --- Weighted average ---
    if scores and weights:
        total_weight = sum(weights)
        final_score = round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 2)
    else:
        final_score = 0.5
    
    return {"score": final_score, **details}


def _compute_fundamentals_data(ticker: str) -> dict:
    """
    Compute real quality/growth/value scores (0-1) from actual financial data.
    Returns scores + raw metrics for data-driven narrative.
    """
    import yfinance as yf
    
    quality_score = 0.5
    growth_score = 0.5
    value_score = 0.5
    metrics = {}
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # --- Context fields for enriched narrative ---
        metrics["company_name"] = info.get("longName", ticker)
        metrics["sector"] = info.get("sector", "")
        metrics["industry"] = info.get("industry", "")
        market_cap = info.get("marketCap")
        if market_cap is not None:
            metrics["market_cap"] = market_cap
        fwd_pe = info.get("forwardPE")
        if fwd_pe is not None and fwd_pe > 0:
            metrics["forward_pe"] = round(fwd_pe, 1)
        beta = info.get("beta")
        if beta is not None:
            metrics["beta"] = round(beta, 2)
        div_yield = info.get("dividendYield")
        if div_yield is not None:
            metrics["dividend_yield"] = round(div_yield * 100, 2)
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        curr_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if high_52 is not None:
            metrics["fifty_two_week_high"] = round(high_52, 2)
        if low_52 is not None:
            metrics["fifty_two_week_low"] = round(low_52, 2)
        if curr_price is not None:
            metrics["current_price"] = round(curr_price, 2)
        
        # --- Quality Score (profitability + financial health) ---
        quality_components = []
        
        roe = info.get('returnOnEquity')
        if roe is not None:
            metrics["roe"] = round(roe * 100, 1)
            roe_s = min(1.0, max(0.0, roe * 100 / 25.0))
            quality_components.append(roe_s)
        
        margin = info.get('profitMargins')
        if margin is not None:
            metrics["profit_margin"] = round(margin * 100, 1)
            margin_s = min(1.0, max(0.0, margin * 100 / 25.0))
            quality_components.append(margin_s)
        
        de = info.get('debtToEquity')
        if de is not None:
            de_ratio = de / 100
            metrics["debt_to_equity"] = round(de_ratio, 2)
            de_s = max(0.0, min(1.0, 1.0 - de_ratio / 3.0))
            quality_components.append(de_s)
        
        cr = info.get('currentRatio')
        if cr is not None:
            metrics["current_ratio"] = round(cr, 2)
            cr_s = min(1.0, max(0.0, cr / 2.5))
            quality_components.append(cr_s)
        
        if quality_components:
            quality_score = round(sum(quality_components) / len(quality_components), 2)
        
        # --- Growth Score (revenue + earnings growth) ---
        growth_components = []
        
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None:
            metrics["revenue_growth"] = round(rev_growth * 100, 1)
            rg_s = min(1.0, max(0.0, (rev_growth * 100 + 5) / 30.0))
            growth_components.append(rg_s)
        
        earn_growth = info.get('earningsGrowth')
        if earn_growth is not None:
            metrics["earnings_growth"] = round(earn_growth * 100, 1)
            eg_s = min(1.0, max(0.0, (earn_growth * 100 + 10) / 40.0))
            growth_components.append(eg_s)
        
        rev_q = info.get('revenueQuarterlyGrowth')
        if rev_q is not None:
            metrics["rev_quarterly_growth"] = round(rev_q * 100, 1)
            rq_s = min(1.0, max(0.0, (rev_q * 100 + 5) / 30.0))
            growth_components.append(rq_s)
        
        if growth_components:
            growth_score = round(sum(growth_components) / len(growth_components), 2)
        
        # --- Value Score (valuation attractiveness) ---
        value_components = []
        
        pe = info.get('trailingPE')
        if pe is not None and pe > 0:
            metrics["pe"] = round(pe, 1)
            pe_s = max(0.0, min(1.0, 1.0 - (pe - 10) / 30.0))
            value_components.append(pe_s)
        
        peg = info.get('pegRatio')
        if peg is not None and peg > 0:
            metrics["peg"] = round(peg, 2)
            peg_s = max(0.0, min(1.0, 1.0 - (peg - 0.5) / 2.0))
            value_components.append(peg_s)
        
        pb = info.get('priceToBook')
        if pb is not None and pb > 0:
            metrics["pb"] = round(pb, 2)
            pb_s = max(0.0, min(1.0, 1.0 - (pb - 1) / 8.0))
            value_components.append(pb_s)
        
        ev_ebitda = info.get('enterpriseToEbitda')
        if ev_ebitda is not None and ev_ebitda > 0:
            metrics["ev_ebitda"] = round(ev_ebitda, 1)
            eve_s = max(0.0, min(1.0, 1.0 - (ev_ebitda - 5) / 20.0))
            value_components.append(eve_s)
        
        if value_components:
            value_score = round(sum(value_components) / len(value_components), 2)
        
        logger.info(f"[{ticker}] Fundamentals scores: quality={quality_score}, growth={growth_score}, value={value_score}")
        
    except Exception as e:
        logger.error(f"[{ticker}] Failed to compute fundamentals scores: {e}")
    
    return {
        "quality_score": quality_score,
        "growth_score": growth_score,
        "value_score": value_score,
        "metrics": metrics,
    }


# =============================================================================
# DATA-DRIVEN NARRATIVE BUILDERS (replace LLM agent narratives)
# =============================================================================

def _build_sentiment_narrative(ticker: str, data: dict) -> str:
    """Build a data-driven sentiment narrative from computed data."""
    score = data["score"]
    label = _score_to_label(score)
    company = data.get("company_name", ticker)
    parts = [f"## Market Sentiment for {company} ({ticker})\n", f"**Overall Score: {score:.0%} ({label})**\n"]

    # --- Top Headlines ---
    articles = data.get("top_articles", [])
    if articles:
        parts.append("### Top Headlines")
        sentiment_icons = {"POSITIVE": "+", "NEGATIVE": "-", "NEUTRAL": "~"}
        for a in articles:
            icon = sentiment_icons.get(a["sentiment"], "~")
            source = f" — *{a['source']}*" if a.get("source") else ""
            parts.append(f"- [{icon}] {a['title']}{source}")
        parts.append("")

    # --- Score Breakdown ---
    parts.append("### Score Breakdown")

    if data.get("news_score") is not None:
        ns = data["news_score"]
        n_lbl = "positive" if ns > 0.6 else "negative" if ns < 0.4 else "mixed"
        art = data.get("num_articles", 0)
        pos = data.get("positive_count", 0)
        neg = data.get("negative_count", 0)
        neu = data.get("neutral_count", 0)
        breakdown = f" ({pos} positive, {neg} negative, {neu} neutral)" if art > 0 else ""
        parts.append(f"- **News Sentiment** (40% weight): {ns:.0%} ({n_lbl}) — {art} articles analyzed{breakdown}")

    if data.get("analyst_score") is not None:
        b, h, s = data["buy_count"], data["hold_count"], data["sell_count"]
        parts.append(f"- **Analyst Ratings** (30% weight): {data['analyst_score']:.0%} — {b} Buy, {h} Hold, {s} Sell")
        # Show recent analyst actions
        actions = data.get("recent_analyst_actions", [])
        for act in actions:
            action_str = f" ({act['action']})" if act.get("action") else ""
            parts.append(f"  - {act['firm']}: **{act['grade']}**{action_str}")

    if data.get("momentum_score") is not None:
        mr = data.get("month_return", 0)
        mom_lbl = "strong uptrend" if mr > 0.05 else "mild uptrend" if mr > 0 else "mild downtrend" if mr > -0.05 else "strong downtrend"
        parts.append(f"- **Price Momentum** (30% weight): {data['momentum_score']:.0%} ({mr:+.1%} 1-month return — {mom_lbl})")

    # --- Key Takeaways ---
    parts.append("\n### Key Takeaways")
    takeaways = []

    # News distribution insight
    pos = data.get("positive_count", 0)
    neg = data.get("negative_count", 0)
    total_art = pos + neg + data.get("neutral_count", 0)
    if total_art > 0:
        if pos > neg * 2:
            takeaways.append(f"News coverage is overwhelmingly positive ({pos}/{total_art} articles bullish)")
        elif neg > pos * 2:
            takeaways.append(f"News coverage is predominantly negative ({neg}/{total_art} articles bearish) — monitor for catalysts")
        elif pos > neg:
            takeaways.append(f"Slightly positive news tilt ({pos} positive vs {neg} negative out of {total_art} articles)")
        elif neg > pos:
            takeaways.append(f"Slightly negative news tilt ({neg} negative vs {pos} positive out of {total_art} articles)")
        else:
            takeaways.append(f"Balanced news sentiment across {total_art} articles — market is undecided")

    # Analyst consensus insight
    b, h, s_cnt = data.get("buy_count", 0), data.get("hold_count", 0), data.get("sell_count", 0)
    total_analysts = b + h + s_cnt
    if total_analysts > 0:
        if b > total_analysts * 0.6:
            takeaways.append(f"Strong analyst consensus: {b}/{total_analysts} recommend Buy or equivalent")
        elif s_cnt > total_analysts * 0.3:
            takeaways.append(f"Analyst caution: {s_cnt}/{total_analysts} have Sell-equivalent ratings")
        else:
            takeaways.append(f"Analyst opinions are mixed with no clear consensus ({b}B/{h}H/{s_cnt}S)")

    # Momentum divergence insight
    mr = data.get("month_return")
    ns = data.get("news_score")
    if mr is not None and ns is not None:
        if mr > 0.05 and ns < 0.4:
            takeaways.append("**Divergence**: Price rising despite negative news — could indicate institutional buying or priced-in negativity")
        elif mr < -0.05 and ns > 0.6:
            takeaways.append("**Divergence**: Price falling despite positive news — watch for delayed reaction or broader market pressure")

    for t in takeaways:
        parts.append(f"- {t}")

    # Overall assessment
    if score >= 0.7:
        parts.append("\n**Outlook**: Strong bullish signals across multiple indicators suggest favorable market conditions.")
    elif score >= 0.55:
        parts.append("\n**Outlook**: Moderately positive signals indicate cautious optimism in the market.")
    elif score >= 0.45:
        parts.append("\n**Outlook**: Mixed signals suggest uncertainty — consider other factors before acting.")
    elif score >= 0.3:
        parts.append("\n**Outlook**: Bearish signals dominate — potential risk or contrarian opportunity if fundamentals are strong.")
    else:
        parts.append("\n**Outlook**: Strongly negative sentiment — high caution warranted.")

    return "\n".join(parts)


def _build_fundamentals_narrative(ticker: str, data: dict) -> str:
    """Build a data-driven fundamentals narrative from computed data."""
    q = data["quality_score"]
    g = data["growth_score"]
    v = data["value_score"]
    m = data.get("metrics", {})

    company = m.get("company_name", ticker)
    parts = [f"## Fundamental Analysis for {company} ({ticker})\n"]

    # --- Sector Context ---
    sector = m.get("sector", "")
    industry = m.get("industry", "")
    market_cap = m.get("market_cap")
    if sector or industry or market_cap:
        context_bits = []
        if sector and industry:
            context_bits.append(f"**{industry}** ({sector})")
        elif sector:
            context_bits.append(f"**{sector}**")
        if market_cap:
            if market_cap >= 200e9:
                cap_tier = "Mega-Cap"
            elif market_cap >= 10e9:
                cap_tier = "Large-Cap"
            elif market_cap >= 2e9:
                cap_tier = "Mid-Cap"
            elif market_cap >= 300e6:
                cap_tier = "Small-Cap"
            else:
                cap_tier = "Micro-Cap"
            cap_str = f"${market_cap / 1e9:.1f}B" if market_cap >= 1e9 else f"${market_cap / 1e6:.0f}M"
            context_bits.append(f"{cap_tier} ({cap_str})")
        if m.get("beta") is not None:
            beta = m["beta"]
            beta_lbl = "low volatility" if beta < 0.8 else "market-average volatility" if beta <= 1.2 else "high volatility"
            context_bits.append(f"Beta {beta:.2f} ({beta_lbl})")
        if m.get("dividend_yield") is not None and m["dividend_yield"] > 0:
            context_bits.append(f"Dividend {m['dividend_yield']:.2f}%")
        parts.append(f"*{' · '.join(context_bits)}*\n")

    # Quality
    q_label = "Strong" if q >= 0.7 else "Average" if q >= 0.4 else "Weak"
    parts.append(f"### Quality: {q:.0%} ({q_label})")
    q_details = []
    if "roe" in m:
        roe_note = "excellent" if m["roe"] >= 20 else "good" if m["roe"] >= 12 else "below average"
        q_details.append(f"ROE {m['roe']:.1f}% (*{roe_note}; benchmark >15%*)")
    if "profit_margin" in m:
        mg_note = "strong" if m["profit_margin"] >= 20 else "healthy" if m["profit_margin"] >= 10 else "thin"
        q_details.append(f"Profit Margin {m['profit_margin']:.1f}% (*{mg_note}*)")
    if "debt_to_equity" in m:
        de_note = "conservative" if m["debt_to_equity"] <= 0.5 else "moderate" if m["debt_to_equity"] <= 1.5 else "high leverage"
        q_details.append(f"D/E {m['debt_to_equity']:.2f} (*{de_note}; <1.0 preferred*)")
    if "current_ratio" in m:
        cr_note = "strong liquidity" if m["current_ratio"] >= 2.0 else "adequate" if m["current_ratio"] >= 1.0 else "liquidity risk"
        q_details.append(f"Current Ratio {m['current_ratio']:.2f} (*{cr_note}; >1.5 healthy*)")
    for d in q_details:
        parts.append(f"- {d}")

    # Growth
    g_label = "High Growth" if g >= 0.7 else "Moderate" if g >= 0.4 else "Low"
    parts.append(f"\n### Growth: {g:.0%} ({g_label})")
    g_details = []
    if "revenue_growth" in m:
        rg = m["revenue_growth"]
        rg_note = "accelerating" if rg >= 15 else "steady" if rg >= 5 else "slowing" if rg >= 0 else "declining"
        g_details.append(f"Revenue Growth {rg:+.1f}% (*{rg_note}*)")
    if "earnings_growth" in m:
        eg = m["earnings_growth"]
        eg_note = "strong expansion" if eg >= 20 else "solid" if eg >= 5 else "compressing" if eg >= 0 else "declining"
        g_details.append(f"Earnings Growth {eg:+.1f}% (*{eg_note}*)")
    if "rev_quarterly_growth" in m:
        rq = m["rev_quarterly_growth"]
        g_details.append(f"Quarterly Revenue Growth {rq:+.1f}%")
    for d in g_details:
        parts.append(f"- {d}")

    # Value
    v_label = "Undervalued" if v >= 0.7 else "Fair" if v >= 0.4 else "Expensive"
    parts.append(f"\n### Value: {v:.0%} ({v_label})")
    v_details = []
    if "pe" in m:
        pe = m["pe"]
        pe_note = "cheap" if pe < 15 else "reasonable" if pe < 25 else "premium" if pe < 40 else "very expensive"
        v_details.append(f"P/E {pe:.1f} (*{pe_note}; market avg ~20-22*)")
    if "forward_pe" in m and "pe" in m:
        fwd = m["forward_pe"]
        compression = "earnings expected to grow" if fwd < m["pe"] else "earnings expected to compress"
        v_details.append(f"Forward P/E {fwd:.1f} (*{compression}*)")
    elif "forward_pe" in m:
        v_details.append(f"Forward P/E {m['forward_pe']:.1f}")
    if "peg" in m:
        peg = m["peg"]
        peg_note = "undervalued for growth" if peg < 1.0 else "fair for growth" if peg < 2.0 else "overvalued relative to growth"
        v_details.append(f"PEG {peg:.2f} (*{peg_note}; <1.0 is attractive*)")
    if "pb" in m:
        pb = m["pb"]
        pb_note = "below book value" if pb < 1.0 else "reasonable" if pb < 3.0 else "premium to book"
        v_details.append(f"P/B {pb:.2f} (*{pb_note}*)")
    if "ev_ebitda" in m:
        eve = m["ev_ebitda"]
        eve_note = "cheap" if eve < 10 else "fair" if eve < 15 else "expensive"
        v_details.append(f"EV/EBITDA {eve:.1f} (*{eve_note}; <12 typically attractive*)")
    for d in v_details:
        parts.append(f"- {d}")

    # --- Key Turning Points to Watch ---
    watchpoints = []
    # Earnings vs revenue divergence
    if "earnings_growth" in m and "revenue_growth" in m:
        eg, rg = m["earnings_growth"], m["revenue_growth"]
        if eg < 0 and rg > 5:
            watchpoints.append("Earnings declining while revenue grows — margin compression, watch for cost management changes")
        elif eg > rg * 2 and eg > 10:
            watchpoints.append("Earnings growing much faster than revenue — margin expansion, but may not be sustainable long-term")
    # High debt with declining margins
    if "debt_to_equity" in m and "profit_margin" in m:
        if m["debt_to_equity"] > 1.5 and m["profit_margin"] < 10:
            watchpoints.append("High leverage with thin margins — monitor interest coverage and cash flow generation closely")
    # 52-week range positioning
    if "current_price" in m and "fifty_two_week_high" in m and "fifty_two_week_low" in m:
        cp = m["current_price"]
        high = m["fifty_two_week_high"]
        low = m["fifty_two_week_low"]
        range_pct = (cp - low) / (high - low) * 100 if high != low else 50
        if range_pct > 90:
            watchpoints.append(f"Trading near 52-week high (${cp:.2f} vs ${high:.2f}) — momentum is strong but watch for resistance")
        elif range_pct < 15:
            watchpoints.append(f"Trading near 52-week low (${cp:.2f} vs ${low:.2f}) — could be a value opportunity or a falling knife")
    # Extreme P/E
    if "pe" in m:
        if m["pe"] > 50:
            watchpoints.append(f"P/E of {m['pe']:.0f}x is very elevated — market expects exceptional growth; any earnings miss could trigger correction")
        elif m["pe"] < 8 and m.get("earnings_growth", 0) >= 0:
            watchpoints.append(f"P/E of {m['pe']:.0f}x is extremely low for a profitable company — potential value trap or hidden gem")
    # PEG signal
    if "peg" in m:
        if m["peg"] > 3.0:
            watchpoints.append(f"PEG of {m['peg']:.1f} suggests you're overpaying for the growth rate — compare with sector peers")

    if watchpoints:
        parts.append("\n### Key Turning Points to Watch")
        for wp in watchpoints:
            parts.append(f"- {wp}")

    # Overall assessment
    avg = (q + g + v) / 3
    if avg >= 0.65:
        parts.append("\n**Overall**: Fundamentals are **solid** across quality, growth, and value dimensions.")
    elif avg >= 0.45:
        parts.append("\n**Overall**: Fundamentals are **mixed** — some pillars are stronger than others.")
    else:
        parts.append("\n**Overall**: Fundamentals are **weak** — caution recommended.")

    return "\n".join(parts)


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
        # Check cache first — avoids Gemini calls for repeated requests
        cache_type = "multiagent_deep" if deep_analysis else "multiagent"
        cached = _get_cached_analysis(ticker, cache_type)
        if cached is not None:
            return cached

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
            
            # Fetch insider/institutional holdings data
            holdings = await _get_holdings_data(ticker)
            if holdings.get("available"):
                result["sections"]["holdings"] = holdings
            
            # Generate synthesis if we have multiple sections
            if len(result["agents_used"]) >= 2:
                synthesis = await _generate_synthesis(ticker, result["sections"], deep_mode=deep_synthesis)
                result["sections"]["synthesis"] = synthesis
        
        # Sanitise before caching / returning (NaN / inf → None)
        result = _sanitize_floats(result)

        # Cache the result for future requests
        _save_to_cache(ticker, cache_type, result)

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
        return _sanitize_floats({
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sentiment": result
        })
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
        return _sanitize_floats({
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fundamentals": result
        })
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
        return _sanitize_floats({
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ml_valuation": result
        })
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
        
        return _sanitize_floats(result)
        
    except Exception as e:
        logger.error(f"Watchlist analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _get_sentiment_analysis(ticker: str) -> Dict[str, Any]:
    """Data-driven sentiment analysis — NO LLM call. Uses computed scores + narrative."""
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _compute_sentiment_data(ticker))

        score = data["score"]
        label = _score_to_label(score)
        analysis = _build_sentiment_narrative(ticker, data)

        return {
            "available": True,
            "agent": "SentimentAnalyzerAgent",
            "analysis": analysis,
            "sentiment_score": score,
            "sentiment_label": label,
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error for {ticker}: {e}")
        return {"available": False, "error": str(e)}


async def _get_fundamentals_analysis(ticker: str) -> Dict[str, Any]:
    """Data-driven fundamentals analysis — NO LLM call. Uses computed scores + narrative."""
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _compute_fundamentals_data(ticker))

        analysis = _build_fundamentals_narrative(ticker, data)

        return {
            "available": True,
            "agent": "FundamentalsAnalyzerAgent",
            "analysis": analysis,
            "quality_score": data["quality_score"],
            "growth_score": data["growth_score"],
            "value_score": data["value_score"],
        }
    except Exception as e:
        logger.error(f"Fundamentals analysis error for {ticker}: {e}")
        return {"available": False, "error": str(e)}


async def _get_ml_valuation(ticker: str) -> Dict[str, Any]:
    """ML valuation via direct tool calls — NO LLM orchestration."""
    try:
        agent = get_enhanced_valuation_agent()
        if not agent:
            return {"available": False, "error": "EnhancedValuationAgent not available"}

        # Build tool lookup from the agent's registered tools
        tool_map = {t.name: t for t in agent.tools}
        loop = asyncio.get_event_loop()

        # Call LSTM-DCF and Consensus tools directly (no LLM decides)
        lstm_future = loop.run_in_executor(
            _executor, lambda: tool_map["LSTM_DCF_Valuation"].func(ticker)
        )
        consensus_future = loop.run_in_executor(
            _executor, lambda: tool_map["ConsensusValuation"].func(ticker)
        )
        lstm_text, consensus_text = await asyncio.gather(lstm_future, consensus_future)

        raw_text = f"{lstm_text}\n\n{consensus_text}"

        # Parse structured metrics from tool output
        import re
        fair_value = None
        consensus_score = None
        margin_of_safety = None

        fv_match = re.search(r'(?:Fair Value|Implied Fair Value)[:\s]*\$?([\d.]+)', raw_text, re.IGNORECASE)
        if fv_match:
            try:
                fair_value = float(fv_match.group(1))
            except ValueError:
                pass

        cs_match = re.search(r'Consensus Score[:\s]*([\d.]+)', raw_text, re.IGNORECASE)
        if cs_match:
            try:
                consensus_score = float(cs_match.group(1))
            except ValueError:
                pass

        gap_match = re.search(r'Valuation Gap[:\s]*([\-+]?[\d.]+)%', raw_text, re.IGNORECASE)
        if gap_match:
            try:
                margin_of_safety = float(gap_match.group(1))
            except ValueError:
                pass

        # Parse additional details for enriched narrative
        current_price = None
        cp_match = re.search(r'Current (?:Price|Market Price)[:\s]*\$?([\d.]+)', raw_text, re.IGNORECASE)
        if cp_match:
            try:
                current_price = float(cp_match.group(1))
            except ValueError:
                pass

        model_version = None
        ver_match = re.search(r'(?:Model Version|Version)[:\s]*(V?\d[\w.]*)', raw_text, re.IGNORECASE)
        if ver_match:
            model_version = ver_match.group(1)

        confidence_level = None
        conf_match = re.search(r'Confidence[:\s]*(\w+)', raw_text, re.IGNORECASE)
        if conf_match:
            confidence_level = conf_match.group(1)

        # Parse growth forecasts if present (V2 model)
        rev_growth = None
        fcf_growth = None
        rg_match = re.search(r'Revenue Growth[:\s]*([\-+]?[\d.]+)%', raw_text, re.IGNORECASE)
        if rg_match:
            try:
                rev_growth = float(rg_match.group(1))
            except ValueError:
                pass
        fg_match = re.search(r'FCF Growth[:\s]*([\-+]?[\d.]+)%', raw_text, re.IGNORECASE)
        if fg_match:
            try:
                fcf_growth = float(fg_match.group(1))
            except ValueError:
                pass

        analysis_text = _build_ml_narrative(
            ticker, raw_text, fair_value, consensus_score, margin_of_safety,
            current_price, model_version, confidence_level, rev_growth, fcf_growth
        )

        return {
            "available": True,
            "agent": "EnhancedValuationAgent",
            "analysis": analysis_text,
            "extracted_metrics": {
                "fair_value": fair_value,
                "consensus_score": consensus_score,
                "margin_of_safety": margin_of_safety,
            },
        }
    except Exception as e:
        logger.error(f"ML valuation agent error for {ticker}: {e}")
        return {
            "available": False,
            "error": str(e)
        }


def _build_ml_narrative(
    ticker: str, raw_text: str,
    fair_value, consensus_score, margin_of_safety,
    current_price, model_version, confidence_level,
    rev_growth, fcf_growth,
) -> str:
    """Build an enriched ML valuation narrative with methodology context."""
    parts = [f"## ML-Powered Valuation for {ticker}\n"]

    # --- Methodology Section ---
    parts.append("### Methodology")
    is_v2 = model_version and "2" in str(model_version)
    if is_v2:
        parts.append(
            "This valuation uses the **LSTM-DCF V2 model**, which analyzes quarterly fundamental data "
            "(revenue, free cash flow, margins) through a Long Short-Term Memory neural network to forecast "
            "10-year cash flows, then discounts them to present value. V2 incorporates fundamental-driven "
            "growth patterns rather than relying solely on price history."
        )
    else:
        parts.append(
            "This valuation uses the **LSTM-DCF model**, which combines machine learning (Long Short-Term Memory "
            "neural networks) with traditional Discounted Cash Flow analysis. The model learns temporal patterns "
            "from 60 periods of historical data across 12 features (price, volume, fundamentals, technicals) "
            "to forecast future cash flows."
        )
    parts.append(
        "\nThe **Consensus Score** blends multiple models: LSTM-DCF (50% weight), "
        "Traditional Valuation (40%), and Risk Assessment (10%) to provide a balanced view.\n"
    )

    # --- Valuation Results ---
    parts.append("### Valuation Results")
    if current_price is not None and fair_value is not None:
        parts.append(f"- **Current Price**: ${current_price:.2f}")
        parts.append(f"- **ML Fair Value**: ${fair_value:.2f}")
        if margin_of_safety is not None:
            if margin_of_safety > 0:
                parts.append(f"- **Valuation Gap**: {margin_of_safety:+.1f}% — stock appears **undervalued** by the model")
            elif margin_of_safety < -5:
                parts.append(f"- **Valuation Gap**: {margin_of_safety:+.1f}% — stock appears **overvalued** by the model")
            else:
                parts.append(f"- **Valuation Gap**: {margin_of_safety:+.1f}% — stock is trading near **fair value**")
    if consensus_score is not None:
        if consensus_score >= 0.7:
            cs_lbl = "Strong Buy signal"
        elif consensus_score >= 0.55:
            cs_lbl = "Moderate Buy signal"
        elif consensus_score >= 0.45:
            cs_lbl = "Hold / Neutral"
        elif consensus_score >= 0.3:
            cs_lbl = "Moderate Sell signal"
        else:
            cs_lbl = "Strong Sell signal"
        parts.append(f"- **Consensus Score**: {consensus_score:.2f} — {cs_lbl}")
    if confidence_level:
        parts.append(f"- **Model Confidence**: {confidence_level}")

    # --- Growth Assumptions ---
    if rev_growth is not None or fcf_growth is not None:
        parts.append("\n### Growth Assumptions Applied")
        if rev_growth is not None:
            rg_note = "above-average growth" if rev_growth > 10 else "moderate growth" if rev_growth > 3 else "low or declining growth"
            parts.append(f"- **Projected Revenue Growth**: {rev_growth:+.1f}% (*{rg_note}*)")
        if fcf_growth is not None:
            fg_note = "strong cash generation" if fcf_growth > 10 else "steady cash flows" if fcf_growth > 0 else "declining free cash flow"
            parts.append(f"- **Projected FCF Growth**: {fcf_growth:+.1f}% (*{fg_note}*)")
        parts.append("*These growth rates are ML-derived forecasts based on historical fundamental trends, not analyst estimates.*")

    # --- What This Means ---
    parts.append("\n### What This Means")
    if margin_of_safety is not None:
        gap = margin_of_safety
        if gap >= 30:
            parts.append(
                f"The model estimates significant upside potential ({gap:+.1f}%). This level of discount "
                "could indicate a strong buying opportunity — but verify with fundamental quality and "
                "check if there are negative catalysts the model may not capture (lawsuits, regulatory risk)."
            )
        elif gap >= 10:
            parts.append(
                f"The model sees moderate upside ({gap:+.1f}%). The stock may be trading below intrinsic value, "
                "suggesting a reasonable entry point for long-term investors. Consider position sizing based on conviction."
            )
        elif gap >= -5:
            parts.append(
                f"The stock is trading approximately at fair value ({gap:+.1f}%). "
                "No strong buy or sell signal from valuation alone. Look to other factors (growth trajectory, "
                "competitive position, sentiment) to make a decision."
            )
        elif gap >= -20:
            parts.append(
                f"The model indicates the stock is moderately overvalued ({gap:+.1f}%). "
                "This doesn't necessarily mean sell — high-quality growth companies often trade at premiums. "
                "Evaluate whether the growth rate justifies the premium."
            )
        else:
            parts.append(
                f"The model flags significant overvaluation ({gap:+.1f}%). "
                "Proceed with caution — either the market expects exceptional future growth the model doesn't "
                "capture, or there may be downside risk if expectations aren't met."
            )
    else:
        parts.append("Insufficient data to determine valuation gap. Review the raw model output below for details.")

    # --- Raw Model Output ---
    parts.append("\n### Detailed Model Output")
    parts.append(raw_text)

    return "\n".join(parts)


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
        
        # Detect contrarian signal using shared utility
        from src.utils.financial_signals import compute_contrarian_signals
        contrarian = compute_contrarian_signals(
            ticker,
            sentiment_score=sentiment_score,
            quality_score=quality_score,
        )
        is_contrarian = contrarian["is_contrarian"]
        contrarian_note = ""
        if is_contrarian:
            signal_descs = "; ".join(s["description"] for s in contrarian["signals"])
            contrarian_note = (
                f"\n\n⚠️ CONTRARIAN SIGNAL DETECTED: {signal_descs}. "
                "This could indicate an accumulation opportunity if the "
                "negative sentiment is overblown."
            )
        
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
            "contrarian_details": contrarian.get("signals", []),
            "wma_200": contrarian.get("wma_200"),
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


async def _get_holdings_data(ticker: str) -> Dict[str, Any]:
    """Fetch insider transactions and institutional holder data."""
    try:
        from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
        fetcher = YFinanceFetcher()

        loop = asyncio.get_event_loop()
        insider_future = loop.run_in_executor(
            None, lambda: fetcher.fetch_insider_transactions(ticker)
        )
        inst_future = loop.run_in_executor(
            None, lambda: fetcher.fetch_institutional_holders(ticker)
        )
        major_future = loop.run_in_executor(
            None, lambda: fetcher.fetch_major_holders(ticker)
        )

        insider, institutional, major = await asyncio.gather(
            insider_future, inst_future, major_future
        )

        return {
            "available": True,
            "insider_transactions": insider or {},
            "institutional_holders": institutional or {},
            "major_holders": major or {},
        }
    except Exception as e:
        logger.warning(f"Holdings data failed for {ticker}: {e}")
        return {"available": False, "error": str(e)}


@router.get("/stock/{ticker}/holdings")
async def get_holdings_only(ticker: str):
    """
    Get insider transactions, institutional holders, and major holder breakdown.
    Data sourced from Yahoo Finance.
    """
    ticker = ticker.upper()
    try:
        result = await _get_holdings_data(ticker)
        return _sanitize_floats({
            "status": "success" if result.get("available") else "partial",
            "ticker": ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "holdings": result,
        })
    except Exception as e:
        logger.error(f"Holdings data error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
