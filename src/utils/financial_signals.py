"""
Financial Signals — Shared utility for reusable stock signals.

Canonical source for:
- calculate_revenue_cagr()  : Multi-year revenue CAGR from yfinance
- compute_200wma_signal()   : Price vs 200-week moving average
- compute_contrarian_signals(): Sentiment-suppressed + fundamentally strong
"""
from typing import Optional, Dict
from config.logging_config import get_logger

logger = get_logger(__name__)


def calculate_revenue_cagr(ticker: str, years: int = 3) -> Optional[Dict]:
    """
    Calculate multi-year revenue CAGR from yfinance annual financials.

    Returns:
        Dict with cagr, years, latest_revenue, oldest_revenue — or None.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        if income_stmt is None or income_stmt.empty:
            return None

        revenue_row = None
        for row_name in ["Total Revenue", "Revenue", "Operating Revenue"]:
            if row_name in income_stmt.index:
                revenue_row = income_stmt.loc[row_name]
                break

        if revenue_row is None:
            return None

        revenues = revenue_row.dropna().sort_index()
        if len(revenues) < 2:
            return None

        latest_revenue = revenues.iloc[-1]
        target_idx = max(0, len(revenues) - years - 1)
        oldest_revenue = revenues.iloc[target_idx]
        actual_years = len(revenues) - 1 - target_idx

        if actual_years < 2 or oldest_revenue <= 0 or latest_revenue <= 0:
            return None

        cagr = ((latest_revenue / oldest_revenue) ** (1 / actual_years) - 1) * 100

        return {
            "cagr": round(cagr, 1),
            "years": actual_years,
            "latest_revenue": float(latest_revenue),
            "oldest_revenue": float(oldest_revenue),
        }
    except Exception as e:
        logger.debug(f"{ticker}: CAGR calculation failed: {e}")
        return None


def compute_200wma_signal(ticker: str) -> Optional[Dict]:
    """
    Compute price relative to the 200-week moving average.

    A stock trading below its 200WMA is considered a potential contrarian
    accumulation zone — historically associated with long-term value entry
    points for quality companies.

    Returns:
        Dict with current_price, wma_200, ratio, below_200wma flag — or None.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        # Need ~4+ years of weekly data for a 200-week MA
        hist = stock.history(period="5y", interval="1wk")

        if hist is None or hist.empty or len(hist) < 200:
            return None

        wma_200 = float(hist["Close"].rolling(window=200).mean().iloc[-1])
        current_price = float(hist["Close"].iloc[-1])

        if wma_200 <= 0:
            return None

        ratio = current_price / wma_200

        return {
            "current_price": round(current_price, 2),
            "wma_200": round(wma_200, 2),
            "price_to_200wma": round(ratio, 3),
            "below_200wma": ratio < 1.0,
            "pct_from_200wma": round((ratio - 1.0) * 100, 1),
        }
    except Exception as e:
        logger.debug(f"{ticker}: 200WMA calculation failed: {e}")
        return None


def compute_contrarian_signals(
    ticker: str,
    sentiment_score: Optional[float] = None,
    quality_score: Optional[float] = None,
) -> Dict:
    """
    Unified contrarian signal detector.

    Combines:
    - Sentiment suppression (score < 0.4)
    - Strong fundamentals  (quality score > 0.6)
    - Price below 200WMA   (optional extra confirmation)

    This is the single source of truth for contrarian detection across
    multi-agent analysis, watchlist scoring, and pipeline triggers.

    Returns:
        Dict with is_contrarian, signals list, and optional 200wma data.
    """
    signals = []
    result: Dict = {
        "ticker": ticker,
        "is_contrarian": False,
        "signals": [],
        "wma_200": None,
    }

    # Sentiment check
    if sentiment_score is not None and sentiment_score < 0.4:
        signals.append({
            "type": "sentiment_suppressed",
            "value": sentiment_score,
            "description": "Market sentiment is bearish (< 0.4)",
        })

    # Quality check
    if quality_score is not None and quality_score > 0.6:
        signals.append({
            "type": "fundamentals_strong",
            "value": quality_score,
            "description": "Fundamental quality is strong (> 0.6)",
        })

    # 200WMA check (additive signal, not required)
    wma_data = compute_200wma_signal(ticker)
    if wma_data:
        result["wma_200"] = wma_data
        if wma_data["below_200wma"]:
            signals.append({
                "type": "below_200wma",
                "value": wma_data["price_to_200wma"],
                "description": (
                    f"Price {wma_data['pct_from_200wma']:.1f}% "
                    f"below 200-week MA (${wma_data['wma_200']})"
                ),
            })

    result["signals"] = signals
    # Contrarian = at least sentiment suppressed + fundamentally strong
    sentiment_hit = any(s["type"] == "sentiment_suppressed" for s in signals)
    quality_hit = any(s["type"] == "fundamentals_strong" for s in signals)
    result["is_contrarian"] = sentiment_hit and quality_hit

    return result
