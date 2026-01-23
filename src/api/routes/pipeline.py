"""
Pipeline API Routes - Phase 3 Quality Screening Pipeline

Endpoints for the 3-stage screening pipeline:
- Stage 1: Attention stocks (weekly scan)
- Stage 2: Qualified stocks with pillar scores
- Stage 3: Classified stocks (Buy/Hold/Watch)
"""
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from typing import Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

router = APIRouter(prefix="/pipeline", tags=["Quality Pipeline"])

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def get_pipeline_db():
    """Lazy import pipeline DB client"""
    from src.data.pipeline_db import pipeline_db, is_pipeline_available
    if not is_pipeline_available():
        raise HTTPException(
            status_code=503,
            detail="Pipeline database not available. Check MONGODB_URI configuration."
        )
    return pipeline_db


def get_stock_screener():
    """Lazy import stock screener"""
    from src.analysis.stock_screener import StockScreener
    return StockScreener(enable_lstm=True)


def get_pillar_scorer():
    """Lazy import pillar scorer"""
    from src.analysis.pillar_scorer import PillarScorer
    return PillarScorer()


# =============================================================================
# ATTENTION STOCKS (Stage 1)
# =============================================================================

@router.get("/attention")
async def get_attention_stocks(
    status: str = Query(default="active", description="Filter by status: active, graduated, expired, or all"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum stocks to return")
):
    """
    Get stocks in the attention list (Stage 1).
    
    These are stocks that have triggered one of:
    - 52-Week Drop (beaten down quality)
    - Quality Growth Gate (4-path qualification)
    - Deep Value (high MoS or FCF yield)
    """
    try:
        db = get_pipeline_db()
        
        status_filter = None if status == "all" else status
        stocks = db.get_attention_stocks(status=status_filter, limit=limit)
        
        # Convert ObjectId to string for JSON serialization
        for stock in stocks:
            stock["_id"] = str(stock["_id"])
            if "first_triggered" in stock:
                stock["first_triggered"] = stock["first_triggered"].isoformat()
            if "last_updated" in stock:
                stock["last_updated"] = stock["last_updated"].isoformat()
            for trigger in stock.get("triggers", []):
                if "triggered_at" in trigger:
                    trigger["triggered_at"] = trigger["triggered_at"].isoformat()
        
        return {
            "status": "success",
            "count": len(stocks),
            "attention_stocks": stocks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting attention stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# QUALIFIED STOCKS (Stage 2)
# =============================================================================

@router.get("/qualified")
async def get_qualified_stocks(
    classification: Optional[str] = Query(default=None, description="Filter: buy, hold, watch"),
    min_score: float = Query(default=60, ge=0, le=100, description="Minimum composite score"),
    sector: Optional[str] = Query(default=None, description="Filter by sector"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum stocks to return"),
    sort_by: str = Query(default="composite_score", description="Sort field")
):
    """
    Get qualified stocks with pillar scores (Stage 2).
    
    These stocks have passed the attention triggers AND scored >= 60 composite.
    Includes full 4-pillar breakdown and classification.
    """
    try:
        db = get_pipeline_db()
        
        stocks = db.get_qualified_stocks(
            classification=classification,
            min_score=min_score,
            sector=sector,
            limit=limit,
            sort_by=sort_by
        )
        
        # Clean for JSON
        for stock in stocks:
            stock["_id"] = str(stock["_id"])
            if "last_updated" in stock:
                stock["last_updated"] = stock["last_updated"].isoformat()
        
        # Get classification counts
        counts = db.get_classified_counts()
        
        return {
            "status": "success",
            "count": len(stocks),
            "classification_counts": counts,
            "qualified_stocks": stocks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting qualified stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classified")
async def get_classified_stocks(
    sector: Optional[str] = Query(default=None, description="Filter by sector"),
    limit_per_class: int = Query(default=20, ge=1, le=50, description="Max stocks per classification")
):
    """
    Get stocks organized by Buy/Hold/Watch classification (Stage 3).
    
    Convenience endpoint that groups qualified stocks by classification.
    """
    try:
        db = get_pipeline_db()
        
        result = {
            "buy": [],
            "hold": [],
            "watch": []
        }
        
        for classification in ["buy", "hold", "watch"]:
            stocks = db.get_qualified_stocks(
                classification=classification,
                sector=sector,
                limit=limit_per_class
            )
            
            # Transform for frontend compatibility
            for stock in stocks:
                stock["_id"] = str(stock["_id"])
                if "last_updated" in stock:
                    stock["last_updated"] = stock["last_updated"].isoformat()
                    
                # Map field names for frontend
                stock["company_name"] = stock.get("name", stock.get("company_name", ""))
                stock["current_price"] = stock.get("price", stock.get("current_price", 0))
                mos_pct = stock.get("margin_of_safety_pct", stock.get("margin_of_safety", 0)) or 0
                stock["margin_of_safety"] = mos_pct
                
                # Calculate fair_value if not present
                # Fair Value = Price / (1 - MoS/100), e.g., $100 with 25% MoS = $133.33 fair value
                price = stock.get("price", stock.get("current_price", 0)) or 0
                if stock.get("fair_value"):
                    pass  # Already has fair value
                elif stock.get("lstm_fair_value"):
                    stock["fair_value"] = stock["lstm_fair_value"]
                elif price > 0 and mos_pct != 0:
                    # Derive fair value from margin of safety
                    # MoS = (FV - Price) / FV * 100 => FV = Price / (1 - MoS/100)
                    stock["fair_value"] = price / (1 - mos_pct / 100) if mos_pct < 100 else price * 2
                    stock["lstm_fair_value"] = stock["fair_value"]  # Use as LSTM estimate
                else:
                    stock["fair_value"] = 0
                
                # Normalize classification to lowercase
                stock["classification"] = stock.get("classification", "").lower()
                
                # Transform pillar_scores from flat dict to nested structure
                pillar_scores_raw = stock.get("pillar_scores", {})
                if pillar_scores_raw and isinstance(list(pillar_scores_raw.values())[0] if pillar_scores_raw else 1, (int, float)):
                    # It's a flat dict {value: 80, quality: 70, ...}
                    stock["pillar_scores"] = {
                        k: {"score": v, "weight": 0.25, "weighted_score": v * 0.25, "components": {}}
                        for k, v in pillar_scores_raw.items()
                    }
            
            result[classification] = stocks
        
        counts = db.get_classified_counts()
        # Normalize keys to lowercase for frontend consistency
        counts_normalized = {k.lower(): v for k, v in counts.items()}
        counts_normalized["total"] = sum(counts_normalized.values())
        
        return {
            "status": "success",
            "counts": counts_normalized,
            "classified": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting classified stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/curated")
async def get_curated_watchlist(
    category: Optional[str] = Query(
        default=None, 
        description="Filter by category: strong_buy, moderate_buy, hold, growth_watch, value_watch, etc."
    )
):
    """
    Get the curated watchlist from Stage 3.
    
    Returns the full curated watchlist with:
    - Strong Buy (highest conviction, MoS >= 35%, Score >= 75)
    - Moderate Buy (good conviction, MoS >= 25%, Score >= 70)
    - Hold (quality monitoring)
    - Watch (by sub-category: growth_watch, value_watch, etc.)
    
    Each stock includes:
    - Analyst-like justifications (short + long)
    - Conviction level
    - Tie-breaker scores for ranking
    """
    try:
        db = get_pipeline_db()
        
        if category:
            # Return single category
            stocks = db.get_curated_by_category(category, limit=50)
            return {
                "status": "success",
                "category": category,
                "count": len(stocks),
                "stocks": stocks
            }
        
        # Return full curated watchlist
        curated = db.get_curated_watchlist()
        
        if not curated:
            # Fallback to raw classified if curated not available
            logger.warning("Curated watchlist not found, run Stage 3 script")
            raise HTTPException(
                status_code=404,
                detail="Curated watchlist not available. Run: python scripts/pipeline/stage3_curate_watchlist.py"
            )
        
        return {
            "status": "success",
            **curated
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting curated watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SINGLE STOCK LOOKUP
# =============================================================================

@router.get("/stock/{ticker}")
async def get_pipeline_stock(ticker: str):
    """
    Get detailed pipeline data for a single stock.
    
    Returns pillar scores, triggers, classification, and momentum data.
    Falls back to real-time calculation if not in database.
    """
    try:
        ticker = ticker.upper()
        db = get_pipeline_db()
        
        # Check qualified stocks first
        stock = db.get_qualified_by_ticker(ticker)
        
        if stock:
            stock["_id"] = str(stock["_id"])
            if "last_updated" in stock:
                stock["last_updated"] = stock["last_updated"].isoformat()
            
            return {
                "status": "success",
                "source": "database",
                "stock": stock
            }
        
        # Not in DB - calculate real-time
        loop = asyncio.get_event_loop()
        
        def calculate_realtime():
            screener = get_stock_screener()
            scorer = get_pillar_scorer()
            
            data = screener._fetch_stock_data(ticker)
            if not data:
                return None
            
            scoring = scorer.calculate_composite(data)
            analysis = scorer.get_strength_weakness_analysis(scoring)
            
            # Get trigger evaluation (v2.2 - pass ticker for CAGR)
            from src.analysis.quality_growth_gate import AttentionTriggers
            triggers = AttentionTriggers.evaluate_all_triggers(data, ticker=ticker)
            
            return {
                "ticker": ticker,
                "company_name": data.get("company_name"),
                "sector": data.get("sector"),
                "pillar_scores": scoring["pillars"],
                "composite_score": scoring["composite_score"],
                "classification": scoring["classification"],
                "qualified": scoring["qualified"],
                "current_price": data.get("current_price"),
                "fair_value": data.get("fair_value"),
                "lstm_fair_value": data.get("lstm_fair_value"),
                "margin_of_safety": data.get("margin_of_safety"),
                "triggers": triggers,
                "momentum": {
                    "price_vs_200ma": data.get("price_vs_200ma"),
                    "price_vs_50ma": data.get("price_vs_50ma"),
                    "sma_50": data.get("sma_50"),
                    "sma_200": data.get("sma_200"),
                },
                "analysis": analysis
            }
        
        result = await loop.run_in_executor(_executor, calculate_realtime)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
        
        return {
            "status": "success",
            "source": "realtime",
            "stock": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline stock {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCAN OPERATIONS
# =============================================================================

@router.post("/trigger-scan")
async def trigger_scan(
    background_tasks: BackgroundTasks,
    scan_type: str = Query(default="attention", description="Scan type: attention or qualified"),
    tickers: Optional[str] = Query(default=None, description="Comma-separated tickers (test mode)"),
    limit: int = Query(default=None, ge=10, le=500, description="Limit stocks (test mode)")
):
    """
    Trigger a manual pipeline scan.
    
    Runs in background. Check /scan-history for status.
    For production, use scheduled tasks instead.
    """
    try:
        ticker_list = None
        if tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(",")]
        
        if scan_type == "attention":
            def run_attention_scan():
                from scripts.pipeline.weekly_attention_scan import WeeklyAttentionScanner
                scanner = WeeklyAttentionScanner()
                return scanner.run_scan(
                    tickers=ticker_list,
                    limit=limit,
                    save_to_db=True
                )
            
            background_tasks.add_task(run_attention_scan)
            message = f"Attention scan started"
            
        elif scan_type == "qualified":
            def run_qualified_update():
                from scripts.pipeline.daily_qualified_update import DailyQualifiedUpdater
                updater = DailyQualifiedUpdater()
                return updater.run_update(save_to_db=True)
            
            background_tasks.add_task(run_qualified_update)
            message = "Qualified update started"
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid scan_type. Use 'attention' or 'qualified'."
            )
        
        return {
            "status": "started",
            "message": message,
            "scan_type": scan_type,
            "tickers": ticker_list,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan-history")
async def get_scan_history(
    limit: int = Query(default=10, ge=1, le=50, description="Number of recent scans")
):
    """
    Get recent scan job history.
    
    Shows status, duration, and counts for recent scans.
    """
    try:
        db = get_pipeline_db()
        scans = db.get_recent_scans(limit=limit)
        
        for scan in scans:
            scan["_id"] = str(scan["_id"])
            if "started_at" in scan:
                scan["started_at"] = scan["started_at"].isoformat()
            if "completed_at" in scan:
                scan["completed_at"] = scan["completed_at"].isoformat()
        
        return {
            "status": "success",
            "count": len(scans),
            "scans": scans
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scan history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SUMMARY & STATS
# =============================================================================

@router.get("/summary")
async def get_pipeline_summary():
    """
    Get overall pipeline statistics.
    
    Returns counts for each stage and classification.
    """
    try:
        db = get_pipeline_db()
        summary = db.get_pipeline_summary()
        
        # Clean last_scan for JSON
        if summary.get("last_scan"):
            scan = summary["last_scan"]
            scan["_id"] = str(scan["_id"])
            if "started_at" in scan:
                scan["started_at"] = scan["started_at"].isoformat()
            if "completed_at" in scan:
                scan["completed_at"] = scan["completed_at"].isoformat()
        
        return {
            "status": "success",
            "pipeline": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
