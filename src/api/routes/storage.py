"""
MongoDB Storage API Routes

Provides endpoints for:
- Watchlist storage and retrieval
- Educational content
- Strategy templates
- Session-based data (no auth required)
"""
from fastapi import APIRouter, HTTPException, Query, Cookie
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import uuid

router = APIRouter(prefix="/storage", tags=["Storage"])


# =========================================================================
# PYDANTIC MODELS
# =========================================================================

class WatchlistCreate(BaseModel):
    name: str
    tickers: List[str]
    is_public: bool = False
    metadata: Optional[dict] = None


class WatchlistResponse(BaseModel):
    id: str
    name: str
    tickers: List[str]
    ticker_count: int
    is_public: bool
    created_at: datetime
    metadata: dict = {}


class StrategyCreate(BaseModel):
    name: str
    description: str
    criteria: dict
    risk_level: str  # low, moderate, aggressive
    metadata: Optional[dict] = None


class EducationContent(BaseModel):
    id: str
    title: str
    category: str
    content: str
    order: int
    metadata: dict = {}


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_mongo_client():
    """Get MongoDB client, raise error if unavailable"""
    try:
        from src.data.mongo_client import mongo_client, is_mongo_available
        if not is_mongo_available():
            raise HTTPException(
                status_code=503,
                detail="MongoDB not configured. Set MONGODB_URI in environment."
            )
        return mongo_client
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB client not available. Install pymongo."
        )


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    if session_id:
        return session_id
    # Generate a new session ID (stored client-side)
    return str(uuid.uuid4())


# =========================================================================
# WATCHLIST ENDPOINTS
# =========================================================================

@router.post("/watchlists", response_model=WatchlistResponse)
async def create_watchlist(
    watchlist: WatchlistCreate,
    session_id: Optional[str] = Cookie(default=None)
):
    """
    Create a new watchlist.
    
    Session ID is stored in cookies for tracking ownership.
    No user auth required - session-based ownership.
    """
    client = get_mongo_client()
    session = get_or_create_session(session_id)
    
    result_id = client.save_watchlist(
        name=watchlist.name,
        tickers=watchlist.tickers,
        session_id=session,
        is_public=watchlist.is_public,
        metadata=watchlist.metadata
    )
    
    if not result_id:
        raise HTTPException(status_code=500, detail="Failed to save watchlist")
    
    saved = client.get_watchlist(result_id)
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to retrieve saved watchlist")
    
    return WatchlistResponse(
        id=saved["_id"],
        name=saved["name"],
        tickers=saved["tickers"],
        ticker_count=len(saved["tickers"]),
        is_public=saved["is_public"],
        created_at=saved["created_at"],
        metadata=saved.get("metadata", {})
    )


@router.get("/watchlists/{watchlist_id}")
async def get_watchlist(watchlist_id: str):
    """Get a specific watchlist by ID"""
    client = get_mongo_client()
    
    watchlist = client.get_watchlist(watchlist_id)
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    return watchlist


@router.get("/watchlists")
async def list_watchlists(
    public: bool = Query(default=True, description="List public watchlists"),
    session_id: Optional[str] = Cookie(default=None),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    List watchlists.
    
    - public=true: Get public watchlists
    - public=false + session_id cookie: Get session's private watchlists
    """
    client = get_mongo_client()
    
    if public:
        watchlists = client.get_public_watchlists(limit=limit)
    elif session_id:
        watchlists = client.get_session_watchlists(session_id)
    else:
        return {"watchlists": [], "message": "No session ID for private watchlists"}
    
    return {"watchlists": watchlists, "count": len(watchlists)}


@router.delete("/watchlists/{watchlist_id}")
async def delete_watchlist(
    watchlist_id: str,
    session_id: Optional[str] = Cookie(default=None)
):
    """Delete a watchlist (session must own it)"""
    if not session_id:
        raise HTTPException(status_code=401, detail="No session ID")
    
    client = get_mongo_client()
    success = client.delete_watchlist(watchlist_id, session_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Watchlist not found or you don't own it"
        )
    
    return {"deleted": True}


# =========================================================================
# EDUCATION ENDPOINTS
# =========================================================================

@router.get("/education")
async def list_education_content(
    category: Optional[str] = Query(
        default=None,
        description="Filter by category: fundamentals, valuation, risk, technical, strategies"
    ),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Get educational content"""
    client = get_mongo_client()
    content = client.get_education_content(category=category, limit=limit)
    
    return {
        "content": content,
        "count": len(content),
        "categories": ["fundamentals", "valuation", "risk", "technical", "strategies"]
    }


# =========================================================================
# STRATEGY ENDPOINTS
# =========================================================================

@router.get("/strategies")
async def list_strategies():
    """Get investment strategy templates"""
    client = get_mongo_client()
    strategies = client.get_strategies()
    
    # If no strategies in DB, return defaults
    if not strategies:
        strategies = [
            {
                "name": "Conservative Income",
                "description": "Focus on dividend-paying, low-volatility stocks",
                "criteria": {"max_beta": 0.8, "min_dividend_yield": 2.0, "max_pe": 20},
                "risk_level": "low"
            },
            {
                "name": "Balanced Growth",
                "description": "Mix of growth and value with moderate risk",
                "criteria": {"max_beta": 1.2, "min_roe": 12, "max_pe": 30},
                "risk_level": "moderate"
            },
            {
                "name": "Aggressive Growth",
                "description": "High-growth stocks with higher volatility tolerance",
                "criteria": {"min_growth": 20, "max_pe": 50},
                "risk_level": "aggressive"
            }
        ]
    
    return {"strategies": strategies, "count": len(strategies)}


@router.post("/strategies")
async def create_strategy(strategy: StrategyCreate):
    """Create a new strategy template"""
    client = get_mongo_client()
    
    result_id = client.save_strategy(
        name=strategy.name,
        description=strategy.description,
        criteria=strategy.criteria,
        risk_level=strategy.risk_level,
        metadata=strategy.metadata
    )
    
    if not result_id:
        raise HTTPException(status_code=500, detail="Failed to save strategy")
    
    return {"id": result_id, "created": True}


# =========================================================================
# HEALTH CHECK
# =========================================================================

@router.get("/health")
async def storage_health():
    """Check MongoDB connection status"""
    try:
        from src.data.mongo_client import is_mongo_available, MONGODB_URI
        
        available = is_mongo_available()
        
        return {
            "mongodb": "connected" if available else "disconnected",
            "configured": bool(MONGODB_URI),
            "note": "Set MONGODB_URI environment variable for full storage features"
        }
    except Exception as e:
        return {
            "mongodb": "error",
            "error": str(e)
        }
