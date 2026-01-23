"""
FastAPI Application - Emetix Stock Screener API

Run with:
    uvicorn src.api.app:app --reload --port 8000

Then access:
    http://localhost:8000/docs - Swagger UI documentation
    http://localhost:8000/api/v2/watchlist - Enhanced top 10 with LSTM-DCF
    http://localhost:8000/api/screener/watchlist - Basic screener
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.screener import router as screener_router
from src.api.routes.risk_profile import router as risk_profile_router
from src.api.routes.storage import router as storage_router
from src.api.routes.pipeline import router as pipeline_router
from src.api.routes.analysis import router as analysis_router
from src.api.routes.multiagent_analysis import router as multiagent_router

# Create FastAPI app
app = FastAPI(
    title="Emetix Stock Screener API",
    description="""
    AI-powered stock screening with 4-Pillar Quality Scoring & Personal Risk Capacity.
    
    ## Phase 3: Quality Growth Pipeline (NEW)
    
    ### 3-Stage Automated Screening
    - **Stage 1**: Attention Triggers (52W Drop, Quality Growth Gate, Deep Value)
    - **Stage 2**: 4-Pillar Scoring (Value/Quality/Growth/Safety, 0-100 each)
    - **Stage 3**: Classification (Buy/Hold/Watch based on MoS + Score)
    
    ### Core Features
    - **LSTM-DCF Fair Value**: ML-powered fair value estimation using deep learning
    - **4-Pillar Scoring**: Value, Quality, Growth, Safety (25% each)
    - **Quality Growth Gate**: 4-path qualification (ROIC + Revenue Growth)
    - **Personal Risk Capacity**: Match stock risk to YOUR investor profile
    - **Buy/Hold/Watch**: Actionable classification based on MoS thresholds
    - **Momentum Check**: 50MA/200MA accumulation zones
    
    ## API Endpoints
    
    ### Pipeline (Phase 3) - `/api/pipeline/`
    - `GET /attention` - Stocks that triggered entry signals
    - `GET /qualified` - Quality-filtered stocks with pillar scores
    - `GET /classified` - Buy/Hold/Watch lists
    - `GET /stock/{ticker}` - Single stock analysis
    - `POST /trigger-scan` - Trigger pipeline scan
    
    ### Stock Analysis - `/api/screener/`
    - `GET /stock/{ticker}` - Detailed stock analysis
    - `GET /charts/{ticker}` - Price + MA overlays
    - `GET /compare` - Side-by-side comparison
    - `GET /methodology` - Scoring explanation
    
    ### Personal Risk Capacity - `/api/risk-profile/`
    - `POST /assess` - Risk questionnaire
    - `POST /position-sizing` - Position sizing calculation
    - `GET /methodology` - Framework documentation
    """,
    version="3.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(screener_router, prefix="/api", tags=["Stock Screener"])
app.include_router(risk_profile_router, tags=["Personal Risk Capacity"])
app.include_router(storage_router, prefix="/api", tags=["MongoDB Storage"])
app.include_router(pipeline_router, prefix="/api", tags=["Quality Pipeline"])
app.include_router(analysis_router, prefix="/api", tags=["AI Analysis"])
app.include_router(multiagent_router, prefix="/api", tags=["Multi-Agent Analysis"])


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Emetix Stock Screener API",
        "version": "3.5.0",
        "status": "healthy",
        "phase": "Phase 3 - Quality Growth Pipeline",
        "docs": "/docs",
        "features": [
            "3-Stage Automated Screening Pipeline",
            "4-Pillar Quality Scoring (Value/Quality/Growth/Safety)",
            "Quality Growth Gate (4-path qualification)",
            "Buy/Hold/Watch Classification",
            "LSTM-DCF Fair Value Estimation",
            "Personal Risk Capacity",
            "Momentum Check (50MA/200MA)"
        ],
        "endpoints": {
            "pipeline": {
                "attention": "GET /api/pipeline/attention",
                "qualified": "GET /api/pipeline/qualified",
                "classified": "GET /api/pipeline/classified",
                "stock": "GET /api/pipeline/stock/{ticker}",
                "trigger_scan": "POST /api/pipeline/trigger-scan",
                "summary": "GET /api/pipeline/summary",
                "description": "Phase 3 Quality Screening Pipeline (RECOMMENDED)"
            },
            "multiagent": {
                "full_analysis": "GET /api/multiagent/stock/{ticker}",
                "sentiment_only": "GET /api/multiagent/stock/{ticker}/sentiment",
                "fundamentals_only": "GET /api/multiagent/stock/{ticker}/fundamentals",
                "ml_valuation": "GET /api/multiagent/stock/{ticker}/ml-valuation",
                "watchlist_analyze": "POST /api/multiagent/watchlist/analyze",
                "description": "Multi-Agent AI Analysis (Sentiment, Fundamentals, ML)"
            },
            "stock_analysis": {
                "single": "GET /api/screener/stock/{ticker}",
                "compare": "GET /api/screener/compare?tickers=AAPL,MSFT,GOOGL",
                "charts": "GET /api/screener/charts/{ticker}",
                "description": "Individual stock analysis"
            },
            "risk_profile": {
                "assess": "POST /api/risk-profile/assess",
                "position_sizing": "POST /api/risk-profile/position-sizing",
                "methodology": "GET /api/risk-profile/methodology",
                "description": "Personal Risk Capacity framework"
            },
            "deprecated": {
                "note": "Legacy endpoints marked deprecated in Swagger UI",
                "endpoints": [
                    "/api/screener/watchlist → /api/pipeline/qualified",
                    "/api/screener/watchlist/categorized → /api/pipeline/classified",
                    "/api/screener/scan → /api/pipeline/trigger-scan"
                ]
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
