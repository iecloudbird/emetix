# 5. API Reference

> **FastAPI Endpoints for Frontend Integration**

---

## 🌐 Base Configuration

### Server

```
Base URL: http://localhost:8000
Documentation: http://localhost:8000/docs (Swagger UI)
```

### Headers

```http
Content-Type: application/json
```

### CORS

Currently configured for development (`allow_origins=["*"]`). Configure appropriately for production.

---

## 📋 Endpoint Summary

### Active Endpoints (Phase 3)

| Category           | Endpoint                                      | Method | Description                     |
| ------------------ | --------------------------------------------- | ------ | ------------------------------- |
| **Pipeline**       | `/api/pipeline/attention`                     | GET    | Stage 1 attention stocks        |
|                    | `/api/pipeline/qualified`                     | GET    | Stage 2 qualified stocks        |
|                    | `/api/pipeline/classified`                    | GET    | Stage 3 buy/hold/watch lists    |
|                    | `/api/pipeline/curated`                       | GET    | Stage 3 curated watchlist       |
|                    | `/api/pipeline/stock/{ticker}`                | GET    | Single stock with pillar scores |
|                    | `/api/pipeline/trigger-scan`                  | POST   | Manual trigger scan (admin)     |
|                    | `/api/pipeline/scan-history`                  | GET    | Recent scan logs                |
| **Multi-Agent AI** | `/api/multiagent/stock/{ticker}`              | GET    | Full multi-agent synthesis      |
|                    | `/api/multiagent/stock/{ticker}/sentiment`    | GET    | News sentiment analysis only    |
|                    | `/api/multiagent/stock/{ticker}/fundamentals` | GET    | Fundamental diagnosis only      |
|                    | `/api/multiagent/stock/{ticker}/ml-valuation` | GET    | LSTM-DCF v2 valuation only      |
|                    | `/api/multiagent/watchlist/analyze`           | POST   | Batch analyze watchlist         |
| **AI Analysis**    | `/api/analysis/stock/{ticker}`                | GET    | AI-powered stock diagnosis      |
|                    | `/api/analysis/stock/{ticker}/quick`          | GET    | Quick analysis (no AI)          |
| **Stock Screener** | `/api/screener/stock/{ticker}`                | GET    | Single stock detail             |
|                    | `/api/screener/charts/{ticker}`               | GET    | Price + 50MA/200MA overlays     |
|                    | `/api/screener/compare`                       | GET    | Side-by-side comparison         |
|                    | `/api/screener/sectors`                       | GET    | All sector benchmarks           |
|                    | `/api/screener/sectors/{sector}`              | GET    | Sector detail with stocks       |
|                    | `/api/screener/summary`                       | GET    | Market overview                 |
|                    | `/api/screener/methodology`                   | GET    | Scoring + pillar methodology    |
| **Risk Profile**   | `/api/risk-profile/assess`                    | POST   | Submit risk questionnaire       |
|                    | `/api/risk-profile/position-sizing`           | POST   | Position sizing calc            |
|                    | `/api/risk-profile/methodology`               | GET    | Framework documentation         |
| **Health**         | `/health`                                     | GET    | Server health check             |

### Removed Endpoints (Phase 3 Cleanup - Jan 2025)

> **Note**: These endpoints have been REMOVED from the codebase. Use the pipeline equivalents.

| Removed Endpoint                      | Replacement                            | Status     |
| ------------------------------------- | -------------------------------------- | ---------- |
| `/api/screener/watchlist`             | `/api/pipeline/qualified`              | ❌ Removed |
| `/api/screener/watchlist/simple`      | `/api/pipeline/qualified`              | ❌ Removed |
| `/api/screener/watchlist/categorized` | `/api/pipeline/classified`             | ❌ Removed |
| `/api/screener/watchlist/for-profile` | `/api/pipeline/qualified?profile_id=X` | ❌ Removed |
| `/api/screener/universe`              | `/api/pipeline/summary`                | ❌ Removed |
| `/api/screener/scan`                  | `/api/pipeline/trigger-scan`           | ❌ Removed |

---

## 📖 Pipeline Endpoints (Phase 3)

### 1. Get Attention Stocks

**Endpoint**: `GET /api/pipeline/attention`

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trigger` | str | null | Filter by trigger type: "52w_drop", "quality_growth", "deep_value" |
| `status` | str | "active"| Status filter: "active", "graduated", "expired" |

**Response**:

```json
{
  "status": "success",
  "count": 245,
  "last_scan": "2026-01-08T02:45:00Z",
  "stocks": [
    {
      "ticker": "META",
      "company_name": "Meta Platforms Inc.",
      "sector": "Technology",
      "triggers": [
        { "type": "52w_drop", "triggered_at": "2026-01-05T00:00:00Z" },
        {
          "type": "quality_growth",
          "path": 2,
          "triggered_at": "2026-01-05T00:00:00Z"
        }
      ],
      "first_triggered": "2026-01-05T00:00:00Z",
      "status": "active"
    }
  ]
}
```

### 2. Get Qualified Stocks

**Endpoint**: `GET /api/pipeline/qualified`

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------------|-------|---------|-------------|
| `classification`| str | null | Filter: "buy", "hold", "watch" |
| `sector` | str | null | Filter by sector |
| `min_score` | int | 60 | Minimum composite score |
| `profile_id` | str | null | Apply personal risk capacity filter |

**Response**:

```json
{
  "status": "success",
  "count": 87,
  "last_updated": "2026-01-08T06:00:00Z",
  "profile_applied": true,
  "stocks": [
    {
      "ticker": "META",
      "company_name": "Meta Platforms Inc.",
      "sector": "Technology",
      "current_price": 380.0,
      "fair_value": 485.5,
      "lstm_fair_value": 490.0,
      "margin_of_safety": 27.8,
      "pillar_scores": {
        "value": 82,
        "quality": 75,
        "growth": 68,
        "safety": 71
      },
      "composite_score": 74,
      "classification": "buy",
      "triggers": ["52w_drop", "quality_growth:path2"],
      "management_signals": {
        "insider_buy_ratio": 2.3,
        "buyback_pct": 3.5,
        "dividend_streak": 0
      },
      "momentum": {
        "below_200ma": true,
        "above_50ma": true,
        "accumulation_zone": true
      },
      "suitability": "excellent",
      "position_sizing": {
        "max_position_pct": 8.5,
        "max_shares": 22
      }
    }
  ]
}
```

### 3. Get Classified Lists

**Endpoint**: `GET /api/pipeline/classified`

**Query Parameters**:
| Parameter | Type | Default | Description |
|-------------|------|---------|-------------|
| `profile_id`| str | null | Apply personal risk capacity filter |

**Response**:

```json
{
  "status": "success",
  "last_updated": "2026-01-08T06:00:00Z",
  "buy": [
    { "ticker": "META", "score": 74, "mos": 27.8, "triggers": ["52w_drop"] }
  ],
  "hold": [
    {
      "ticker": "MSFT",
      "score": 72,
      "mos": 5.2,
      "triggers": ["quality_growth:path1"]
    }
  ],
  "watch": [
    { "ticker": "NVDA", "score": 65, "mos": -8.5, "triggers": ["deep_value"] }
  ]
}
```

### 4. Get Curated Watchlist

**Endpoint**: `GET /api/pipeline/curated`

**Description**: Returns the Stage 3 curated watchlist with AI-generated justifications, conviction levels, and tie-breaker scores. This is the final output of the 3-stage pipeline, providing actionable insights for each stock.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-------------|------|---------|-------------|
| `category` | str | null | Filter: "Strong Buy", "Moderate Buy", "Hold" |

**Response**:

```json
{
  "status": "success",
  "total_count": 45,
  "last_updated": "2026-01-08T06:00:00Z",
  "categories": {
    "Strong Buy": 5,
    "Moderate Buy": 15,
    "Hold": 25
  },
  "stocks": [
    {
      "ticker": "META",
      "name": "Meta Platforms Inc.",
      "sector": "Technology",
      "category": "Strong Buy",
      "conviction": "High",
      "scores": {
        "consensus": 82,
        "lstm_dcf": 85,
        "garp": 78,
        "risk": 80,
        "valuation": 75
      },
      "fair_value": 485.5,
      "current_price": 380.0,
      "margin_of_safety": 27.8,
      "beta": 1.15,
      "risk_level": "Medium",
      "justification": "Strong fundamentals with solid growth trajectory. LSTM-DCF model indicates significant undervaluation. Quality metrics support investment thesis.",
      "tie_breaker_score": 0.85
    }
  ],
  "metadata": {
    "pipeline_stage": 3,
    "scoring_weights": {
      "lstm_dcf": 0.5,
      "garp": 0.25,
      "risk": 0.25
    },
    "category_thresholds": {
      "strong_buy": ">= 75 consensus + >= 20% MoS",
      "moderate_buy": ">= 65 consensus OR >= 15% MoS",
      "hold": "< 65 consensus AND < 15% MoS"
    }
  }
}
```

**Frontend Usage**:

```typescript
import { fetchCuratedWatchlist } from "@/lib/api";

// Get all curated stocks
const all = await fetchCuratedWatchlist();

// Get only Strong Buy stocks
const strongBuy = await fetchCuratedWatchlist("Strong Buy");
```

---

## 🤖 Multi-Agent Analysis Endpoints (NEW - Jan 2025)

The Multi-Agent AI system provides comprehensive stock analysis through coordinated AI agents.

### 1. Full Multi-Agent Synthesis

**Endpoint**: `GET /api/multiagent/stock/{ticker}`

**Description**: Runs all 4 analysis agents (Sentiment, Fundamentals, ML Valuation, Synthesis) and returns a comprehensive analysis.

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | string | Stock ticker symbol (e.g., AAPL) |

**Response**:

```json
{
  "status": "success",
  "ticker": "AAPL",
  "timestamp": "2025-01-15T10:00:00Z",
  "synthesis": "**Overall Assessment**: Apple represents a high-quality company...",
  "sentiment_analysis": "**Market Sentiment**: Recent news coverage...",
  "fundamental_diagnosis": "**Financial Health**: Strong balance sheet...",
  "ml_valuation": "**LSTM-DCF v2 Valuation**: Fair value estimated at $195.50...",
  "processing_time_seconds": 8.5
}
```

### 2. Sentiment Analysis Only

**Endpoint**: `GET /api/multiagent/stock/{ticker}/sentiment`

**Description**: News sentiment analysis using Finnhub news API with AI-powered interpretation.

### 3. Fundamental Diagnosis Only

**Endpoint**: `GET /api/multiagent/stock/{ticker}/fundamentals`

**Description**: AI diagnosis of fundamental metrics (P/E, P/B, ROE, debt, margins).

### 4. ML Valuation Only

**Endpoint**: `GET /api/multiagent/stock/{ticker}/ml-valuation`

**Description**: LSTM-DCF v2 fair value estimation with quarterly fundamental features.

### 5. Batch Watchlist Analysis

**Endpoint**: `POST /api/multiagent/watchlist/analyze`

**Request Body**:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "analysis_type": "quick"
}
```

**Response**:

```json
{
  "status": "success",
  "results": [
    { "ticker": "AAPL", "synthesis": "..." },
    { "ticker": "MSFT", "synthesis": "..." }
  ],
  "processing_time_seconds": 25.2
}
```

---

## 📖 Stock Screener Endpoints (Active)

### 1. Get Stock Analysis

**Endpoint**: `GET /api/screener/stock/{ticker}`

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | string | Stock ticker symbol (e.g., AAPL) |

**Response**:

```json
{
  "status": "success",
  "data": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology",
    "current_price": 175.5,
    "fair_value": 210.0,
    "lstm_fair_value": 215.0,
    "traditional_fair_value": 200.0,
    "margin_of_safety": 19.66,
    "pe_ratio": 28.5,
    "pb_ratio": 45.2,
    "roe": 147.0,
    "debt_equity": 1.52,
    "beta": 1.15,
    "market_cap": 2750000000000,
    "dividend_yield": 0.5,
    "volatility": 25.3,
    "valuation_score": 78.5,
    "assessment": "Undervalued",
    "recommendation": "BUY",
    "risk_level": "MEDIUM",
    "justification": "..."
  },
  "sector_benchmarks": {
    "avg_pe": 28.0,
    "avg_pb": 6.0,
    "avg_roe": 18.0,
    "avg_margin": 20.0
  }
}
```

---

### 4. Compare Stocks

**Endpoint**: `GET /api/screener/compare`

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tickers` | string | Comma-separated tickers (max 10) |

**Example**: `/api/screener/compare?tickers=AAPL,MSFT,GOOGL`

**Response**:

```json
{
  "status": "success",
  "comparison": [
    {
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "sector": "Technology",
      "price": 175.5,
      "fair_value": 210.0,
      "margin_of_safety": 19.66,
      "score": 78.5,
      "recommendation": "BUY",
      "pe_ratio": 28.5,
      "sector_pe": 28.0,
      "pb_ratio": 45.2,
      "roe": 147.0,
      "debt_equity": 1.52,
      "fcf_yield": 3.5,
      "dividend_yield": 0.5
    },
    {
      "ticker": "MSFT",
      "company": "Microsoft Corporation",
      "...": "..."
    }
  ]
}
```

---

### 5. Get Chart Data

**Endpoint**: `GET /api/screener/charts/{ticker}`

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | string | Stock ticker symbol |

**Response**:

```json
{
  "status": "success",
  "data": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "charts": {
      "price_1y": [
        { "date": "2024-12-31", "price": 192.5, "volume": 45000000 },
        { "date": "2025-01-02", "price": 195.25, "volume": 52000000 }
      ],
      "price_5y": [
        { "date": "2020-12-31", "price": 132.5 },
        { "date": "2021-01-04", "price": 129.25 }
      ],
      "technical": [
        { "date": "2025-12-31", "price": 175.5, "ma50": 172.3, "ma200": 168.5 }
      ]
    },
    "metrics": {
      "current_price": 175.5,
      "high_52w": 199.62,
      "low_52w": 164.08,
      "pe_ratio": 28.5,
      "market_cap": 2750000000000
    },
    "fundamental_chart": {
      "pe_history": null,
      "revenue_growth": 0.08,
      "earnings_growth": 0.12
    }
  }
}
```

**Chart Data Structure**:

- `price_1y`: Daily OHLC + volume (1 year)
- `price_5y`: Weekly prices (5 years)
- `technical`: Daily prices with MA50, MA200 overlays

---

### 6. Get Sector Benchmarks

**Endpoint**: `GET /api/screener/sectors`

**Response**:

```json
{
  "status": "success",
  "description": "Industry sector averages for valuation benchmarking",
  "metadata": {
    "use_dynamic": true,
    "dynamic_count": 8,
    "total_sectors": 12,
    "last_updated": "2025-12-31T10:00:00.000Z"
  },
  "benchmarks": {
    "Technology": {
      "avg_pe": 28.0,
      "avg_pb": 6.0,
      "avg_roe": 18.0,
      "avg_margin": 20.0,
      "sample_size": 25,
      "source": "dynamic"
    },
    "Healthcare": {
      "avg_pe": 22.0,
      "avg_pb": 4.0,
      "avg_roe": 15.0,
      "avg_margin": 15.0,
      "sample_size": 18,
      "source": "dynamic"
    }
  }
}
```

---

### 7. Get Stocks by Sector

**Endpoint**: `GET /api/screener/sectors/{sector}`

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `sector` | string | Sector name (URL encoded) |

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 10 | Number of stocks |

**Example**: `/api/screener/sectors/Technology?n=5`

---

### 8. Get Methodology

**Endpoint**: `GET /api/screener/methodology`

**Response**:

```json
{
  "status": "success",
  "methodology": {
    "overview": "Hybrid LSTM-DCF + Traditional fundamental screening",
    "fair_value_models": {
      "lstm_dcf": {
        "description": "Deep learning model predicting FCF growth rate",
        "formula": "FV = FCF × (1 + predicted_growth) / (WACC - terminal_growth)",
        "inputs": "12-16 features including price, volume, fundamentals",
        "training": "Trained on 10+ years of S&P 500 data"
      },
      "traditional_dcf": {
        "description": "Sector-adjusted P/E and P/B multiple valuation",
        "formula": "FV = 0.6 × (EPS × SectorPE) + 0.3 × (BookValue × SectorPB) + 0.1 × CurrentPrice"
      }
    },
    "valuation_score": {
      "pe_score": "20% weight - P/E ratio vs sector average",
      "pb_score": "15% weight - P/B ratio vs sector average",
      "margin_of_safety": "20% weight - Upside to fair value",
      "fcf_yield": "15% weight - Free cash flow yield",
      "financial_health": "15% weight - Debt/equity, current ratio",
      "profitability": "15% weight - ROE vs sector average"
    },
    "recommendations": {
      "STRONG BUY": "Score ≥ 80 AND Margin of Safety > 20%",
      "BUY": "Score ≥ 70 OR (Score ≥ 65 AND Analyst Upside > 20%)",
      "ACCUMULATE": "Score 60-69",
      "HOLD": "Score 50-59",
      "REDUCE": "Score 40-49",
      "SELL": "Score < 40"
    }
  }
}
```

---

### 9. Health Check

**Endpoint**: `GET /health`

**Response**:

```json
{
  "status": "healthy"
}
```

---

## 🎯 Personal Risk Capacity Endpoints (Phase 2)

### 1. Submit Risk Questionnaire

**Endpoint**: `POST /api/risk-profile/assess`

**Request Body**:

```json
{
  "experience_level": "beginner",
  "investment_horizon": "medium",
  "emergency_fund_months": 6,
  "monthly_investment_percent": 20,
  "max_tolerable_loss_percent": 15,
  "panic_sell_response": "hold_wait",
  "volatility_comfort": 3,
  "portfolio_value": 50000,
  "monthly_income": 5000
}
```

**Response**:

```json
{
  "profile_id": "abc123",
  "risk_capacity": 72.5,
  "risk_tolerance": 60.0,
  "emotional_buffer": 1.75,
  "adjusted_mos_threshold": 35.0,
  "suitable_beta_range": { "min": 0.5, "max": 1.5 },
  "overall_profile": "moderate",
  "recommendations": [
    "Focus on established companies with stable earnings",
    "Consider stocks with MoS > 35%"
  ]
}
```

---

### 2. Get Position Sizing

**Endpoint**: `POST /api/risk-profile/position-sizing`

**Request Body**:

```json
{
  "profile_id": "abc123",
  "ticker": "AAPL",
  "current_price": 175.5,
  "margin_of_safety": 20.0,
  "beta": 1.2
}
```

**Response**:

```json
{
  "ticker": "AAPL",
  "max_position_percent": 8.5,
  "max_position_value": 4250.0,
  "max_shares": 24,
  "risk_factors": ["Beta above comfort range", "MoS below threshold"],
  "recommendation": "Consider smaller position due to risk factors",
  "methodology": "Kelly-inspired with emotional buffer adjustment"
}
```

---

### 3. Get Suitable Stocks

**Endpoint**: `GET /api/risk-profile/suitable-stocks`

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile_id` | str | required | Profile ID from assessment |
| `n` | int | 10 | Number of stocks |

**Response**:

```json
{
  "profile_id": "abc123",
  "suitable_count": 8,
  "total_screened": 20,
  "filters_applied": {
    "beta_range": { "min": 0.5, "max": 1.5 },
    "min_mos": 35.0
  },
  "stocks": [
    {
      "ticker": "JNJ",
      "suitability": "excellent",
      "beta": 0.6,
      "margin_of_safety": 42.0,
      "position_sizing": {
        "max_position_percent": 10.0,
        "max_shares": 15
      }
    }
  ]
}
```

---

### 4. Risk Profile Methodology

**Endpoint**: `GET /api/risk-profile/methodology`

Returns documentation of the Personal Risk Capacity Framework including:

- Risk Capacity calculation formula
- Risk Tolerance scoring
- Emotional Buffer factors by experience level
- Position sizing methodology

---

## ⚠️ Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

| Code | Description                      |
| ---- | -------------------------------- |
| 200  | Success                          |
| 400  | Bad Request (invalid parameters) |
| 404  | Not Found (ticker not found)     |
| 500  | Internal Server Error            |

---

## 🚀 Quick Start

### Start Server

```powershell
cd c:\Users\Hans8899\Desktop\fyp\emetix
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Test Endpoints

```powershell
# Health check
curl http://localhost:8000/health

# Get top 5 undervalued stocks
curl "http://localhost:8000/api/screener/watchlist?n=5"

# Get AAPL analysis
curl http://localhost:8000/api/screener/stock/AAPL

# Compare stocks
curl "http://localhost:8000/api/screener/compare?tickers=AAPL,MSFT,GOOGL"
```

---

_Next: [6. Frontend Integration Guide](./06_FRONTEND_GUIDE.md)_
