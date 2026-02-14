# 5. API Reference

---

## Base URL

| Environment | URL                                                       |
| ----------- | --------------------------------------------------------- |
| Local       | `http://localhost:8000`                                   |
| Production  | Render.com deployment URL (set via `NEXT_PUBLIC_API_URL`) |

**Common headers**: `Content-Type: application/json`

---

## Root Endpoints

| Method | Path      | Description                             |
| ------ | --------- | --------------------------------------- |
| GET    | `/`       | API information (name, version, status) |
| GET    | `/health` | Health check                            |

---

## 1. Screener Router (`/api`)

Stock lookup, charts, sector data, and agent-powered analysis.

| Method | Path                         | Description                                      |
| ------ | ---------------------------- | ------------------------------------------------ |
| GET    | `/api/stock/{ticker}`        | Full stock data (price, fundamentals, valuation) |
| GET    | `/api/charts/{ticker}`       | Historical price/volume chart data               |
| GET    | `/api/sectors`               | All sector summaries                             |
| GET    | `/api/sectors/{sector}`      | Stocks within a specific sector                  |
| GET    | `/api/compare`               | Compare multiple stocks (`?tickers=AAPL,MSFT`)   |
| GET    | `/api/summary`               | Market summary and overview                      |
| GET    | `/api/methodology`           | Scoring methodology explanation                  |
| GET    | `/api/education/{ticker}`    | Educational content for a stock                  |
| GET    | `/api/models/status`         | ML model availability status                     |
| POST   | `/api/agent/analyze`         | Trigger AI agent stock analysis                  |
| POST   | `/api/agent/build-watchlist` | Trigger AI watchlist builder                     |
| GET    | `/api/agent/status`          | Agent processing status                          |
| GET    | `/api/universe/info`         | Ticker universe metadata (~5,800 stocks)         |

### Example — Fetch Stock Data

```
GET /api/stock/AAPL
```

Response:

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "current_price": 195.5,
  "pe_ratio": 31.2,
  "forward_pe": 28.5,
  "pb_ratio": 48.1,
  "beta": 1.24,
  "dividend_yield": 0.005,
  "market_cap": 3010000000000,
  "sector": "Technology",
  "valuation_score": 65,
  "risk_level": "Medium"
}
```

---

## 2. Pipeline Router (`/api/pipeline`)

Three-stage Quality Growth Pipeline endpoints.

| Method | Path                           | Description                                  |
| ------ | ------------------------------ | -------------------------------------------- |
| GET    | `/api/pipeline/attention`      | Stage 1 — Attention stocks                   |
| GET    | `/api/pipeline/qualified`      | Stage 2 — Qualified stocks                   |
| GET    | `/api/pipeline/classified`     | Stage 3 — Classified stocks (Buy/Hold/Watch) |
| GET    | `/api/pipeline/curated`        | Final curated watchlist (~100 stocks)        |
| GET    | `/api/pipeline/stock/{ticker}` | Pipeline status for a specific stock         |
| POST   | `/api/pipeline/trigger-scan`   | Trigger a new pipeline scan                  |
| GET    | `/api/pipeline/scan-history`   | History of past pipeline scans               |
| GET    | `/api/pipeline/summary`        | Pipeline statistics and overview             |

### Example — Curated Watchlist

```
GET /api/pipeline/curated
```

Response:

```json
{
  "curated_stocks": [
    {
      "ticker": "MSFT",
      "composite_score": 78,
      "classification": "Buy",
      "pillars": {
        "value": 72,
        "quality": 85,
        "growth": 80,
        "safety": 70,
        "momentum": 65
      },
      "margin_of_safety": 0.28
    }
  ],
  "total_count": 98,
  "last_updated": "2025-01-15T10:30:00Z"
}
```

---

## 3. Analysis Router (`/api/analysis`)

AI-powered stock analysis endpoints.

| Method | Path                                 | Description                          |
| ------ | ------------------------------------ | ------------------------------------ |
| GET    | `/api/analysis/stock/{ticker}`       | Full AI analysis (uses agent system) |
| GET    | `/api/analysis/stock/{ticker}/quick` | Quick analysis (lighter computation) |

---

## 4. Multi-Agent Router (`/api/multiagent`)

Full multi-agent orchestration endpoints.

| Method | Path                                          | Description                           |
| ------ | --------------------------------------------- | ------------------------------------- |
| GET    | `/api/multiagent/stock/{ticker}`              | Complete multi-agent analysis         |
| GET    | `/api/multiagent/stock/{ticker}/sentiment`    | Sentiment analysis only               |
| GET    | `/api/multiagent/stock/{ticker}/fundamentals` | Fundamentals analysis only            |
| GET    | `/api/multiagent/stock/{ticker}/ml-valuation` | LSTM-DCF + consensus only             |
| POST   | `/api/multiagent/watchlist/analyze`           | Analyse multiple stocks for watchlist |

### Example — Multi-Agent Analysis

```
GET /api/multiagent/stock/AAPL
```

Response:

```json
{
  "ticker": "AAPL",
  "analysis": {
    "summary": "Apple shows strong fundamentals with moderate valuation...",
    "recommendation": "Hold",
    "confidence": 0.72,
    "fair_value": 198.5,
    "current_price": 195.5,
    "margin_of_safety": 0.015
  },
  "scores": {
    "growth_score": 75,
    "sentiment_score": 68,
    "value_score": 62,
    "fundamentals_score": 81
  },
  "consensus": {
    "lstm_dcf": 0.7,
    "garp": 0.65,
    "risk": 0.6,
    "weighted_score": 0.6625
  }
}
```

---

## 5. Risk Profile Router (`/api/risk-profile`)

Personal risk capacity assessment and position sizing.

| Method | Path                                     | Description                            |
| ------ | ---------------------------------------- | -------------------------------------- |
| POST   | `/api/risk-profile/assess`               | Submit risk questionnaire, get profile |
| GET    | `/api/risk-profile/profile/{profile_id}` | Retrieve saved risk profile            |
| POST   | `/api/risk-profile/position-sizing`      | Calculate position sizes for portfolio |
| GET    | `/api/risk-profile/suitable-stocks`      | Get stocks matching risk profile       |
| GET    | `/api/risk-profile/methodology`          | Risk assessment methodology            |

### Example — Risk Assessment

```
POST /api/risk-profile/assess
Content-Type: application/json

{
  "age": 30,
  "investment_horizon": 10,
  "risk_tolerance": "moderate",
  "income_stability": "stable",
  "investment_experience": "intermediate"
}
```

---

## 6. Storage Router (`/api/storage`)

MongoDB-backed persistent storage for watchlists and strategies.

| Method | Path                                     | Description                 |
| ------ | ---------------------------------------- | --------------------------- |
| POST   | `/api/storage/watchlists`                | Create a new watchlist      |
| GET    | `/api/storage/watchlists`                | List all watchlists         |
| GET    | `/api/storage/watchlists/{watchlist_id}` | Get specific watchlist      |
| DELETE | `/api/storage/watchlists/{watchlist_id}` | Delete a watchlist          |
| GET    | `/api/storage/education`                 | Educational content         |
| GET    | `/api/storage/strategies`                | List investment strategies  |
| POST   | `/api/storage/strategies`                | Save an investment strategy |
| GET    | `/api/storage/health`                    | MongoDB connection health   |

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Stock ticker INVALID not found",
  "status_code": 404
}
```

| Status | Meaning                                 |
| ------ | --------------------------------------- |
| 200    | Success                                 |
| 404    | Resource not found                      |
| 422    | Validation error (invalid request body) |
| 500    | Internal server error                   |

---

## Rate Limiting

| Tier    | Limit               |
| ------- | ------------------- |
| Default | 100 requests/hour   |
| Premium | 1,000 requests/hour |

---

## Frontend API Client

The frontend consumes these endpoints via `frontend/src/lib/api.ts`, which exports 25 typed functions. See [06 — Frontend Guide](06_FRONTEND_GUIDE.md) for details.
