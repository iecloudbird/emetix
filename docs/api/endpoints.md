# API Endpoints Specification

## Base URL

```
http://localhost:5000/api/v1
```

## Authentication

Currently using API key authentication. Include in header:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### 1. Stock Analysis

#### GET /stocks/{ticker}

Get comprehensive stock analysis.

**Parameters:**

- `ticker` (path, required): Stock ticker symbol

**Response:**

```json
{
  "ticker": "AAPL",
  "fundamentals": {
    "pe_ratio": 25.5,
    "debt_equity": 1.2,
    "beta": 1.1,
    "market_cap": 2500000000000
  },
  "valuation": {
    "fair_value": 165.5,
    "current_price": 150.0,
    "margin_pct": 10.33,
    "recommendation": "BUY"
  },
  "risk": {
    "risk_category": "Low Risk",
    "risk_score": 0,
    "confidence": 85.2
  }
}
```

#### GET /stocks/{ticker}/risk

Get risk assessment for a stock.

#### GET /stocks/{ticker}/valuation

Get valuation analysis for a stock.

### 2. Portfolio Management

#### POST /portfolio/analyze

Analyze a portfolio.

**Request Body:**

```json
{
  "holdings": [
    { "ticker": "AAPL", "shares": 10 },
    { "ticker": "MSFT", "shares": 5 }
  ]
}
```

**Response:**

```json
{
  "total_value": 50000,
  "diversification_score": 7.5,
  "risk_level": "Medium",
  "recommendations": [...]
}
```

### 3. Watchlist

#### GET /watchlist/scan

Get AI-recommended stocks.

**Query Parameters:**

- `risk_level`: "low", "medium", "high"
- `min_undervalued_pct`: Minimum undervaluation percentage
- `limit`: Number of results

**Response:**

```json
{
  "recommendations": [
    {
      "ticker": "AAPL",
      "score": 8.5,
      "reason": "Undervalued by 20%, low risk",
      "metrics": {...}
    }
  ]
}
```

### 4. Agent Queries

#### POST /agent/query

Ask the AI agent a question.

**Request Body:**

```json
{
  "query": "Should I invest in Tesla?",
  "context": {
    "risk_tolerance": "low",
    "investment_horizon": "long-term"
  }
}
```

**Response:**

```json
{
  "agent": "RiskAgent",
  "analysis": "Based on current metrics...",
  "recommendation": "HOLD"
}
```

## Error Responses

```json
{
  "error": "Invalid ticker symbol",
  "code": 400,
  "message": "The ticker 'XYZ123' could not be found"
}
```

## Rate Limiting

- Free tier: 100 requests/hour
- Premium: 1000 requests/hour

---

_More endpoints coming in Phase 3_
