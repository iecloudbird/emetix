# Stock Valuation Metrics & Undervalued Growth Detection

## Overview

This document outlines the comprehensive valuation metrics framework implemented in JobHedge Investor for identifying fair value and undervalued stocks, with special focus on growth stocks with lagged price performance.

## Core Valuation Metrics

### 1. Price Ratios

- **P/E Ratio (Price-to-Earnings)**: Stock price ÷ EPS

  - **Trailing P/E**: Uses past 12 months earnings
  - **Forward P/E**: Uses projected earnings
  - **Threshold**: Below industry average suggests undervaluation

- **P/B Ratio (Price-to-Book)**: Stock price ÷ Book value per share

  - **P/B < 1**: Trading below net asset value (potential undervaluation)
  - **Use case**: Asset-heavy industries (banking, manufacturing)

- **P/S Ratio (Price-to-Sales)**: Stock price ÷ Revenue per share

  - **Use case**: High-growth or unprofitable companies
  - **Threshold**: Low P/S with expanding revenue signals undervaluation

- **PEG Ratio**: P/E ÷ EPS growth rate
  - **PEG < 1**: Undervalued considering growth
  - **PEG > 2**: Potentially overvalued

### 2. Financial Health Metrics

- **Debt-to-Equity (D/E)**: Total debt ÷ Shareholders' equity

  - **D/E < 1**: Strong financial position
  - **Lower risk**: Supports undervaluation thesis

- **Current Ratio**: Current assets ÷ Current liabilities
  - **> 1.5**: Good liquidity position
  - **Indicates**: Financial stability

### 3. Cash Flow & Profitability

- **Free Cash Flow Yield**: FCF per share ÷ Stock price

  - **High yield (>5-10%)**: Strong cash generation
  - **Indicates**: Undervaluation if not reflected in price

- **ROE (Return on Equity)**: Net income ÷ Shareholders' equity
  - **> 15%**: Excellent profitability
  - **ROA (Return on Assets)**: > 5% indicates efficient asset use

### 4. Enterprise Value Metrics

- **EV/EBITDA**: Enterprise value ÷ EBITDA

  - **< 10**: Potential undervaluation
  - **Best for**: Capital-intensive firms

- **Dividend Yield**: Annual dividends ÷ Stock price
  - **> 3-4%**: High yield for mature companies
  - **Indicates**: Potential undervaluation (if sustainable)

## Undervalued Growth Stock Detection

### Strategy: Growth at Reasonable Price (GARP)

Identify stocks with:

1. **Strong Revenue Growth** (>10-20% YoY)
2. **Lagged Stock Performance** (<5% YTD or below market)
3. **Reasonable Valuation** (PEG < 1.5)

### Key Screening Parameters

#### Growth Metrics

- **Revenue Growth Rate**: >15% YoY (last 1-3 years)
- **EPS Growth Rate**: >10-15% YoY
- **Profit Margin Expansion**: Gross margin >40%

#### Price Performance Filters

- **Stock Return Threshold**: <5% YTD or <market benchmark
- **Relative Strength**: Underperforming vs. sector/index
- **Price Momentum**: Low or negative

#### Valuation Constraints

- **PEG Ratio**: <1.5 (growth not overpriced)
- **P/S Ratio**: <2-3 for high-growth firms
- **Market Cap**: >$1B (stability filter)

#### Quality Filters

- **ROE**: >10% (profitability)
- **Free Cash Flow**: >0 (cash generation)
- **Debt-to-Equity**: <2 (manageable debt)

## Implementation in JobHedge Investor

### 1. Enhanced Data Fetching

Extended `YFinanceFetcher` to collect additional metrics:

- Forward P/E, P/B, P/S ratios
- Free cash flow and margins
- Growth rates and analyst estimates

### 2. Valuation Analysis Module

New `src/analysis/valuation_analyzer.py` provides:

- Comprehensive metric calculations
- Peer comparison capabilities
- Undervaluation scoring system

### 3. Growth Stock Screener

`src/analysis/growth_screener.py` implements:

- Multi-criteria filtering
- Ranking algorithms
- Performance tracking

### 4. Integration with AI Agents

Enhanced agent tools for:

- Real-time valuation analysis
- Growth opportunity identification
- Risk-adjusted recommendations

## Usage Examples

### Basic Valuation Analysis

```python
from src.analysis import ValuationAnalyzer

analyzer = ValuationAnalyzer()
result = analyzer.analyze_stock('AAPL')
print(f"Valuation Score: {result['score']}")
print(f"Fair Value Estimate: ${result['fair_value']:.2f}")
```

### Growth Stock Screening

```python
from src.analysis import GrowthScreener

screener = GrowthScreener()
candidates = screener.find_undervalued_growth_stocks(
    min_revenue_growth=15,
    max_ytd_return=5,
    max_peg_ratio=1.5
)
```

### Agent Integration

```python
from src.agents import ValuationAgent

agent = ValuationAgent()
analysis = agent.comprehensive_analysis('TSLA')
print(analysis['recommendation'])
```

## Real-World Examples (October 2025)

### Undervalued Growth Candidates

Based on screening criteria:

1. **Globant (GLOB)**

   - Revenue Growth: 15% (2024)
   - YTD Return: -72%
   - Opportunity: Tech services recovery play

2. **Molina Healthcare (MOH)**

   - Revenue Growth: 18% YoY (Q2 2025)
   - YTD Return: -30%
   - Catalyst: Healthcare membership expansion

3. **Gartner (IT)**
   - Revenue Growth: 6-8%
   - YTD Return: -47%
   - Value Play: Enterprise research demand

## Risk Considerations

### Model Limitations

- **Market Sentiment**: Metrics don't capture emotional factors
- **Sector Rotation**: Industry trends affect valuations
- **Economic Cycles**: Macro conditions impact all metrics

### Quality Checks

- **Management Quality**: Assess leadership effectiveness
- **Competitive Moats**: Evaluate sustainable advantages
- **Market Conditions**: Consider broader economic environment

## Integration with Existing Models

### Enhanced Risk Assessment

Combine valuation metrics with existing risk classifier:

- Low valuation + Low risk = Strong buy candidate
- High growth + Reasonable price = Growth opportunity

### Portfolio Optimization

Use metrics for:

- Diversification across value/growth styles
- Risk-adjusted position sizing
- Rebalancing triggers

## Future Enhancements

### Phase 3 Developments

- **Real-time Screening**: Automated daily scans
- **Peer Comparison**: Industry-relative metrics
- **Backtesting**: Historical performance validation
- **Alert System**: Notification of opportunities

### Advanced Analytics

- **Sector Analysis**: Industry-specific thresholds
- **Momentum Integration**: Technical + fundamental signals
- **ESG Factors**: Sustainability metrics inclusion
