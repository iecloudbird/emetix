# 📊 Stock Analysis Tool - Quick Guide

## 🚀 Your New Go-To Tool: `analyze_stock.py`

This script provides **comprehensive stock analysis** using all JobHedge Investor systems in one place!

---

## ✅ What It Includes

### **5 Analysis Modules:**

1. **Valuation Metrics** (12+ financial ratios)

   - P/E, P/B, P/S, PEG ratios
   - Financial health (debt, liquidity)
   - Profitability (ROE, margins)
   - FCF yield, dividend yield
   - Fair value estimation
   - Risk assessment

2. **Growth Opportunity Analysis** (GARP screening)

   - Revenue/earnings growth
   - YTD performance
   - PEG ratio analysis
   - ROE and margins
   - Investment thesis

3. **News Sentiment Analysis** (4 sources)

   - Yahoo Finance, NewsAPI, Finnhub, Google News
   - Automatic deduplication (85% similarity)
   - Smart fallback when rate limits hit
   - Sentiment scoring (0-1 scale)
   - Top headlines with sentiment labels

4. **AI-Powered Analysis** (optional, requires GROQ_API_KEY)

   - Valuation Agent (comprehensive insights)
   - Risk Agent (risk assessment)
   - Natural language explanations

5. **Final Summary & Recommendation**
   - Overall assessment
   - Investment decision
   - Special notes (GARP screening, contrarian opportunities)

---

## 📖 Usage Examples

### **1. Analyze Single Stock**

```bash
# Full analysis with AI agents
python scripts/analyze_stock.py AAPL

# Basic analysis (no AI, faster)
python scripts/analyze_stock.py AAPL --basic

# Multiple stocks
python scripts/analyze_stock.py AAPL MSFT GOOGL
```

### **2. Compare Stocks**

```bash
# Side-by-side comparison
python scripts/analyze_stock.py AAPL MSFT GOOGL --compare

# Output shows:
# - Valuation scores ranked
# - Growth screening results
# - News sentiment comparison
```

### **3. Find Best Opportunities**

```bash
# Screen for GARP opportunities
python scripts/analyze_stock.py AAPL MSFT GOOGL TSLA NVDA META AMZN --opportunities

# Shows stocks passing:
# - Revenue growth >15%
# - YTD return <5%
# - PEG <1.5
# - Strong fundamentals
```

---

## 📊 Sample Output (AAPL)

```
================================================================================
📊 COMPREHENSIVE ANALYSIS: AAPL
================================================================================

────────────────────────────────────────────────────────────────────────────────
1️⃣  VALUATION METRICS
────────────────────────────────────────────────────────────────────────────────

💰 Current Price: $257.83
🎯 Fair Value Est: $129.17
📈 Valuation Score: 50/100
🏷️  Assessment: SLIGHTLY OVERVALUED
⚠️  Risk Level: HIGH
💡 Recommendation: WEAK HOLD

📊 Key Metrics:
   • pe_ratio: 39.12
   • pb_ratio: 58.19
   • peg_ratio: 0.00
   • debt_equity: 1.54
   • roe: 149.81%
   • fcf_yield: 2.48%

────────────────────────────────────────────────────────────────────────────────
2️⃣  GROWTH OPPORTUNITY ANALYSIS
────────────────────────────────────────────────────────────────────────────────

🚀 Opportunity Score: 72/100
✅ Passed GARP Screening: NO
📈 Growth Momentum: DECLINING
💎 Valuation Attractiveness: FAIR

📊 Growth Metrics:
   • revenue_growth: 9.60%
   • ytd_return: 6.11%
   • roe: 149.81%

────────────────────────────────────────────────────────────────────────────────
3️⃣  NEWS SENTIMENT ANALYSIS
────────────────────────────────────────────────────────────────────────────────

📰 Total Articles: 45
📋 Unique Articles: 35 (10 duplicates removed)
📊 Sentiment Score: 0.67/1.0
🎯 Sentiment Label: BULLISH
🌐 Sources: Yahoo Finance, NewsAPI, Google News

📈 Sentiment Breakdown:
   • Positive: 13 (37.1%)
   • Negative: 1 (2.9%)
   • Neutral: 21 (60.0%)

📰 Top 3 Recent Headlines:
   1. ⚪ [NEUTRAL] Analyst Says Apple (AAPL) Is Going to Make a New All-Time High
   2. 🟢 [POSITIVE] Apple (NASDAQ:AAPL) Raised to Strong-Buy at CLSA
   3. 🟢 [POSITIVE] JPMorgan Stays Bullish on Apple (AAPL)

────────────────────────────────────────────────────────────────────────────────
5️⃣  FINAL SUMMARY
────────────────────────────────────────────────────────────────────────────────

📊 Overall Assessment:
   • Valuation: SLIGHTLY OVERVALUED
   • Score: 50/100
   • Risk: HIGH
   • Sentiment: BULLISH (0.67)
   • Growth: DECLINING

💡 Investment Decision: WEAK HOLD
```

---

## 🎯 Common Use Cases

### **For Quick Analysis:**

```bash
# Fast check of any stock
python scripts/analyze_stock.py TSLA --basic
```

### **For Deep Dive:**

```bash
# Full analysis with AI insights
python scripts/analyze_stock.py NVDA
```

### **For Portfolio Building:**

```bash
# Compare candidates
python scripts/analyze_stock.py AAPL MSFT GOOGL AMZN --compare
```

### **For Value Hunting:**

```bash
# Find undervalued growth stocks
python scripts/analyze_stock.py AAPL MSFT GOOGL TSLA NVDA META CRM ADBE --opportunities
```

---

## 🔧 System Requirements

### **Basic Mode** (always works):

- ✅ Valuation metrics
- ✅ Growth screening
- ✅ News sentiment
- ✅ Summary & recommendation

### **Full Mode** (requires GROQ_API_KEY):

- ✅ All basic features
- ✅ AI Valuation Agent analysis
- ✅ AI Risk Agent assessment
- ✅ Natural language insights

**Get GROQ API Key (FREE)**: https://console.groq.com

---

## 📚 Other Available Test Scripts

### **1. `test_valuation_system.py`** - Original comprehensive test

```bash
python scripts/test_valuation_system.py
```

Tests all modules with demo stocks (AAPL, NVDA, etc.)

### **2. `test_smart_news_fallback.py`** - News sentiment testing

```bash
python scripts/test_smart_news_fallback.py MSFT
```

Tests multi-source news fetching with fallback system

### **3. `test_agent.py`** - Risk agent testing

```bash
python scripts/test_agent.py
```

Tests Risk Agent with stock comparison

### **4. `analyze_stock.py`** ⭐ **RECOMMENDED**

```bash
python scripts/analyze_stock.py AAPL
```

**All-in-one tool** - Use this for production analysis!

---

## 💡 Pro Tips

### **1. Batch Analysis**

```bash
# Analyze your entire watchlist
python scripts/analyze_stock.py AAPL MSFT GOOGL TSLA NVDA META AMZN NFLX
```

### **2. Compare Similar Stocks**

```bash
# Tech stocks
python scripts/analyze_stock.py AAPL MSFT GOOGL --compare

# EV manufacturers
python scripts/analyze_stock.py TSLA RIVN LCID --compare
```

### **3. Screen for Opportunities**

```bash
# S&P 500 sample
python scripts/analyze_stock.py AAPL MSFT GOOGL AMZN META TSLA NVDA V JPM --opportunities
```

### **4. Quick Daily Check**

```bash
# Fast sentiment check
python scripts/test_smart_news_fallback.py AAPL

# Quick valuation
python scripts/analyze_stock.py AAPL --basic
```

---

## 🎨 Output Features

- ✅ **Color-coded** emojis for quick visual scanning
- ✅ **Clean formatting** with section dividers
- ✅ **Comprehensive data** from 12+ metrics
- ✅ **Actionable recommendations** (BUY/HOLD/SELL)
- ✅ **News headlines** with sentiment labels
- ✅ **Risk warnings** and special notes
- ✅ **Investment thesis** for growth stocks

---

## 🚀 Next Steps

1. **Test with your favorite stock:**

   ```bash
   python scripts/analyze_stock.py [YOUR_TICKER] --basic
   ```

2. **Compare your watchlist:**

   ```bash
   python scripts/analyze_stock.py TICKER1 TICKER2 TICKER3 --compare
   ```

3. **Enable AI analysis** (optional):
   - Get free GROQ API key: https://console.groq.com
   - Add to `.env`: `GROQ_API_KEY=your_key_here`
   - Run full analysis: `python scripts/analyze_stock.py AAPL`

---

## 📊 What Makes This Tool Special

| Feature                    | Status | Source                 |
| -------------------------- | ------ | ---------------------- |
| 12+ Valuation Metrics      | ✅     | `ValuationAnalyzer`    |
| GARP Growth Screening      | ✅     | `GrowthScreener`       |
| Multi-Source News (4 APIs) | ✅     | `NewsSentimentFetcher` |
| Smart Fallback System      | ✅     | Tiered API priority    |
| Deduplication (85%)        | ✅     | Title similarity       |
| AI Valuation Analysis      | ✅     | `ValuationAgent`       |
| AI Risk Assessment         | ✅     | `RiskAgent`            |
| Side-by-Side Comparison    | ✅     | Ranking algorithm      |
| Opportunity Screening      | ✅     | GARP strategy          |

---

**Your complete stock analysis system in one command!** 🎉

**Quick Start**: `python scripts/analyze_stock.py AAPL --basic`
