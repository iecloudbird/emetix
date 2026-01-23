# 1. Executive Summary

> **Emetix: AI-Powered Low-Risk Stock Watchlist Platform**

---

## 🎯 Project Overview

**Emetix** (אמת "truth" + Matrix) is an AI-powered stock valuation platform that provides institutional-grade analysis for retail investors. The platform combines machine learning models, multi-agent AI orchestration, and comprehensive financial analysis to identify undervalued, low-risk investment opportunities.

### Mission Statement

_Democratize professional stock analysis by making institutional-grade valuation tools accessible to retail investors through AI-powered automation._

---

## 💡 Value Proposition

### The Problem

| Challenge                 | Impact                                                      |
| ------------------------- | ----------------------------------------------------------- |
| **Information Asymmetry** | Retail investors lack access to professional-grade analysis |
| **Time Constraints**      | Manual DCF calculations take 2-4 hours per stock            |
| **Complexity Barrier**    | Financial modeling requires specialized knowledge           |
| **Cost Prohibitive**      | Bloomberg terminals cost $24,000/year                       |
| **Emotional Bias**        | Human decisions prone to FOMO, loss aversion                |

### Our Solution

```
┌────────────────────────────────────────────────────────────────┐
│                    EMETIX PLATFORM                              │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  LSTM-DCF    │  │ GARP Scorer  │  │  LangChain   │         │
│  │  Fair Value  │  │  PEG-based   │  │  AI Agents   │         │
│  │    50%       │  │     25%      │  │   (Gemini)   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └────────────┬────┴─────────────────┘                  │
│                      ▼                                         │
│         ┌──────────────────────┐                               │
│         │  Consensus Scorer    │                               │
│         │  LSTM 50% + GARP 25% │                               │
│         │  + Risk 25%          │                               │
│         └──────────────────────┘                               │
│                      │                                         │
│                      ▼                                         │
│         ┌──────────────────────┐                               │
│         │ Classified Stocks    │                               │
│         │ Buy / Hold / Watch   │                               │
│         └──────────────────────┘                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Key Achievements

### Technical Milestones

| Achievement               | Details                                               |
| ------------------------- | ----------------------------------------------------- |
| ✅ **LSTM-DCF Model**     | Deep learning growth prediction (1.3 MB, 104 tickers) |
| ✅ **GARP Scoring**       | Transparent PEG-based value scoring (replaced RF)     |
| ✅ **Multi-Agent System** | 8 specialized AI agents with Gemini LLM               |
| ✅ **3-Stage Pipeline**   | Attention → Qualified → Classified screening          |
| ✅ **4-Pillar Scoring**   | Value, Quality, Growth, Safety (25% each)             |
| ✅ **FastAPI Backend**    | Production-ready REST API with 15+ endpoints          |
| ✅ **GPU Training**       | 6-minute training on RTX 3050 (10x faster)            |

### Model Evaluation (January 2026)

| Model       | Training Data                        | Key Finding                          |
| ----------- | ------------------------------------ | ------------------------------------ |
| LSTM-DCF    | 8,355 samples, 104 tickers, 22 years | Predicts revenue & FCF growth        |
| GARP Scorer | Pure formula                         | Transparent PEG-based (no black-box) |

> **Current Weights**: LSTM-DCF 50%, GARP 25%, Risk 25%

### Validation Results

- **Backtest Coverage**: 78 stocks, 1,431 predictions, 7 sectors
- **SPY Win Rate**: 80% of cohorts outperformed S&P 500 (2010-2020)
- **Model Size**: Combined < 2 MB (deployment-friendly)

---

## 👥 Target Users

### Primary Audience

| User Type              | Description                          | Key Needs                         |
| ---------------------- | ------------------------------------ | --------------------------------- |
| **Retail Investors**   | Long-term, value-focused individuals | Fair value estimates, risk scores |
| **Financial Students** | Learning investment analysis         | Educational insights, methodology |
| **DIY Traders**        | Self-directed investors              | Quick screening, watchlists       |

### User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER JOURNEY                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DISCOVER          2. ANALYZE           3. DECIDE            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Market Scan  │ →  │ Stock Detail │ →  │ Add to       │       │
│  │ Top 10 List  │    │ Fair Value   │    │ Watchlist    │       │
│  │ Sector View  │    │ Risk Level   │    │ Set Alerts   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ API:         │    │ API:         │    │ API:         │       │
│  │ /watchlist   │    │ /stock/{id}  │    │ /compare     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Competitive Landscape

| Feature               | Basic Screeners | **Emetix**          | Bloomberg        |
| --------------------- | --------------- | ------------------- | ---------------- |
| **Price**             | $0-50/month     | **Free/Premium**    | $24,000/year     |
| **Valuation Metrics** | 3-5 basic       | **12+ advanced**    | 20+ professional |
| **AI Analysis**       | None            | **Multi-agent LLM** | Rules-based      |
| **ML Predictions**    | None            | **LSTM-DCF + GARP** | Statistical      |
| **Fair Value**        | N/A             | **LSTM-DCF**        | DCF templates    |
| **User Level**        | Beginner        | **Intermediate**    | Professional     |

---

## 🎯 Project Scope (FYP)

### Timeline: 30-Week Final Year Project

| Phase       | Duration    | Deliverables               | Status         |
| ----------- | ----------- | -------------------------- | -------------- |
| **Phase 1** | Weeks 1-4   | Environment, data pipeline | ✅ Complete    |
| **Phase 2** | Weeks 5-10  | Core valuation, agents     | ✅ Complete    |
| **Phase 3** | Weeks 11-16 | ML models, training        | ✅ Complete    |
| **Phase 4** | Weeks 17-22 | FastAPI backend            | ✅ Complete    |
| **Phase 5** | Weeks 23-28 | Frontend (React/Next.js)   | 🔄 In Progress |
| **Phase 6** | Weeks 29-30 | Testing, documentation     | 📋 Planned     |

### Current Status: Week 24

- Backend API: **100% functional**
- ML Models: **Trained and deployed**
- Documentation: **Comprehensive**
- Frontend: **Ready to implement**

---

## 🌍 Social Impact

### Democratizing Financial Literacy

| Initiative                  | Implementation                            | Impact                    |
| --------------------------- | ----------------------------------------- | ------------------------- |
| **AI Educational Insights** | Explain why metrics matter                | Reduces knowledge barrier |
| **Low-Risk Focus**          | Default filter for conservative investors | Protects beginners        |
| **Plain Language**          | Justifications in everyday terms          | Accessible to non-experts |
| **Multi-Language**          | Bahasa/Chinese/Tamil (future)             | Malaysian market reach    |

### Ethical AI Considerations

- **Transparency**: All scoring methodology documented
- **No Black Box**: Users can see why stocks are ranked
- **Risk Warnings**: Clear risk level classifications
- **No Financial Advice Disclaimer**: Educational tool only

---

## 📈 Success Metrics

### Technical KPIs

| Metric            | Target          | Current    |
| ----------------- | --------------- | ---------- |
| API Response Time | < 300ms         | ✅ ~200ms  |
| Model Accuracy    | > 60% direction | 🔄 Testing |
| Uptime            | > 99%           | N/A (Dev)  |
| Stock Coverage    | 150+ tickers    | ✅ 156     |

### Business KPIs (Future)

| Metric             | Year 1 Target |
| ------------------ | ------------- |
| Active Users       | 1,000         |
| Premium Conversion | 10%           |
| API Partners       | 5             |

---

_Next: [2. System Architecture](./02_SYSTEM_ARCHITECTURE.md)_
