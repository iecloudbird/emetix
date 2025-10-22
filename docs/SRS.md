# Software Requirements Specification (SRS)

## JobHedge Investor - AI-Powered Stock Valuation Platform

**Document Version:** 1.0  
**Date:** October 22, 2025  
**Project Phase:** Phase 2→3 Transition  
**Status:** Living Document

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document provides a comprehensive description of the JobHedge Investor platform, an AI-powered stock valuation system designed for retail investors. It outlines functional and non-functional requirements, system architecture, and implementation stages.

### 1.2 Scope

JobHedge Investor is a multi-agent stock analysis platform that combines:

- Traditional financial analysis (DCF, P/E, PEG ratios)
- Machine learning models (LSTM-DCF, Random Forest Ensemble)
- AI agents powered by Large Language Models (Groq LLM)
- News sentiment analysis from multiple sources
- Intelligent watchlist management with contrarian opportunity detection

### 1.3 Intended Audience

- Development Team
- Project Stakeholders
- Future Contributors
- Academic Reviewers (FYP Assessment)

### 1.4 Project Overview

- **Timeline:** 30-week Final Year Project
- **Current Stage:** Phase 6 (ML-Powered Watchlist Integration)
- **Technology Stack:** Python, PyTorch, scikit-learn, LangChain, Groq LLM
- **Target Users:** Retail investors seeking institutional-grade analysis

---

## 2. Overall Description

### 2.1 Product Perspective

JobHedge Investor fills the gap between:

- **Basic stock screeners** (limited metrics, no AI)
- **Professional platforms** (expensive, Bloomberg Terminal costs $24K/year)

Our platform provides institutional-grade analysis accessible to retail investors.

### 2.2 Product Functions

1. **Stock Valuation** - 12+ financial metrics with 0-100 scoring
2. **Growth Screening** - GARP strategy (Growth at Reasonable Price)
3. **News Sentiment** - 4-source aggregation with deduplication
4. **ML Predictions** - LSTM-DCF fair value + RF expected returns
5. **Watchlist Management** - ML-enhanced scoring with contrarian detection
6. **Multi-Agent Orchestration** - Supervisor coordinates specialized agents

### 2.3 User Classes and Characteristics

- **Primary:** Retail investors (long-term, value-focused)
- **Secondary:** Financial students, data scientists
- **Expertise Level:** Intermediate to advanced financial knowledge

### 2.4 Operating Environment

- **Platform:** Windows 10/11, macOS, Linux
- **Python:** 3.10+
- **GPU:** Optional (CUDA 11.8+ for training)
- **Internet:** Required for data fetching and API calls

### 2.5 Design and Implementation Constraints

- **Rate Limits:** Free tier APIs (NewsAPI: 100/day, Groq: varies)
- **Data Sources:** Yahoo Finance (primary), Alpha Vantage (secondary)
- **Model Size:** Keep models under 2 MB for deployment
- **Inference Time:** < 300ms per stock (user experience requirement)

---

## 3. System Features

### 3.1 Valuation Analysis

**Priority:** HIGH  
**Description:** Comprehensive stock valuation with 12+ metrics

**Functional Requirements:**

- FR-VA-1: System shall calculate P/E, P/B, P/S, PEG ratios
- FR-VA-2: System shall compute fair value using DCF model
- FR-VA-3: System shall provide 0-100 scoring with weighted components
- FR-VA-4: System shall generate buy/hold/sell recommendations
- FR-VA-5: System shall compare stocks against industry peers

**Non-Functional Requirements:**

- NFR-VA-1: Valuation analysis shall complete in < 2 seconds
- NFR-VA-2: Scores shall be accurate within ±5% of manual calculations

### 3.2 ML-Powered Predictions

**Priority:** HIGH  
**Description:** Deep learning and ensemble models for advanced forecasting

**Functional Requirements:**

- FR-ML-1: System shall use LSTM for time-series forecasting (60 periods)
- FR-ML-2: System shall calculate DCF with LSTM-predicted cash flows
- FR-ML-3: System shall use Random Forest for multi-metric analysis
- FR-ML-4: System shall provide consensus scoring (4 models)
- FR-ML-5: System shall show confidence levels for predictions

**Non-Functional Requirements:**

- NFR-ML-1: ML inference shall complete in < 300ms per stock
- NFR-ML-2: LSTM validation loss shall be < 0.0001
- NFR-ML-3: RF feature importance shall be interpretable
- NFR-ML-4: Models shall gracefully fallback if unavailable

### 3.3 Watchlist Management

**Priority:** HIGH  
**Description:** Intelligent watchlist with ML-enhanced scoring

**Functional Requirements:**

- FR-WL-1: System shall score stocks using ML predictions (45% weight)
- FR-WL-2: System shall detect contrarian opportunities
- FR-WL-3: System shall rank stocks by composite score
- FR-WL-4: System shall generate automated alerts
- FR-WL-5: System shall persist watchlist data

**Non-Functional Requirements:**

- NFR-WL-1: Watchlist generation for 10 stocks shall complete in < 2 minutes
- NFR-WL-2: Scores shall update in real-time when data changes
- NFR-WL-3: System shall handle 100+ stocks in watchlist

### 3.4 Multi-Agent System

**Priority:** MEDIUM  
**Description:** AI agents for orchestrated analysis

**Functional Requirements:**

- FR-MA-1: SupervisorAgent shall coordinate all specialized agents
- FR-MA-2: DataFetcherAgent shall retrieve stock data
- FR-MA-3: SentimentAnalyzerAgent shall analyze news sentiment
- FR-MA-4: FundamentalsAnalyzerAgent shall compute financial metrics
- FR-MA-5: WatchlistManagerAgent shall manage intelligent watchlist
- FR-MA-6: EnhancedValuationAgent shall provide ML predictions

**Non-Functional Requirements:**

- NFR-MA-1: Agent queries shall complete in < 10 seconds
- NFR-MA-2: System shall handle agent failures gracefully
- NFR-MA-3: Agents shall log all actions for debugging

### 3.5 News Sentiment Analysis

**Priority:** MEDIUM  
**Description:** Multi-source news aggregation with sentiment scoring

**Functional Requirements:**

- FR-NS-1: System shall fetch news from 4 sources (Yahoo, NewsAPI, Finnhub, Google)
- FR-NS-2: System shall deduplicate articles (85% similarity threshold)
- FR-NS-3: System shall score sentiment (0-1 scale)
- FR-NS-4: System shall fallback to Finnhub on rate limits
- FR-NS-5: System shall provide confidence levels

**Non-Functional Requirements:**

- NFR-NS-1: News fetching shall complete in < 5 seconds
- NFR-NS-2: System shall return 30-40 unique articles per stock
- NFR-NS-3: Sentiment accuracy shall be > 70% vs manual labeling

---

## 4. External Interface Requirements

### 4.1 User Interfaces

- **CLI:** Primary interface for Phase 2
- **API:** REST API planned for Phase 3
- **Web UI:** React frontend planned for Phase 3

### 4.2 Hardware Interfaces

- **GPU:** Optional NVIDIA GPU for training (CUDA 11.8+)
- **Storage:** 10 GB minimum for models and data

### 4.3 Software Interfaces

- **Yahoo Finance API:** Primary stock data source
- **Alpha Vantage API:** Secondary financial data
- **Groq LLM API:** AI agent intelligence
- **NewsAPI:** News articles (100/day limit)
- **Finnhub API:** Fallback news source

### 4.4 Communications Interfaces

- **HTTP/HTTPS:** For API calls
- **JSON:** Data exchange format
- **WebSocket:** Planned for real-time updates (Phase 3)

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

- **Response Time:** < 2 seconds for valuation analysis
- **ML Inference:** < 300ms per stock (all models)
- **Throughput:** Handle 10 concurrent stock analyses
- **Data Fetching:** < 5 seconds for complete dataset

### 5.2 Safety Requirements

- **Data Validation:** All inputs validated before processing
- **Error Handling:** Graceful fallbacks, no crashes
- **API Rate Limiting:** Respect free tier limits

### 5.3 Security Requirements

- **API Keys:** Stored in .env file (never committed)
- **Data Privacy:** No user PII collected
- **Input Sanitization:** Prevent injection attacks

### 5.4 Software Quality Attributes

- **Reliability:** 99% uptime during analysis
- **Maintainability:** Modular architecture, clear documentation
- **Portability:** Cross-platform (Windows, macOS, Linux)
- **Usability:** Clear error messages, helpful logging

---

## 6. Implementation Stages

See `docs/implementation_stages.md` for detailed stage-by-stage progress tracking.

**Summary:**

- ✅ Stage 1-5: Core models and agents
- 🔄 Stage 6: ML-Powered Watchlist Integration (Current)
- 📅 Stage 7-10: API, Frontend, Deployment

---

## 7. Appendices

### Appendix A: Glossary

- **DCF:** Discounted Cash Flow valuation model
- **LSTM:** Long Short-Term Memory neural network
- **RF:** Random Forest ensemble model
- **GARP:** Growth at Reasonable Price investment strategy
- **P/E:** Price-to-Earnings ratio
- **PEG:** Price/Earnings-to-Growth ratio

### Appendix B: Analysis Models

- **Traditional:** Linear regression, classification
- **Deep Learning:** LSTM-DCF hybrid
- **Ensemble:** Random Forest multi-metric
- **Consensus:** 4-model weighted voting

### Appendix C: References

- Financial Modeling Best Practices
- Machine Learning for Finance (López de Prado)
- LangChain Documentation
- PyTorch Lightning Training Guide

---

**Document Control:**

- **Last Updated:** October 22, 2025
- **Version:** 1.0
- **Next Review:** Weekly during active development
- **Owner:** FYP Development Team
