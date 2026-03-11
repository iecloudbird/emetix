# Democratizing Stock Valuation: A Multi-Agent AI Platform with Personal Risk Capacity Framework for Retail Investors

**Shean Hans Teoh**
Technological University of the Shannon
Athlone, Ireland
A00303759@student.tus.ie

**Dr. Ye**
Technological University of the Shannon
Athlone, Ireland

---

## ABSTRACT

Retail investors face significant barriers to professional-grade stock valuation tools. Bloomberg Terminal costs $25,000 annually, while free platforms like Yahoo Finance provide raw data without AI-driven analysis. Meanwhile, fintech solutions focus predominantly on high-frequency trading rather than educational, risk-aware long-term strategies. Emetix addresses this gap through a multi-agent AI platform combining an LSTM-DCF V2 deep learning model, a 3-stage Quality Growth Pipeline that filters approximately 5,800 US equities to around 100 curated picks, and a novel Personal Risk Capacity (PRC) framework. The PRC introduces investor-specific risk personalisation using a 5-tier emotional buffer scale, shifting the question from "is this stock risky?" to "is this stock risky for you?" Backtesting across 2010–2020 shows 91% of yearly cohorts outperforming the S&P 500 in 5-year holding periods, while full stock analysis completes in seconds compared to 2–4 hours of manual DCF work. The platform is deployed as a free, full-stack web application, demonstrating that multi-agent AI orchestration with personalised risk profiling can meaningfully bridge the gap between institutional and retail investment analysis.

---

## 1 Introduction

### 1.1 Context and Rationale

Retail participation in US equity markets has grown substantially. As of 2025, 62% of Americans hold stocks, matching pre-2008 levels, with everyday investors now controlling 38% of US equities [12]. This expansion, driven by commission-free trading apps and social media communities, has introduced millions of new investors into markets they may not fully understand.

The 2022 bear market demonstrated the consequences of this knowledge gap. Rapid information overload combined with incomplete risk evaluations led to widespread losses, particularly among investors making decisions driven by sentiment rather than fundamental analysis [11]. Professional analysts rely on tools like Bloomberg Terminal to perform discounted cash flow (DCF) valuations, screen for quality metrics, and assess sector-level risk — but at $25,000 per year, these tools are firmly out of reach for retail investors [1].

Free alternatives such as Yahoo Finance and Google Finance provide useful data but lack integrated analysis. A retail investor can look up a stock's P/E ratio, but calculating intrinsic value through DCF requires manually projecting cash flows, selecting discount rates, and assessing growth assumptions — a process that takes 2–4 hours per stock even for experienced analysts. This asymmetry disproportionately affects younger and lower-income investors who lack both the financial resources for premium tools and the expertise to perform manual valuation [12].

Emetix addresses this gap through automation. Rather than replacing investor judgement, it applies systematic AI-driven screening and quantitative scoring to reduce the universe of US equities to a manageable, quality-filtered watchlist — empowering retail investors to focus their research time on the stocks most likely to merit attention.

### 1.2 Research Questions

This project investigates three questions:

1. Can a multi-agent AI system automate institutional-quality stock valuation for retail investors by integrating multiple data sources with parallel LLM-driven analysis?
2. Does an LSTM-DCF hybrid model improve upon traditional static DCF methods by incorporating dynamic time-series forecasting of fundamentals?
3. How can risk assessment be personalised at the individual stock level, shifting from one-size-fits-all metrics like beta to a framework that considers investor experience and emotional tolerance?

### 1.3 Contributions

Emetix makes four primary contributions:

- **Multi-agent orchestration**: Eight LangGraph agents coordinate parallel analysis tasks including valuation, risk assessment, sentiment analysis, and watchlist management.
- **LSTM-DCF V2 model**: A 2-layer LSTM trained on quarterly fundamentals forecasts 10-year free cash flow to firm (FCFF), feeding directly into DCF valuation.
- **Personal Risk Capacity (PRC) framework**: A 5-tier emotional buffer system adjusts risk thresholds based on investor experience, shifting from objective stock risk to personalised investor-stock fit.
- **Deployed full-stack platform**: The system runs as an accessible web application on Vercel (frontend), Render.com (backend), and MongoDB Atlas (database), at no cost to end users.

### 1.4 Report Structure

Section 2 reviews relevant literature on market theory, valuation methods, machine learning in finance, and risk personalisation. Section 3 presents the system design and methodology, covering architecture, the 3-stage pipeline, LSTM model, multi-agent system, and PRC framework. Section 4 evaluates results through backtesting, system testing, and a discussion of limitations. Section 5 concludes with implications and future work.

---

## 2 Related Work

### 2.1 Efficient Market Hypothesis and Behavioural Finance

The Efficient Market Hypothesis (EMH) posits that asset prices fully reflect all available information, implying that consistent outperformance through screening is not possible [2]. In its strong form, EMH suggests neither fundamental nor technical analysis offers an edge, and that stock prices follow a random walk.

However, behavioural finance research by Kahneman and Tversky [5] demonstrates that investors systematically deviate from rational decision-making through biases including overconfidence, loss aversion, and herd behaviour. These cognitive biases create pockets of temporary mispricing that value-oriented approaches can exploit — a phenomenon well-documented in the value premium literature [3]. Emetix operates on the premise that markets are largely efficient but that systematic, multi-factor screening can identify stocks temporarily suppressed by sentiment rather than deteriorating fundamentals.

### 2.2 Traditional Valuation Methods

Discounted Cash Flow analysis remains the most widely used intrinsic valuation method, projecting future cash flows and discounting them at a rate reflecting the cost of capital [1]. Its primary limitation is sensitivity to growth rate assumptions and discount rate selection — small changes in inputs produce large swings in fair value estimates. Multi-factor models such as Fama-French [3] improve on single-metric approaches by incorporating size, value, and profitability factors, but remain academically oriented and inaccessible to typical retail investors.

### 2.3 Machine Learning in Financial Valuation

Recent work demonstrates that machine learning can reduce DCF forecasting errors by 20–30% compared to static methods [6]. Long Short-Term Memory (LSTM) networks have proven effective for financial time-series tasks due to their ability to capture long-range dependencies in sequential data [7]. Geertsema and Lu [4] apply ML to relative valuation, achieving improved accuracy in cross-sectional stock ranking. However, a notable gap persists: most ML-finance research focuses on price prediction for trading purposes rather than intrinsic value estimation for long-term investing. Emetix addresses this by using LSTM specifically for fundamental forecasting (revenue, FCFF) rather than price movement prediction.

### 2.4 Multi-Agent AI Systems

LangChain and LangGraph provide frameworks for building AI agents that decompose complex tasks into modular, parallelisable subtasks [9]. Current fintech AI integrations are largely limited to conversational chatbots that provide surface-level commentary. No existing platform combines multi-agent orchestration with quantitative ML valuation and personalised risk profiling into a unified screening pipeline.

### 2.5 Risk Assessment and Personalisation

Beta, the standard measure of systematic risk, captures a stock's volatility relative to the market [8] but is entirely backward-looking and stock-centric — it tells you how a stock moved historically, not how an individual investor will respond to that movement. Robo-advisors such as Betterment and Wealthfront personalise portfolio allocation based on risk questionnaires but operate at the portfolio level, not the individual stock level. They answer "what asset mix suits you?" but not "is this specific stock suitable for you?"

A clear gap exists in the literature for stock-level risk personalisation that accounts for investor-specific factors such as experience level, financial cushion, and emotional tolerance for volatility. Emetix's PRC framework addresses this gap by introducing investor-stock fit as a first-class screening criterion.

---

## 3 System Design and Methodology

### 3.1 Requirements

Functional requirements for Emetix include: automated data collection from three financial APIs (Yahoo Finance, Finnhub, Alpha Vantage); ML-based fair value estimation via LSTM-DCF; multi-agent AI analysis through LangGraph; a 3-stage pipeline producing Buy/Hold/Watch classifications; and personalised risk profiling via the PRC framework. Non-functional requirements include analysis response times under 5 seconds, educational plain-language summaries, and privacy compliance — no personal data is stored server-side, with risk profiles held exclusively in the browser's localStorage.

### 3.2 Architecture Overview

The system follows a 5-layer architecture, illustrated in Figure 1:

**Layer 1 — User Interface**: Next.js 16.1.1 with React 19 frontend, featuring a stock screener, AI analysis panels, risk profiling, and a command palette for quick navigation.

**Layer 2 — API Gateway**: FastAPI backend exposing 41 RESTful endpoints across 6 routers (pipeline, analysis, multi-agent, storage, screener, risk profile).

**Layer 3 — AI Agents**: Eight LangGraph-based agents, each with specialised tools, orchestrated by a Supervisor agent.

**Layer 4 — ML and Analysis**: LSTM-DCF V2 model, 5-Pillar Composite Scorer, Consensus Scorer, and Quality Growth Gate.

**Layer 5 — Data**: Yahoo Finance (primary), Finnhub, and Alpha Vantage for financial data; MongoDB Atlas (9 collections across 2 databases) for pipeline storage.

*Figure 1: High-level system architecture showing the 5-layer design from user interface through data sources.*

**Table 1: Technology Stack**

| Layer | Technologies |
|-------|-------------|
| Frontend | Next.js 16.1.1, React 19.2.3, Tailwind CSS v4, shadcn/ui |
| Backend | FastAPI, Python 3.10, Uvicorn |
| AI/LLM | LangGraph, Gemini 2.5-flash-lite (default), Gemini 2.5-flash (supervisor) |
| ML | PyTorch Lightning, LSTM-DCF V2 |
| Data Sources | Yahoo Finance, Finnhub, Alpha Vantage |
| Database | MongoDB Atlas (9 collections, 2 databases) |
| Deployment | Vercel (frontend), Render.com (backend) |

### 3.3 Three-Stage Quality Growth Pipeline

The pipeline progressively filters approximately 5,800 US-listed equities down to around 100 curated picks through three stages:

**Stage 1 — Attention Scan** (weekly): Applies five quantitative triggers targeting undervaluation signals (e.g., forward P/E below sector median, high insider ownership, improving margins). A Beneish M-Score veto (threshold: −2.22) flags potential earnings manipulation, rejecting companies whose accrual patterns suggest aggressive accounting [1]. This reduces the universe to approximately 200–400 candidate stocks.

**Stage 2 — Qualification** (daily): Applies the 5-Pillar Composite Scoring system (v3.1). Only stocks achieving a composite score of 60 or above advance, producing approximately 100–200 qualified stocks.

**Stage 3 — Curation** (on-demand): Classifies qualified stocks as Buy (composite ≥70, margin of safety ≥25%), Hold (composite ≥60, margin of safety −5% to 25%), or Watch (all others, with sub-categories: high-quality-expensive, cheap-junk, or needs-research). The final curated watchlist contains approximately 100 stocks.

*Figure 2: Pipeline funnel showing progressive filtering from ~5,800 stocks to ~100 curated picks across three stages.*

**Table 2: 5-Pillar Composite Scoring Weights (v3.1)**

| Pillar | Weight | Key Components |
|--------|--------|---------------|
| Value | 25% | Margin of Safety (40%), P/E vs Sector Median (30%), P/B Ratio (30%) |
| Quality | 25% | FCF Return on Invested Capital, Return on Equity |
| Growth | 20% | Revenue Growth, Earnings Growth |
| Safety | 15% | Debt-to-Equity Ratio, Beta |
| Momentum | 15% | RSI, Price Trend Indicators |

*Figure 5: 5-Pillar Composite scoring radar chart illustrating pillar weights and example score distribution.*

### 3.4 LSTM-DCF V2 Model

The core ML component is a Long Short-Term Memory network that forecasts 10-year free cash flow to firm (FCFF), which then feeds into a traditional DCF calculation for intrinsic value estimation.

**Architecture**: 2-layer LSTM with hidden size 128, dropout rate 0.3, and Huber Loss function (robust to outliers in financial data). The model processes sequences of 8 consecutive quarters (2 years) of fundamentals including revenue, free cash flow, operating margins, and growth rates.

**Training**: Conducted using PyTorch Lightning with GPU acceleration (CUDA 11.8). Training completes in approximately 6 minutes on an NVIDIA RTX 3050, compared to 30–60 minutes on CPU. Early stopping and a learning rate scheduler prevent overfitting.

**Consensus Scoring**: The LSTM-DCF output is combined with a GARP (Growth at a Reasonable Price) score and a Risk score through weighted consensus: LSTM-DCF 50%, GARP 25%, Risk 25%. This diversified weighting reduces single-model bias — replacing an earlier Random Forest component that was found to rely on P/E ratio at 99.93% feature importance, effectively functioning as a single-metric filter rather than a genuine ensemble member.

*Figure 3: LSTM-DCF V2 training pipeline from quarterly fundamentals through sequence processing to DCF fair value output.*

### 3.5 Multi-Agent System

Eight specialised agents, built using LangGraph's `create_react_agent` pattern, handle distinct analysis tasks:

1. **Supervisor**: Routes queries and orchestrates agent execution using Gemini 2.5-flash.
2. **Data Fetcher**: Retrieves live market data from Yahoo Finance with caching.
3. **Fundamentals Analyser**: Evaluates financial statement metrics.
4. **Valuation Agent**: Performs traditional DCF valuation.
5. **Enhanced Valuation Agent**: Integrates LSTM-DCF model output with 5 specialised tools.
6. **Risk Agent**: Classifies stock risk via beta thresholds and volatility metrics.
7. **Sentiment Analyser**: Processes news sentiment from Finnhub.
8. **Watchlist Manager**: Manages the curated watchlist and portfolio tracking.

All agent tools follow a strict pattern: they return strings (never raise exceptions) and handle errors gracefully with descriptive messages. The default LLM is Google Gemini 2.5-flash-lite, with Gemini 2.5-flash reserved for the Supervisor, and Groq (Llama-3.3-70b) available as a fallback provider.

*Figure 4: Multi-agent orchestration flow showing Supervisor routing to specialised agents.*

### 3.6 Personal Risk Capacity Framework

The PRC framework is Emetix's novel contribution to risk assessment. Traditional metrics like beta evaluate whether a stock is risky in market terms, but ignore whether a specific investor can absorb that risk. A stock with beta 0.8 is conventionally labelled "low risk," yet for a first-time investor with no emergency fund, even low-volatility holdings may trigger panic selling during normal market corrections.

**Design**: A questionnaire evaluates five dimensions: investment experience, time horizon, emergency fund adequacy, loss tolerance, and panic-sell tendency. From these inputs, the system calculates:

- **Risk Capacity Score**: Financial ability to absorb losses (emergency fund, savings rate, maximum tolerable loss)
- **Risk Tolerance Score**: Emotional ability to handle volatility (panic response, volatility comfort, horizon)
- **Emotional Buffer Factor**: A 5-tier multiplier applied to the base margin of safety threshold (20%):

| Experience Level | Buffer Factor | Adjusted MoS Threshold |
|-----------------|---------------|----------------------|
| Professional | 1.0× | 20% |
| Experienced | 1.25× | 25% |
| Intermediate | 1.5× | 30% |
| Beginner | 1.75× | 35% |
| First-time | 2.0× | 40% |

**Implementation**: Risk profiles are stored exclusively in the browser via a `useLocalRiskProfile()` React hook — no personal data leaves the client. In the screener, a toggle chip filters the stock list to show only stocks matching the investor's adjusted risk parameters (beta range, required margin of safety). When inactive, a contextual hint encourages the user to enable it: "Filter stocks to match your risk profile." This design educates users about personal risk tolerance through interaction rather than instruction.

*Figure 6: PRC framework flow from questionnaire input to screener filter integration.*

### 3.7 Frontend, UX, and Deployment

The frontend is built with Next.js 16, React 19, Tailwind CSS v4, and shadcn/ui components. Key UX features include a Smart Screener with tabbed Buy/Hold/Watch views, an AI Stock Preview Panel for instant company summaries, a Command Palette (Cmd+K) for quick navigation, and LLM analysis caching via localStorage (2-hour TTL for single-agent, 1-hour for multi-agent). A Cold-Start Awareness Banner detects when the Render.com backend wakes from its free-tier sleep, displaying a progress timer until the API responds.

The frontend auto-deploys from Git to Vercel. MongoDB Atlas hosts 9 collections across 2 databases. No user-specific data is persisted server-side.

*Figure 7: Screener interface showing risk profile chip, Buy/Hold/Watch tabs, and AI preview panel.*

---

## 4 Results, Testing, and Discussion

### 4.1 Backtesting Results

Backtesting was conducted using year-by-year cohort analysis across the 2010–2020 period. For each entry year, the system generated stock selections based on predicted upside from the LSTM-DCF pipeline. Actual 5-year returns were then compared against a matched S&P 500 (SPY) baseline held over the identical period, ensuring a fair like-for-like comparison.

**Table 3: Backtesting Results by Entry Year Cohort**

| Entry Year | Stocks Selected | Avg 5-Year Return | SPY 5-Year Return | Beat SPY |
|-----------|----------------|-------------------|-------------------|----------|
| 2010 | 1 | 97.0% | 105.3% | No |
| 2011 | 4 | 203.0% | 99.1% | Yes |
| 2012 | 3 | 147.9% | 136.5% | Yes |
| 2013 | 4 | 1,057.5% | 91.1% | Yes |
| 2014 | 4 | 481.9% | 97.5% | Yes |
| 2015 | 4 | 592.1% | 103.6% | Yes |
| 2016 | 4 | 638.7% | 164.4% | Yes |
| 2017 | 4 | 237.3% | 88.5% | Yes |
| 2018 | 4 | 450.9% | 95.5% | Yes |
| 2019 | 4 | 816.6% | 158.4% | Yes |
| 2020 | 3 | 766.8% | 130.8% | Yes |

Overall, 10 of 11 cohorts (91%) outperformed the S&P 500 benchmark. The single underperforming cohort (2010) selected only one stock (INTC), which illustrates the risk of concentrated positions and the importance of selection count.

These results require careful interpretation. Signal correlation between the system's predicted upside and actual 5-year returns is moderate (0.2–0.4), meaning the system tends to rank stocks in roughly the right order but with considerable noise. Individual direction accuracy — whether a predicted "undervalued" stock actually rose — is approximately 31%, which is weak on a per-stock basis. However, the system's portfolio-level outperformance is driven not by precise prediction but by its consistent ability to surface quality growth companies through multi-factor screening. The frequently selected tickers across cohorts (NVDA, AMD, AAPL, GOOGL, MSFT) reflect the system's bias toward companies with strong fundamentals and growth trajectories — exactly the profile the 5-Pillar scoring system is designed to identify.

A significant caveat is that this 2010–2020 backtest period coincides with an extended technology bull market. The system's performance during prolonged bear markets, sector rotations, or interest rate shocks remains unvalidated. This is an inherent limitation of any fundamentals-based screening approach applied retrospectively to a period of exceptional growth.

### 4.2 Pipeline Performance

The 3-stage pipeline operates at scale: from ~5,800 US-listed equities, Stage 1 produces ~200–400 attention candidates, Stage 2 qualifies ~100–200 via composite scoring, and Stage 3 curates the final ~100 picks with Buy/Hold/Watch classification. End-to-end analysis per stock completes in seconds, compared to 2–4 hours of manual DCF work. The system integrates three data sources with automatic fallback — if Alpha Vantage's rate limit (25 calls/day) is exhausted, the pipeline falls back to Yahoo Finance and Finnhub.

### 4.3 Model Evaluation

The LSTM-DCF V2 model was evaluated against the Software Requirements Specification (SRS) established during the design phase. The primary non-functional requirement was that ML inference must complete in under 300ms per stock. In testing, average inference time was 24ms — well within the target — with a maximum of 46ms for cold-start inference. The model successfully loads trained weights from `.pth` checkpoint files and produces FCFF growth forecasts that feed into the DCF calculation.

The architectural shift from V1 to V2 was motivated by practical experience with financial data. V1 used a 3-layer LSTM with MSE Loss, which proved sensitive to outliers — quarterly financial data frequently contains extreme values from one-off events such as restructuring charges, goodwill impairments, or pandemic-related write-downs. V2's adoption of Huber Loss (with delta=1.0) provides linear penalty for large errors rather than quadratic, producing more stable training convergence and more robust inference.

The consensus scoring system similarly evolved through practical evaluation. The original Random Forest ensemble component was deprecated after analysis revealed that a single feature (P/E ratio) accounted for 99.93% of the model's feature importance. This meant the RF was effectively functioning as a P/E filter masquerading as a multi-factor model. It was replaced with a transparent GARP score combining Forward P/E and PEG ratio, which provides genuine complementary signal alongside the LSTM-DCF fair value estimate.

### 4.4 System Testing

A three-tier testing strategy was employed to validate the system at different levels of granularity.

**Unit testing** (pytest): Covers agent tool functions, scoring calculations, data fetcher responses, and consensus scorer logic. All agent tools are tested to ensure they return strings rather than raising exceptions, following the LangChain tool pattern. The 5-Pillar scorer was unit-tested with known input data to verify that weight calculations, threshold boundaries, and classification logic (Buy/Hold/Watch) produce correct outputs. Edge cases including missing data fields and zero-value metrics were specifically covered.

**Integration testing**: End-to-end pipeline execution tests verify that stocks flow correctly through all three stages — from attention scan through qualification to curation — with appropriate database writes to MongoDB collections. API endpoint tests confirm that the 41 REST endpoints return correct response schemas and handle error conditions (invalid tickers, missing data, rate limit exhaustion) gracefully.

**Black-box testing**: Frontend user flows were tested across the primary interaction paths: screener browsing → stock preview → single-agent AI analysis → multi-agent analysis → risk profile questionnaire → filtered screener view. The cold-start banner was tested under Render.com's actual 30–90 second sleep/wake cycle to verify correct polling behaviour, timer accuracy, and auto-dismiss on backend response. Cross-browser testing confirmed localStorage caching persistence and TTL expiration across Chrome and Firefox.

### 4.5 Known Limitations

Acknowledging the system's boundaries is essential for honest evaluation and appropriate use:

**Qualitative blind spots**: Emetix cannot assess management quality, competitive moats, regulatory exposure, or industry disruption risk. A company may score excellently on quantitative metrics while facing an existential threat invisible to financial statements — for example, a patent cliff or management fraud. These qualitative factors frequently determine long-term outcomes but resist systematic quantification.

**Data limitations**: The system relies exclusively on publicly available data through free API tiers. This means no access to forward analyst consensus estimates, limited small-cap coverage (companies with thin trading volume often lack sufficient fundamental data), and inconsistent data quality for international ADRs. Finnhub's rate limit (60 requests/minute) constrains real-time sentiment retrieval, and Alpha Vantage's 25 calls/day limit requires careful request management during data collection.

**Model limitations**: The LSTM-DCF is trained on historical quarterly data and fundamentally cannot predict regime changes — events like COVID-19, geopolitical conflicts, or sudden regulatory shifts that invalidate the statistical relationships learned during training. Individual direction accuracy of approximately 31% clearly positions Emetix as a screening and ranking tool, not a predictive oracle.

**Deployment constraints**: Render.com's free tier introduces cold-start latency of 30–90 seconds after inactivity, which the Cold-Start Banner manages gracefully but cannot eliminate. This impacts first-time user experience and would need to be addressed for production-scale deployment.

### 4.6 Discussion

Emetix is explicitly designed as decision support, not financial advice. All AI-generated outputs carry appropriate disclaimers, and the platform's language consistently frames results as "analysis" and "insights" rather than "recommendations."

Compared to Bloomberg Terminal ($25,000/year), Emetix necessarily trades depth and breadth for accessibility. It cannot replicate real-time tick data, proprietary analyst estimates, fixed-income analysis, or the decades of institutional data that justify Bloomberg's price point. However, the core value proposition — systematic screening, quantitative scoring, and AI-generated analysis — is delivered at zero cost. For a retail investor deciding where to focus their limited research time, this represents a meaningful advancement over manually scanning financial statements.

The multi-agent architecture provides structural advantages over monolithic designs. Individual agents can be updated, retrained, or replaced without affecting other components. New analysis capabilities (e.g., ESG scoring, technical pattern recognition) can be added by creating new agent modules. The LangGraph framework's `create_react_agent` pattern proved well-suited to financial tool orchestration, where each agent wraps domain-specific calculations in a consistent interface.

The PRC framework represents the project's primary conceptual innovation. While its quantitative impact on portfolio returns has not been measured — doing so would require a longitudinal controlled study with real investors — its qualitative contribution is clear. By incorporating investor experience and emotional tolerance into screening criteria, the PRC transforms a generic screener into a personalised tool that protects less experienced investors from stocks that, while quantitatively "low risk," may be psychologically inappropriate for their specific situation. The privacy-first implementation via localStorage addresses legitimate concerns about financial data sensitivity while keeping the architecture simple.

From a user experience perspective, the platform achieves its core accessibility goal. Full AI-driven stock analysis completes in seconds — in contrast to the 2–4 hours typically required for manual DCF work — and is presented in plain-language summaries rather than raw financial jargon. The cold-start banner, LLM caching, and tabbed screener interface collectively ensure that the application remains responsive and intuitive, even when constrained by the free-tier infrastructure on which it runs.

---

## 5 Conclusions and Recommendations

### 5.1 Summary

This project designed, implemented, and deployed Emetix — a multi-agent AI platform that automates stock analysis for retail investors. The 3-stage Quality Growth Pipeline filters approximately 5,800 US equities to around 100 curated picks through attention scanning, 5-Pillar composite scoring, and Buy/Hold/Watch classification. The LSTM-DCF V2 model provides dynamic fair value estimation from quarterly fundamentals, and consensus scoring mitigates single-model bias. The Personal Risk Capacity framework introduces personalised risk assessment using a 5-tier emotional buffer system, moving beyond generic metrics to ask "is this stock risky for you?" rather than simply "is this stock risky?"

Backtesting showed 91% of yearly cohorts outperforming S&P 500 over 5-year holding periods, while acknowledging that individual prediction accuracy remains modest. The system's strength lies in systematic quality screening rather than price forecasting. Deployed freely on Vercel, Render.com, and MongoDB Atlas, the platform demonstrates that meaningful financial analysis automation is achievable at retail scale.

### 5.2 Implications and Future Work

The project demonstrates the viability of multi-agent AI for financial analysis democratisation. The modular architecture provides a replicable pattern for other data-intensive domains. The PRC approach to personalised risk could extend to bonds, ETFs, and other instruments.

Future development should prioritise real-time data streaming, international market expansion, a mobile application for broader reach, automated quarterly model retraining, and — most critically — a controlled user study with retail investors to empirically validate the PRC framework's impact on investment behaviour and portfolio outcomes.

---

## REFERENCES

[1] A. Damodaran, *Investment Valuation: Tools and Techniques for Determining the Value of Any Asset*, 3rd ed. Hoboken, NJ, USA: Wiley, 2012.

[2] E. F. Fama, "Efficient capital markets: A review of theory and empirical work," *J. Finance*, vol. 25, no. 2, pp. 383–417, May 1970.

[3] E. F. Fama and K. R. French, "Common risk factors in the returns on stocks and bonds," *J. Financial Econ.*, vol. 33, no. 1, pp. 3–56, Feb. 1993.

[4] P. Geertsema and H. Lu, "Machine learning for relative stock valuation," *J. Financial Data Sci.*, vol. 2, no. 3, pp. 125–143, 2020.

[5] D. Kahneman and A. Tversky, "Prospect theory: An analysis of decision under risk," *Econometrica*, vol. 47, no. 2, pp. 263–292, Mar. 1979.

[6] W. Li and V. W. S. Tam, "Deep learning enhanced DCF valuation in financial markets," *Expert Syst. Appl.*, vol. 187, p. 115897, Jan. 2025.

[7] R. Rekhi, A. Mehta, and S. S. Singh, "Machine learning applications in stock market prediction," in *Proc. Int. Conf. Comput. Intell. Data Sci.*, 2014, pp. 1–6.

[8] W. F. Sharpe, "Capital asset prices: A theory of market equilibrium under conditions of risk," *J. Finance*, vol. 19, no. 3, pp. 425–442, Sep. 1964.

[9] LangChain, "LangGraph: Multi-agent orchestration framework," 2025. [Online]. Available: https://langchain.com/docs/langgraph

[10] PyTorch Lightning, "PyTorch Lightning: Deep learning framework," 2025. [Online]. Available: https://pytorch-lightning.readthedocs.io

[11] Wikipedia, "2022 stock market decline," 2025. [Online]. Available: https://en.wikipedia.org/wiki/2022_stock_market_decline

[12] B. Elad, "Stock market participation statistics 2025," *CoinLaw*, Sep. 2025. [Online]. Available: https://coinlaw.io/stock-market-participation-statistics/

---

## FIGURES AND TABLES INDEX

| Item | Description | Notes |
|------|-------------|-------|
| Figure 1 | High-level 5-layer system architecture | Create diagram from Section 3.2 |
| Figure 2 | 3-Stage Pipeline funnel diagram | ~5,800 → ~200–400 → ~100–200 → ~100 |
| Figure 3 | LSTM-DCF V2 training pipeline | Quarterly data → 8-quarter sequences → LSTM → DCF |
| Figure 4 | Multi-agent orchestration flow | Supervisor routing to 8 specialist agents |
| Figure 5 | 5-Pillar Composite scoring radar chart | Visualise pillar weights and example scores |
| Figure 6 | PRC framework flow | Questionnaire → scores → emotional buffer → screener filter |
| Figure 7 | Screener UI with risk profile chip | Screenshot of deployed application |
| Table 1 | Technology stack | Section 3.2 |
| Table 2 | 5-Pillar scoring weights (v3.1) | Section 3.3 |
| Table 3 | Backtesting results by cohort | Section 4.1 |

*NB: Insert all figures into text boxes in the FinalReportPaperTemplate.docx. Set text box borders to zero before submission.*

---

**Total word count (body text): ~4,200 words** *(within ±10% of 4,000 target)*
