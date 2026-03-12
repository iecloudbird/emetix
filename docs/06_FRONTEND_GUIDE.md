# 6. Frontend Guide

---

## Technology Stack

| Technology            | Version            | Purpose                           |
| --------------------- | ------------------ | --------------------------------- |
| Next.js               | 16.1.1             | React framework (App Router)      |
| React                 | 19.2.3             | UI library                        |
| TypeScript            | 5                  | Type safety                       |
| Tailwind CSS          | v4                 | Utility-first styling             |
| shadcn/ui             | (Radix primitives) | Accessible UI components          |
| Recharts              | 3.6                | Data visualisation / charts       |
| @tanstack/react-query | 5.90               | Server state management & caching |
| react-hook-form       | 7.70               | Form handling                     |
| zod                   | 4.3.5              | Schema validation                 |
| next-themes           | 0.4.6              | Dark/light mode                   |
| lucide-react          | 0.562              | Icon library                      |

---

## Project Structure

```
frontend/
├── src/
│   ├── app/                          # Next.js App Router pages
│   │   ├── layout.tsx                # Root layout (providers, ColdStartBanner, Navbar)
│   │   ├── providers.tsx             # Theme + react-query providers
│   │   ├── page.tsx                  # Home — dashboard overview
│   │   ├── about/page.tsx            # About page
│   │   ├── screener/page.tsx         # Stock screener (filters, preview, risk profile)
│   │   ├── stock/[ticker]/page.tsx   # Stock detail (charts, AI analysis)
│   │   ├── pipeline/page.tsx         # Pipeline dashboard (3 stages)
│   │   └── risk-assessment/page.tsx  # Risk questionnaire & profile results
│   ├── components/
│   │   ├── ui/                       # shadcn/ui primitives (Radix-based)
│   │   ├── layout/                   # Navbar, CommandPalette, ColdStartBanner, Theme
│   │   ├── charts/                   # Recharts chart components
│   │   ├── stocks/                   # AIAnalysisPanel, MultiAgentAnalysisPanel, StockCard
│   │   ├── screener/                 # StockPreviewPanel (AI instant insights)
│   │   ├── pipeline/                 # Pipeline stage views
│   │   ├── profile/                  # Risk questionnaire form
│   │   └── risk-profile/             # ProfileResults, PositionSizing, SuitabilityBadge
│   ├── hooks/
│   │   ├── use-stocks.ts             # Stock data hooks
│   │   ├── use-pipeline.ts           # Pipeline data hooks
│   │   └── useRiskProfile.ts         # localStorage risk profile hook
│   ├── lib/
│   │   ├── api.ts                    # API client (25+ typed functions)
│   │   └── utils.ts                  # Utility functions (cn(), etc.)
│   └── types/                        # TypeScript type definitions
├── public/                           # Static assets
├── package.json
├── tsconfig.json
└── next.config.ts
```

---

## Pages (6 Routes)

| Route              | File                           | Description                                            |
| ------------------ | ------------------------------ | ------------------------------------------------------ |
| `/`                | `app/page.tsx`                 | Dashboard — market summary, top picks, quick stats     |
| `/screener`        | `app/screener/page.tsx`        | Stock screener with filters, preview panel, risk match |
| `/stock/[ticker]`  | `app/stock/[ticker]/page.tsx`  | Stock detail — charts, fundamentals, AI analysis tabs  |
| `/pipeline`        | `app/pipeline/page.tsx`        | Pipeline dashboard — attention → qualified → curated   |
| `/risk-assessment` | `app/risk-assessment/page.tsx` | Risk questionnaire and personalised profile results    |
| `/about`           | `app/about/page.tsx`           | About Emetix, methodology, technology explanation      |

---

## Key UX Features (Thesis-Worthy)

### 1. Command Palette (Cmd+K / Ctrl+K)

**Component**: `components/layout/CommandPalette.tsx`

A global quick-search dialog accessible from anywhere via keyboard shortcut. Users type a stock ticker or company name and navigate directly to the stock detail page — modelled after VS Code's command palette.

- Searches across all ~5,800 stocks in the universe
- Keyboard-first navigation (arrow keys + Enter)
- Debounced input with clean, minimal design
- Accessible from the Navbar search button

### 2. AI Stock Preview Panel

**Component**: `components/screener/StockPreviewPanel.tsx`

A side panel in the screener that shows an AI-generated analysis when a stock row is clicked — no page navigation needed.

- **AI Headline**: One-line Gemini-generated insight (e.g., "Strong quality compounder trading at moderate discount")
- **About the Company**: Real company description fetched from Yahoo Finance (`longBusinessSummary`), truncated to 2 sentences with overflow detection and expand/collapse toggle
- **Key Metrics Grid**: Composite score, classification, margin of safety, beta, P/E
- **Strengths & Weaknesses**: AI-identified bullet points
- **Visual emphasis**: `border-x-4` coloured by classification (green = Buy, blue = Hold, amber = Watch)

### 3. Risk Profile Integration in Screener

**Hook**: `hooks/useRiskProfile.ts` (localStorage-based)

After completing the risk assessment questionnaire, the user's risk profile is saved to localStorage. The screener reads this profile and offers a **Risk Profile** toggle chip in the filter bar:

- **Active**: Filters stocks to match user's beta tolerance and required margin of safety
- **Inactive**: Shows all stocks with a subtle tip encouraging activation
- **Pulse animation**: The chip gently pulses when inactive to draw attention
- **Inline indicator**: When active, "Filtered: Beta ≤ X · MoS ≥ Y%" appears in results summary
- **Null-beta handling**: Stocks with no beta data are included (not excluded)
- **Zero URL coupling**: Reads directly from localStorage — no query params needed

### 4. LLM Analysis Caching (localStorage)

**Components**: `AIAnalysisPanel.tsx`, `MultiAgentAnalysisPanel.tsx`

LLM API calls (Gemini) are expensive and slow. The frontend caches results in localStorage with time-to-live (TTL):

| Cache                | TTL            | Key Prefix             | Purpose                                     |
| -------------------- | -------------- | ---------------------- | ------------------------------------------- |
| AI (Deep) Analysis   | 2 hours        | `emetix_llm_analysis_` | Single-agent Gemini analysis                |
| Multi-Agent Analysis | 1 hour (local) | `emetix_multiagent_`   | Client cache; server-side 8h TTL in MongoDB |

- Cache is checked before API calls — instant results on repeat visits
- Uses React `useMemo` for lazy initialisation (avoids React 19 `setState`-in-effect warnings)
- Caching operates silently with no visible UI indicators — clean UX

### 5. Cold-Start Awareness Banner

**Component**: `components/layout/ColdStartBanner.tsx`

The backend is hosted on Render's free tier, which sleeps after 15 minutes of inactivity. This banner provides honest, user-friendly context:

- **Auto-detects** backend unavailability by polling `/health` every 5 seconds
- **Amber state**: "Backend is waking up..." with live elapsed timer and honest explanation (student FYP, free hosting)
- **Green confirmation**: "Backend is online — you're all set!" auto-hides after 3 seconds
- **Dismiss (×) button**: Users can close the banner at any time
- **Production only**: Never renders in local development (`process.env.NODE_ENV` check)
- **Max poll time**: Stops after 3 minutes to avoid infinite requests

### 6. Stock Screener Features

**Page**: `app/screener/page.tsx`

| Feature              | Description                                         |
| -------------------- | --------------------------------------------------- |
| Search               | Real-time filter by ticker or company name          |
| Sector Filter        | Dropdown to filter by sector                        |
| Min Score Filter     | Dropdown to set composite score threshold           |
| Risk Profile Toggle  | One-click chip to filter by personal risk profile   |
| Buy / Hold / Watch   | Classification-based tab navigation with counts     |
| Pagination           | Client-side with prev/next and page number buttons  |
| Click-to-Preview     | Click any stock row to open AI preview panel        |
| Results Summary      | Filtered/total counts with Buy/Hold/Watch breakdown |
| Methodology Info Bar | Inline scoring criteria explanation with tooltips   |
| Last Updated         | Shows when screener data was last refreshed         |

### 7. Stock Detail Page

**Page**: `app/stock/[ticker]/page.tsx`

| Tab / Section        | Description                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| Overview             | Price, key metrics, classification badge                                                        |
| Charts               | Interactive Recharts price/volume visualisation                                                 |
| Financials           | Analyst-style infographic (revenue bars, margins, multiple compression, beat-down detection)    |
| Fundamentals         | Detailed metrics with collapsible extended view                                                 |
| AI Analysis          | Gemini-powered deep analysis (cached 2 hrs)                                                     |
| Multi-Agent Analysis | Data-first analysis: enriched narratives + 1 LLM synthesis (server-cached 8h, client-cached 1h) |

### 8. Mobile Responsive Navigation

**Component**: `components/layout/Navbar.tsx`

- Hamburger menu on mobile with slide-out drawer
- Desktop: Full navigation bar with all routes + Cmd+K search trigger

---

## State Management

| Concern      | Solution                                                             |
| ------------ | -------------------------------------------------------------------- |
| Server state | `@tanstack/react-query` — caching, refetching, loading/error states  |
| Form state   | `react-hook-form` + `zod` resolver for risk assessment questionnaire |
| Theme        | `next-themes` (dark/light toggle, persisted)                         |
| Risk profile | `useLocalRiskProfile` hook — localStorage with structured read/write |
| LLM cache    | Direct localStorage with TTL-based expiry                            |
| Client state | Component-level `useState` / `useMemo` — no global state library     |

---

## API Client (`src/lib/api.ts`)

Base URL: `process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"`

### Key Exported Types

| Type                    | Fields                                                                 |
| ----------------------- | ---------------------------------------------------------------------- |
| `Stock`                 | ticker, company_name, current_price, composite_score, beta, sector     |
| `QuickAnalysisResponse` | status, ticker, summary (headline, score, classification, description) |
| `ForwardMetrics`        | forward_pe, forward_eps, peg_ratio, price_target                       |
| `ValuationStatus`       | fair_value, margin_of_safety, model_used                               |

### Exported Functions (25+)

| Category    | Functions                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------- |
| Screener    | `fetchWatchlist`, `fetchStock`, `fetchCharts`, `fetchSectors`, `compareStocks`                |
| Pipeline    | `fetchPipelineAttention/Qualified/Classified`, `fetchCuratedWatchlist`, `triggerPipelineScan` |
| AI Analysis | `fetchAIAnalysis`, `fetchQuickAnalysis`                                                       |
| Multi-Agent | `fetchMultiAgentAnalysis`, `fetchSentimentAnalysis`, `fetchMLValuationAnalysis`               |
| Risk        | `assessRiskProfile`, `getRiskProfile`, `getPositionSizing`, `getSuitableStocks`               |
| Health      | `healthCheck`                                                                                 |

---

## Development

### Setup

```bash
cd frontend
npm install
```

### Environment Variables (`frontend/.env.local`)

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Commands

```bash
npm run dev      # Development server at http://localhost:3000
npm run build    # Production build
```

### Path Aliases

All imports use `@/` prefix (configured in `tsconfig.json`):

```typescript
import { fetchStock } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { useLocalRiskProfile } from "@/hooks/useRiskProfile";
```

---

## Deployment

Deployed on **Vercel** with automatic deploys from `main` branch.

| Setting         | Value                   |
| --------------- | ----------------------- |
| Root Directory  | `frontend`              |
| Framework       | Next.js (auto-detected) |
| Build Command   | `npm run build`         |
| Node.js Version | 18.x                    |
| Env Variable    | `NEXT_PUBLIC_API_URL`   |

See [07 — Deployment](07_DEPLOYMENT.md) for full details.
