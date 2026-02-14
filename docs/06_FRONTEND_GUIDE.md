# 6. Frontend Guide

---

## Technology Stack

| Technology            | Version            | Purpose                      |
| --------------------- | ------------------ | ---------------------------- |
| Next.js               | 16.1.1             | React framework (App Router) |
| React                 | 19.2.3             | UI library                   |
| TypeScript            | 5                  | Type safety                  |
| Tailwind CSS          | v4                 | Utility-first styling        |
| shadcn/ui             | (Radix primitives) | Accessible UI components     |
| Recharts              | 3.6                | Data visualisation / charts  |
| @tanstack/react-query | 5.90               | Server state management      |
| react-hook-form       | 7.70               | Form handling                |
| zod                   | 4.3.5              | Schema validation            |
| next-themes           | 0.4.6              | Dark/light mode              |
| lucide-react          | 0.562              | Icon library                 |

---

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── layout.tsx          # Root layout (providers, theme)
│   │   ├── page.tsx            # Home page
│   │   ├── about/page.tsx      # About page
│   │   ├── screener/page.tsx   # Stock screener
│   │   ├── stock/[ticker]/page.tsx  # Stock detail (dynamic)
│   │   ├── pipeline/page.tsx   # Pipeline dashboard
│   │   └── profile/
│   │       └── risk-assessment/page.tsx  # Risk assessment
│   ├── components/
│   │   ├── ui/                 # shadcn/ui primitives
│   │   ├── layout/             # Header, Footer, Navigation
│   │   ├── charts/             # Recharts chart components
│   │   ├── stocks/             # Stock cards, tables, details
│   │   ├── pipeline/           # Pipeline stage views
│   │   ├── profile/            # Risk profile forms/results
│   │   └── risk-profile/       # Risk assessment components
│   ├── hooks/                  # Custom React hooks
│   ├── lib/
│   │   ├── api.ts              # API client (25 functions)
│   │   └── utils.ts            # Utility functions (cn(), etc.)
│   └── types/                  # TypeScript type definitions
├── public/                     # Static assets
├── package.json
├── tsconfig.json
├── next.config.ts
├── postcss.config.mjs
└── components.json             # shadcn/ui config
```

---

## Pages (6 Routes)

| Route                      | File                                   | Description                                          |
| -------------------------- | -------------------------------------- | ---------------------------------------------------- |
| `/`                        | `app/page.tsx`                         | Home — dashboard overview, market summary            |
| `/screener`                | `app/screener/page.tsx`                | Stock screener with filtering and search             |
| `/stock/[ticker]`          | `app/stock/[ticker]/page.tsx`          | Stock detail — charts, fundamentals, AI analysis     |
| `/pipeline`                | `app/pipeline/page.tsx`                | Pipeline dashboard — attention → qualified → curated |
| `/profile/risk-assessment` | `app/profile/risk-assessment/page.tsx` | Risk questionnaire and profile results               |
| `/about`                   | `app/about/page.tsx`                   | About Emetix, methodology explanation                |

---

## API Client (`src/lib/api.ts`)

Base URL: `process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"`

### Exported Functions (25)

| Function                            | Endpoint                                          | Category    |
| ----------------------------------- | ------------------------------------------------- | ----------- |
| `fetchWatchlist()`                  | `GET /api/summary`                                | Screener    |
| `fetchCategorizedWatchlist()`       | `GET /api/summary`                                | Screener    |
| `fetchStock(ticker)`                | `GET /api/stock/{ticker}`                         | Screener    |
| `fetchCharts(ticker)`               | `GET /api/charts/{ticker}`                        | Screener    |
| `fetchSectors()`                    | `GET /api/sectors`                                | Screener    |
| `compareStocks(tickers)`            | `GET /api/compare`                                | Screener    |
| `healthCheck()`                     | `GET /health`                                     | Health      |
| `assessRiskProfile(data)`           | `POST /api/risk-profile/assess`                   | Risk        |
| `getRiskProfile(id)`                | `GET /api/risk-profile/profile/{id}`              | Risk        |
| `getPositionSizing(req)`            | `POST /api/risk-profile/position-sizing`          | Risk        |
| `getSuitableStocks(id)`             | `GET /api/risk-profile/suitable-stocks`           | Risk        |
| `getRiskMethodology()`              | `GET /api/risk-profile/methodology`               | Risk        |
| `fetchPipelineAttention()`          | `GET /api/pipeline/attention`                     | Pipeline    |
| `fetchPipelineQualified()`          | `GET /api/pipeline/qualified`                     | Pipeline    |
| `fetchPipelineClassified()`         | `GET /api/pipeline/classified`                    | Pipeline    |
| `fetchCuratedWatchlist()`           | `GET /api/pipeline/curated`                       | Pipeline    |
| `fetchPipelineStock(ticker)`        | `GET /api/pipeline/stock/{ticker}`                | Pipeline    |
| `fetchPipelineSummary()`            | `GET /api/pipeline/summary`                       | Pipeline    |
| `triggerPipelineScan()`             | `POST /api/pipeline/trigger-scan`                 | Pipeline    |
| `fetchAIAnalysis(ticker)`           | `GET /api/analysis/stock/{ticker}`                | AI Analysis |
| `fetchQuickAnalysis(ticker)`        | `GET /api/analysis/stock/{ticker}/quick`          | AI Analysis |
| `fetchMultiAgentAnalysis(ticker)`   | `GET /api/multiagent/stock/{ticker}`              | Multi-Agent |
| `fetchSentimentAnalysis(ticker)`    | `GET /api/multiagent/stock/{ticker}/sentiment`    | Multi-Agent |
| `fetchFundamentalsAnalysis(ticker)` | `GET /api/multiagent/stock/{ticker}/fundamentals` | Multi-Agent |
| `fetchMLValuationAnalysis(ticker)`  | `GET /api/multiagent/stock/{ticker}/ml-valuation` | Multi-Agent |

### Key Exported Types

`Stock`, `WatchlistResponse`, `ForwardMetrics`, `ValuationStatus`, `ListCategory`, `StockDetailResponse`, `ChartResponse`, `SectorData`

---

## UI Components

### shadcn/ui Primitives (`components/ui/`)

Built on Radix UI:

| Component    | Radix Package                  |
| ------------ | ------------------------------ |
| Accordion    | `@radix-ui/react-accordion`    |
| Alert Dialog | `@radix-ui/react-alert-dialog` |
| Dialog       | `@radix-ui/react-dialog`       |
| Label        | `@radix-ui/react-label`        |
| Progress     | `@radix-ui/react-progress`     |
| Radio Group  | `@radix-ui/react-radio-group`  |
| Select       | `@radix-ui/react-select`       |
| Separator    | `@radix-ui/react-separator`    |
| Slider       | `@radix-ui/react-slider`       |
| Slot         | `@radix-ui/react-slot`         |
| Switch       | `@radix-ui/react-switch`       |
| Tabs         | `@radix-ui/react-tabs`         |
| Tooltip      | `@radix-ui/react-tooltip`      |

### Feature Components

| Directory                  | Components                                    |
| -------------------------- | --------------------------------------------- |
| `components/layout/`       | Header, Footer, Navigation, ThemeToggle       |
| `components/charts/`       | Price charts, volume charts (Recharts)        |
| `components/stocks/`       | Stock cards, comparison tables, detail panels |
| `components/pipeline/`     | Pipeline stage views, progress indicators     |
| `components/profile/`      | Risk questionnaire form, results display      |
| `components/risk-profile/` | Position sizing, suitability views            |

---

## State Management

- **Server state**: `@tanstack/react-query` — handles caching, refetching, loading/error states for all API calls
- **Form state**: `react-hook-form` + `zod` resolver for validated forms (risk assessment questionnaire)
- **Theme**: `next-themes` (dark/light mode toggle)
- **No global client state library** — component-level state where needed

---

## Development

### Prerequisites

- Node.js 18+
- npm or pnpm

### Setup

```bash
cd frontend
npm install
```

### Environment Variables

Create `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Run Development Server

```bash
npm run dev
```

Opens at `http://localhost:3000`.

### Build for Production

```bash
npm run build
```

### Path Aliases

Configured in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

All imports use `@/` prefix:

```typescript
import { fetchStock } from "@/lib/api";
import { Button } from "@/components/ui/button";
```

---

## Deployment

Deployed on **Vercel** with automatic deploys.

| Setting          | Value                   |
| ---------------- | ----------------------- |
| Root Directory   | `frontend`              |
| Framework        | Next.js (auto-detected) |
| Build Command    | `npm run build`         |
| Output Directory | `.next`                 |
| Node.js Version  | 18.x                    |

Environment variable on Vercel: `NEXT_PUBLIC_API_URL` = production backend URL.

See [07 — Deployment](07_DEPLOYMENT.md) for full deployment details.
