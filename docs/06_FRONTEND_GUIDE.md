# 6. Frontend Integration Guide

> **React/Next.js Implementation with Recommended Libraries**

---

## 🎨 Design Philosophy (Phase 2 - Jan 2026)

### Core Principles

| Principle          | Implementation                                        |
| ------------------ | ----------------------------------------------------- |
| **Modern & Clean** | Minimal UI, generous whitespace, subtle shadows       |
| **Intuitive**      | Simple navigation, clear CTAs, progressive disclosure |
| **Comfortable**    | Eye-friendly colors, dark/light theme toggle          |
| **Uncluttered**    | Focus on essential data, expandable details           |

### Design Inspirations

| Platform          | Borrowed Elements                                            |
| ----------------- | ------------------------------------------------------------ |
| **Stock Rover**   | Tabular dashboards, color-coded margins, linked detail pages |
| **WallStreetZen** | Simple lists with % below fair value, score gauges           |
| **GuruFocus**     | Advanced tables with expandable justifications               |
| **Finviz**        | Sector heatmaps, clean visual hierarchy                      |

### Color Coding Standards

| Metric                 | Green | Yellow | Red   |
| ---------------------- | ----- | ------ | ----- |
| **Margin of Safety**   | > 20% | 10-20% | < 10% |
| **Valuation Score**    | > 70  | 50-70  | < 50  |
| **Risk Level**         | LOW   | MEDIUM | HIGH  |
| **Direction Accuracy** | > 60% | 50-60% | < 50% |

### Theme Support

- **Light Mode**: Clean whites, soft grays, accessible contrast
- **Dark Mode**: Deep grays (#1a1a1a), muted accents, reduced eye strain
- **Toggle**: Persistent preference via localStorage

---

## 🛠️ Technology Stack

### Recommended Stack

| Category          | Technology      | Version | Purpose                         |
| ----------------- | --------------- | ------- | ------------------------------- |
| **Framework**     | Next.js         | 14.x    | React framework with App Router |
| **UI Components** | shadcn/ui       | latest  | Radix-based component library   |
| **Styling**       | TailwindCSS     | 3.4.x   | Utility-first CSS               |
| **Charts**        | Recharts        | 2.10.x  | React chart library             |
| **Data Fetching** | TanStack Query  | 5.x     | Server state management         |
| **State**         | Zustand         | 4.x     | Client state (watchlist)        |
| **Forms**         | React Hook Form | 7.x     | Form handling                   |
| **Validation**    | Zod             | 3.x     | Schema validation               |
| **HTTP Client**   | Axios           | 1.6.x   | API requests                    |
| **Icons**         | Lucide React    | 0.300.x | Icon library                    |
| **Date**          | date-fns        | 3.x     | Date formatting                 |

---

## 📦 Project Setup

### Initialize Next.js Project

```bash
npx create-next-app@latest emetix-frontend --typescript --tailwind --app --src-dir

cd emetix-frontend
```

### Install Dependencies

```bash
# Core UI
npm install @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-tabs
npm install class-variance-authority clsx tailwind-merge

# Data & State
npm install @tanstack/react-query axios zustand

# Charts
npm install recharts

# Forms & Validation
npm install react-hook-form @hookform/resolvers zod

# Utilities
npm install date-fns lucide-react

# Dev Dependencies
npm install -D @types/node @types/react @types/react-dom
```

### Install shadcn/ui

```bash
npx shadcn-ui@latest init

# Add components as needed
npx shadcn-ui@latest add button card table badge tabs
```

---

## 📁 Project Structure

```
src/
├── app/
│   ├── layout.tsx              # Root layout with providers
│   ├── page.tsx                # Top Picks (home) - curated recommendations
│   ├── globals.css             # Global styles
│   │
│   ├── screener/
│   │   └── page.tsx            # Stock Screener - full analysis universe (500+)
│   │
│   ├── stock/
│   │   └── [ticker]/
│   │       └── page.tsx        # Stock detail page with AI analysis
│   │
│   ├── profile/
│   │   └── risk-assessment/
│   │       └── page.tsx        # Personal Risk Capacity questionnaire
│   │
│   └── about/
│       └── page.tsx            # About page with project info
│
├── components/
│   ├── ui/                     # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── table.tsx
│   │   ├── input.tsx           # Text input component
│   │   ├── select.tsx          # Dropdown select component
│   │   ├── progress.tsx        # Score gauges
│   │   ├── slider.tsx          # Questionnaire inputs
│   │   ├── tooltip.tsx         # Position sizing hints
│   │   └── switch.tsx          # Theme toggle
│   │
│   ├── layout/
│   │   ├── Navbar.tsx          # Top navigation (Top Picks, Screener, Risk Profile, About)
│   │   ├── Footer.tsx
│   │   └── ThemeToggle.tsx     # Dark/light mode toggle
│   │
│   ├── charts/
│   │   ├── PriceChart.tsx      # 1Y/5Y price chart
│   │   ├── TechnicalChart.tsx  # MA50/MA200 overlay
│   │   ├── SectorPieChart.tsx  # Sector distribution
│   │   └── PillarRadarChart.tsx  # 4-pillar scoring visualization
│   │
│   ├── stocks/
│   │   ├── WatchlistTable.tsx  # Legacy watchlist table
│   │   ├── AIAnalysisPanel.tsx # AI-powered stock analysis
│   │   └── MultiAgentAnalysisPanel.tsx  # Multi-agent deep analysis
│   │
│   ├── pipeline/
│   │   ├── PipelineDashboard.tsx  # Reusable pipeline component
│   │   └── PipelineBadges.tsx     # Classification & score badges
│   │
│   └── risk-profile/
│       ├── RiskQuestionnaire.tsx   # 7-question form
│       ├── RiskAssessmentModal.tsx # First-time user modal
│       ├── ProfileResults.tsx      # Profile summary
│       └── SuitabilityBadge.tsx    # Stock suitability indicator
│
├── hooks/
│   ├── use-stocks.ts           # Watchlist, curated, stock hooks
│   ├── use-pipeline.ts         # Pipeline API hooks (classified, curated)
│   ├── useRiskProfile.ts       # Risk assessment (client-side)
│   └── useCharts.ts            # Chart data hooks
│
├── lib/
│   ├── api.ts                  # API client with all endpoints
│   ├── utils.ts                # Helper functions
│   └── risk-scoring.ts         # Client-side risk calculations
│
└── types/
    ├── api.ts                  # TypeScript interfaces
    └── risk-profile.ts         # Risk profile types
```

---

## 🏠 Main Pages Overview

### Top Picks (Home Page - `/`)

**Data Source**: `GET /api/pipeline/curated`

**Features**:

- Curated stock recommendations (Buy + Hold categories)
- Analyst-like justifications for each pick
- Conviction badges (Strong/Moderate/Monitor)
- Risk profile integration with suitability indicators
- CTA for users without risk profile to take assessment
- Quick stats (Buy signals, Hold count, Avg score, Universe size)

**Risk Profile Integration**:

- Shows suitability badge (✓/⚠/✗) next to each stock
- Suitability calculated based on user's profile type, beta range, and required MoS
- Prompts users without profile to complete assessment

### Stock Screener (`/screener`)

**Data Source**: `GET /api/pipeline/classified`

**Features**:

- Full 500+ stock universe with filters
- Search by ticker or company name
- Filter by sector, classification (Buy/Hold/Watch)
- Minimum score filter (60-80)
- Pagination (25 stocks per page)
- Sortable columns
- Quick methodology tooltips

### Stock Detail (`/stock/[ticker]`)

**Data Source**: `GET /api/pipeline/stock/{ticker}`, `GET /api/screener/charts/{ticker}`

**Features**:

- Price chart with technical indicators
- 4-pillar scoring breakdown
- AI Analysis tab with multi-agent deep analysis
- Fair value comparison (LSTM vs Traditional)
- Momentum indicators
- Position sizing calculator (if risk profile exists)

### Risk Profile (`/profile/risk-assessment`)

**Data Source**: Client-side calculation (no API)

**Features**:

- 7-question progressive form
- Local storage persistence
- Profile types: Conservative, Moderate, Aggressive
- Suitable beta range calculation
- Required margin of safety threshold
- Integration with Top Picks suitability badges

---

## 🎯 Phase 2 Components (Personal Risk Capacity)

### P2-FE-01: Risk Questionnaire Page

**Route**: `/profile/risk-assessment`

**Features**:

- 7-question progressive form
- Slider inputs for numeric values
- Radio groups for categorical choices
- Real-time validation with Zod
- Progress indicator

**Questions**:

1. Experience Level (dropdown: first_time → professional)
2. Investment Horizon (dropdown: short → very_long)
3. Emergency Fund Months (slider: 0-36)
4. Monthly Investment % (slider: 0-100)
5. Max Tolerable Loss % (slider: 0-100)
6. Panic Sell Response (5 radio options)
7. Volatility Comfort (1-5 star rating)

---

### P2-FE-02: Profile Summary Card

**Display Elements**:

- Risk Capacity gauge (0-100)
- Risk Tolerance gauge (0-100)
- Emotional Buffer factor (1.0x - 2.0x)
- Suitable beta range
- Adjusted MoS threshold
- Personalized recommendations list

---

### P2-FE-03: Enhanced Watchlist

**Enhancements**:

- Profile filter toggle (show only suitable stocks)
- Suitability indicator column (✓ / ⚠ / ✗)
- Color-coded margin of safety
- Beta comparison to profile range
- Header showing adjusted MoS threshold

---

### P2-FE-04: Position Sizing Tooltip

**Trigger**: Hover/click on info icon (ℹ️) next to signal column

**Tooltip Content**:

- Max position % of portfolio
- Max dollar amount
- Max shares to buy
- Risk factors list
- Methodology explanation

---

## 🔗 API Integration

### API Client Setup

```typescript
// src/lib/api.ts
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// API endpoints
export const endpoints = {
  watchlist: (n = 10, rescan = false) =>
    `/api/screener/watchlist?n=${n}&rescan=${rescan}`,
  watchlistSimple: (n = 10) => `/api/screener/watchlist/simple?n=${n}`,
  stock: (ticker: string) => `/api/screener/stock/${ticker}`,
  compare: (tickers: string[]) =>
    `/api/screener/compare?tickers=${tickers.join(",")}`,
  charts: (ticker: string) => `/api/screener/charts/${ticker}`,
  sectors: () => "/api/screener/sectors",
  sectorStocks: (sector: string, n = 10) =>
    `/api/screener/sectors/${encodeURIComponent(sector)}?n=${n}`,
  methodology: () => "/api/screener/methodology",
};
```

### React Query Setup

```typescript
// src/lib/queryClient.ts
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 30 * 60 * 1000, // 30 minutes (formerly cacheTime)
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});
```

### Provider Setup

```typescript
// src/app/providers.tsx
"use client";

import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}
```

---

## 🪝 Custom Hooks

### useWatchlist Hook

```typescript
// src/hooks/useWatchlist.ts
import { useQuery } from "@tanstack/react-query";
import { api, endpoints } from "@/lib/api";
import type { WatchlistResponse } from "@/types/api";

export function useWatchlist(n = 10, rescan = false) {
  return useQuery<WatchlistResponse>({
    queryKey: ["watchlist", n, rescan],
    queryFn: async () => {
      const { data } = await api.get(endpoints.watchlist(n, rescan));
      return data;
    },
  });
}
```

### useStock Hook

```typescript
// src/hooks/useStock.ts
import { useQuery } from "@tanstack/react-query";
import { api, endpoints } from "@/lib/api";
import type { StockResponse } from "@/types/api";

export function useStock(ticker: string) {
  return useQuery<StockResponse>({
    queryKey: ["stock", ticker],
    queryFn: async () => {
      const { data } = await api.get(endpoints.stock(ticker));
      return data;
    },
    enabled: !!ticker,
  });
}
```

### useCharts Hook

```typescript
// src/hooks/useCharts.ts
import { useQuery } from "@tanstack/react-query";
import { api, endpoints } from "@/lib/api";
import type { ChartResponse } from "@/types/api";

export function useCharts(ticker: string) {
  return useQuery<ChartResponse>({
    queryKey: ["charts", ticker],
    queryFn: async () => {
      const { data } = await api.get(endpoints.charts(ticker));
      return data;
    },
    enabled: !!ticker,
  });
}
```

---

## 📊 Chart Components

### Price Chart with Recharts

```typescript
// src/components/charts/PriceChart.tsx
"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { format } from "date-fns";

interface PriceChartProps {
  data: Array<{ date: string; price: number; volume?: number }>;
  period: "1Y" | "5Y";
}

export function PriceChart({ data, period }: PriceChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tickFormatter={(date) =>
            format(new Date(date), period === "1Y" ? "MMM" : "yyyy")
          }
        />
        <YAxis domain={["auto", "auto"]} />
        <Tooltip
          labelFormatter={(date) => format(new Date(date), "MMM dd, yyyy")}
          formatter={(value: number) => [`$${value.toFixed(2)}`, "Price"]}
        />
        <Line
          type="monotone"
          dataKey="price"
          stroke="#2563eb"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

### Technical Chart with MA Overlays

```typescript
// src/components/charts/TechnicalChart.tsx
"use client";

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface TechnicalChartProps {
  data: Array<{
    date: string;
    price: number;
    ma50: number | null;
    ma200: number | null;
  }>;
}

export function TechnicalChart({ data }: TechnicalChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={data}>
        <XAxis dataKey="date" />
        <YAxis domain={["auto", "auto"]} />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="price"
          stroke="#2563eb"
          name="Price"
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="ma50"
          stroke="#f59e0b"
          name="MA50"
          dot={false}
          strokeDasharray="5 5"
        />
        <Line
          type="monotone"
          dataKey="ma200"
          stroke="#10b981"
          name="MA200"
          dot={false}
          strokeDasharray="5 5"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
```

---

## 🎨 UI Components

### Stock Card

```typescript
// src/components/cards/StockCard.tsx
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface StockCardProps {
  rank: number;
  ticker: string;
  company: string;
  sector: string;
  price: number;
  fairValue: number;
  marginOfSafety: number;
  score: number;
  recommendation: string;
}

export function StockCard({
  rank,
  ticker,
  company,
  sector,
  price,
  fairValue,
  marginOfSafety,
  score,
  recommendation,
}: StockCardProps) {
  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case "STRONG BUY":
        return "bg-green-600";
      case "BUY":
        return "bg-green-500";
      case "ACCUMULATE":
        return "bg-blue-500";
      case "HOLD":
        return "bg-yellow-500";
      case "REDUCE":
        return "bg-orange-500";
      case "SELL":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <span className="text-sm text-muted-foreground">#{rank}</span>
          <h3 className="text-xl font-bold">{ticker}</h3>
          <p className="text-sm text-muted-foreground">{company}</p>
        </div>
        <Badge className={getRecommendationColor(recommendation)}>
          {recommendation}
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-muted-foreground">Price</p>
            <p className="text-lg font-semibold">${price.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Fair Value</p>
            <p className="text-lg font-semibold">${fairValue.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Margin of Safety</p>
            <p
              className={`text-lg font-semibold ${
                marginOfSafety > 0 ? "text-green-600" : "text-red-600"
              }`}
            >
              {marginOfSafety > 0 ? "+" : ""}
              {marginOfSafety.toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Score</p>
            <p className="text-lg font-semibold">{score.toFixed(1)}/100</p>
          </div>
        </div>
        <Badge variant="outline" className="mt-4">
          {sector}
        </Badge>
      </CardContent>
    </Card>
  );
}
```

---

## 📝 TypeScript Interfaces

```typescript
// src/types/api.ts

export interface WatchlistStock {
  rank: number;
  ticker: string;
  company_name: string;
  sector: string;
  current_price: number;
  fair_value: number;
  margin_of_safety: number;
  valuation_score: number;
  recommendation: string;
  assessment: string;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  justification: string;
}

export interface WatchlistResponse {
  status: string;
  timestamp: string;
  model_info: {
    lstm_enabled: boolean;
    lstm_model: string | null;
    valuation_method: string;
  };
  scan_params: {
    universe_size: number;
    min_market_cap: number;
    max_pe_ratio: number;
    max_debt_equity: number;
  };
  benchmark_info: {
    use_dynamic_benchmarks: boolean;
    dynamic_sectors: number;
    total_sectors: number;
    benchmark_coverage: string;
  };
  summary: {
    results_count: number;
    avg_valuation_score: number;
    avg_margin_of_safety: number;
    sector_distribution: Record<string, number>;
  };
  sector_benchmarks: Record<string, SectorBenchmark>;
  watchlist: WatchlistStock[];
}

export interface SectorBenchmark {
  avg_pe: number;
  avg_pb: number;
  avg_roe: number;
  avg_margin: number;
  sample_size?: number;
  source?: "dynamic" | "default";
}

export interface ChartDataPoint {
  date: string;
  price: number;
  volume?: number;
  ma50?: number | null;
  ma200?: number | null;
}

export interface ChartResponse {
  status: string;
  data: {
    ticker: string;
    company_name: string;
    charts: {
      price_1y: ChartDataPoint[];
      price_5y: ChartDataPoint[];
      technical: ChartDataPoint[];
    };
    metrics: {
      current_price: number;
      high_52w: number;
      low_52w: number;
      pe_ratio: number;
      market_cap: number;
    };
  };
}
```

---

## 🌐 Environment Configuration

### Environment Variables

```env
# .env.local

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Production Configuration

```env
# .env.production
NEXT_PUBLIC_API_URL=https://api.emetix.onrender.com
```

---

## 🗄️ Data Storage Architecture

Emetix uses a **hybrid client-side + cloud storage** approach for simplicity (no user auth required):

### localStorage (Client-Side)

Risk profiles are stored in the browser's localStorage for instant access:

```typescript
// src/hooks/useRiskProfile.ts

// Keys used for localStorage
const PROFILE_ID_KEY = "emetix_profile_id";
const PROFILE_DATA_KEY = "emetix_risk_profile";

// Get stored profile
function getStoredProfile(): RiskProfileResponse | null {
  if (typeof window === "undefined") return null;
  try {
    const data = localStorage.getItem(PROFILE_DATA_KEY);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
}

// Save profile after questionnaire submission
function saveProfile(profile: RiskProfileResponse): void {
  localStorage.setItem(PROFILE_ID_KEY, profile.profile_id);
  localStorage.setItem(PROFILE_DATA_KEY, JSON.stringify(profile));
}

// Clear profile
export function clearStoredProfile(): void {
  localStorage.removeItem(PROFILE_ID_KEY);
  localStorage.removeItem(PROFILE_DATA_KEY);
}
```

### MongoDB Atlas (Server-Side)

Watchlists and strategies are stored in MongoDB Atlas via the `/api/storage/` endpoints:

```typescript
// Create a watchlist
const response = await fetch(`${API_URL}/api/storage/watchlists`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    name: "My Low-Risk Picks",
    tickers: ["AAPL", "MSFT", "JNJ"],
    is_public: false,
  }),
});

// Fetch watchlists (session-based, no auth needed)
const watchlists = await fetch(`${API_URL}/api/storage/watchlists`);
```

### Why No User Auth?

For an FYP thesis project, we prioritize:

- **Simplicity**: No JWT, no OAuth, no session management
- **Speed**: localStorage is instant, MongoDB is fast
- **Privacy**: Risk profiles stay on user's device
- **Thesis Focus**: Core value is ML + Risk Framework, not auth

````

---

## 🪶 Lightweight Alternative Stack

For a minimal footprint, use native fetch and React state instead of heavy libraries:

### Minimal Dependencies

```bash
# Essential only
npm install recharts lucide-react
````

### Native Fetch (No Axios/TanStack)

```typescript
// src/lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function fetchWatchlist(n = 10) {
  const res = await fetch(`${API_URL}/api/screener/watchlist?n=${n}`);
  if (!res.ok) throw new Error("Failed to fetch");
  return res.json();
}

export async function fetchStock(ticker: string) {
  const res = await fetch(`${API_URL}/api/screener/stock/${ticker}`);
  if (!res.ok) throw new Error("Failed to fetch");
  return res.json();
}
```

### Simple Data Hook (No TanStack)

```typescript
// src/hooks/useWatchlist.ts
import { useState, useEffect } from "react";
import { fetchWatchlist } from "@/lib/api";

export function useWatchlist(n = 10) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    fetchWatchlist(n)
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [n]);

  return { data, loading, error };
}
```

### Minimal Component

```tsx
// Lightweight card without shadcn
export function StockCard({ stock }: { stock: Stock }) {
  return (
    <div className="border rounded-lg p-4 hover:shadow-md transition">
      <h3 className="font-bold text-lg">{stock.ticker}</h3>
      <p className="text-gray-600">{stock.company_name}</p>
      <div className="flex justify-between mt-2">
        <span>${stock.current_price.toFixed(2)}</span>
        <span className={stock.upside > 0 ? "text-green-600" : "text-red-600"}>
          {stock.upside > 0 ? "+" : ""}
          {stock.upside.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
```

---

## 🚀 Running the Frontend

```bash
# Development
npm run dev

# Build
npm run build

# Production
npm start
```

---

_Next: [7. Deployment Guide](./07_DEPLOYMENT.md)_
