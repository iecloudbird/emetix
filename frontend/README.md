# Emetix Frontend

Next.js 16 web application for the Emetix stock analysis platform.

---

## Tech Stack

| Technology | Version | Purpose |
| ---------- | ------- | ------- |
| Next.js | 16.1.1 | React framework (App Router) |
| React | 19.2.3 | UI library |
| TypeScript | 5 | Type safety |
| Tailwind CSS | v4 | Styling |
| shadcn/ui | Radix primitives | Accessible UI components |
| Recharts | 3.6 | Charts & data visualisation |
| @tanstack/react-query | 5.90 | Server state management |
| react-hook-form + zod | 7.70 / 4.3 | Form handling & validation |

---

## Pages

| Route | Description |
| ----- | ----------- |
| `/` | Home — dashboard overview |
| `/screener` | Stock screener with search & filters |
| `/stock/[ticker]` | Stock detail — charts, fundamentals, AI analysis |
| `/pipeline` | Pipeline dashboard (attention → qualified → curated) |
| `/profile/risk-assessment` | Risk questionnaire & profile results |
| `/about` | About Emetix & methodology |

---

## Setup

```bash
npm install

# Create .env.local
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local

# Development
npm run dev          # http://localhost:3000

# Production build
npm run build
```

---

## Project Structure

```
src/
├── app/              # Next.js App Router pages
├── components/
│   ├── ui/           # shadcn/ui primitives (Radix)
│   ├── layout/       # Header, Footer, Navigation
│   ├── charts/       # Price & volume charts (Recharts)
│   ├── stocks/       # Stock cards, tables, detail panels
│   ├── pipeline/     # Pipeline stage views
│   └── profile/      # Risk assessment forms & results
├── hooks/            # Custom React hooks
├── lib/
│   ├── api.ts        # API client (25 exported functions)
│   └── utils.ts      # Utility functions
└── types/            # TypeScript type definitions
```

---

## API Client

All backend communication goes through `src/lib/api.ts`.

Base URL: `process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"`

Imports use the `@/` path alias (mapped to `./src/`):

```typescript
import { fetchStock } from "@/lib/api";
```

---

## Deployment

Deployed on **Vercel** with root directory set to `frontend`.

See [docs/07_DEPLOYMENT.md](../docs/07_DEPLOYMENT.md) for full details.
