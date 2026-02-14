# 7. Deployment Guide

---

## Architecture

```
┌────────────────────────┐          ┌─────────────────────────┐
│  Vercel                │  HTTPS   │  Render.com             │
│  (Frontend)            │ ◄──────► │  (Backend API)          │
│  Next.js 16.1.1        │          │  FastAPI + Uvicorn      │
│  emetix-woad.vercel.app│          │  Python 3.10            │
└────────────────────────┘          └───────────┬─────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │  MongoDB Atlas           │
                                    │  (Cloud Database)        │
                                    └─────────────────────────┘
```

---

## Backend Deployment (Render.com)

### Configuration

| Setting       | Value                                                 |
| ------------- | ----------------------------------------------------- |
| Service type  | Web Service                                           |
| Name          | `emetix-backend`                                      |
| Runtime       | Python 3.10                                           |
| Plan          | Free                                                  |
| Build command | `pip install -r requirements.txt`                     |
| Start command | `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT` |
| Port          | `8000` (set via Render dashboard)                     |

### Procfile

```
web: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

### render.yaml

```yaml
services:
  - type: web
    name: emetix-backend
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.10"
      - key: PORT
        value: "8000"
      - key: ENVIRONMENT
        value: production
```

### Environment Variables (Set in Render Dashboard)

| Variable                | Required | Description                                       |
| ----------------------- | -------- | ------------------------------------------------- |
| `GOOGLE_GEMINI_API_KEY` | **Yes**  | Primary LLM API key (Google AI Studio)            |
| `GROQ_API_KEY`          | No       | Fallback LLM API key                              |
| `FINNHUB_API_KEY`       | No       | Finnhub financial data                            |
| `ALPHA_VANTAGE_API_KEY` | No       | Alpha Vantage financial data                      |
| `NEWS_API_KEY`          | No       | NewsAPI for sentiment                             |
| `MONGODB_URI`           | **Yes**  | MongoDB Atlas connection string                   |
| `MONGODB_DATABASE`      | No       | Database name (default: `emetix_pipeline`)        |
| `LLM_PROVIDER`          | No       | `gemini` (default) / `groq` / `auto`              |
| `PORT`                  | No       | Server port (default: `5000`, Render sets `8000`) |
| `LOG_LEVEL`             | No       | `INFO` (default)                                  |

### CORS

Production CORS is configured in `render.yaml` to allow requests from `https://emetix-woad.vercel.app`.

In `src/api/app.py`, CORS middleware allows all origins (`"*"`) in development. For production, the Render.com headers restrict to the Vercel domain.

---

## Frontend Deployment (Vercel)

### Configuration

| Setting         | Value                    |
| --------------- | ------------------------ |
| Platform        | Vercel                   |
| Domain          | `emetix-woad.vercel.app` |
| Root Directory  | `frontend`               |
| Framework       | Next.js (auto-detected)  |
| Build Command   | `npm run build`          |
| Node.js Version | 18.x                     |

### Environment Variables (Set in Vercel Dashboard)

| Variable              | Value                               |
| --------------------- | ----------------------------------- |
| `NEXT_PUBLIC_API_URL` | `https://<your-render-backend-url>` |

### Deployment

Vercel auto-deploys from the Git repository. Push to `main` to trigger a production deployment.

---

## MongoDB Atlas

### Setup

1. Create a free cluster at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create a database user with read/write permissions
3. Whitelist your IP (or `0.0.0.0/0` for Render.com)
4. Copy the connection string to `MONGODB_URI`

### Collections

| Collection          | Purpose                          |
| ------------------- | -------------------------------- |
| `universe_stocks`   | Full ~5,800 stock universe       |
| `attention_stocks`  | Stage 1 attention scan results   |
| `qualified_stocks`  | Stage 2 qualified stocks         |
| `classified_stocks` | Stage 3 classified stocks        |
| `curated_watchlist` | Final curated watchlist          |
| `watchlists`        | User-saved watchlists            |
| `strategies`        | User-saved investment strategies |

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas)
- Git

### Backend Setup

```bash
# Clone & enter project
git clone <repo-url>
cd emetix

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
# (see Environment Variables table above)

# Start backend server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
cd frontend
npm install

# Create .env.local
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local

# Start development server
npm run dev
```

### Verify

- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:3000`

---

## GPU Training (Optional)

For LSTM-DCF model training:

```bash
# Verify CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Training automatically uses GPU when available
python scripts/lstm/train_lstm_dcf_v2.py
```

Tested on RTX 3050 (~6 min training vs ~30–60 min on CPU).

---

## Troubleshooting

| Issue                    | Solution                                                                  |
| ------------------------ | ------------------------------------------------------------------------- |
| GEMINI API errors        | Verify `GOOGLE_GEMINI_API_KEY` is set; check quota at aistudio.google.com |
| MongoDB connection fails | Check `MONGODB_URI` format; whitelist IP in Atlas                         |
| GPU not detected         | CUDA 11.8 required; training falls back to CPU                            |
| Frontend can't reach API | Verify `NEXT_PUBLIC_API_URL` in `.env.local`                              |
| Import errors            | Ensure virtual environment is activated (`.\venv\Scripts\Activate.ps1`)   |
| DataFrame ValueError     | Use `.empty` check, never truthiness check on DataFrames                  |
| Render cold start        | Free plan sleeps after 15 min idle; first request takes ~30s              |
