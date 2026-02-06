# 7. Deployment Guide

> **Environment Setup, Production Deployment, and DevOps**

---

## 🖥️ Development Environment

### Prerequisites

| Software | Version | Purpose                 |
| -------- | ------- | ----------------------- |
| Python   | 3.10+   | Backend runtime         |
| Node.js  | 18+     | Frontend runtime        |
| Git      | 2.30+   | Version control         |
| CUDA     | 11.8+   | GPU training (optional) |

### Backend Setup

```powershell
# 1. Clone repository
git clone https://github.com/your-org/emetix.git
cd emetix

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your API keys:
#   GEMINI_API_KEY=your_google_ai_key_here
#   ALPHA_VANTAGE_API_KEY=your_key_here

# 5. Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from src.analysis.stock_screener import StockScreener; print('OK')"
```

### Environment Variables

```env
# .env file
# Required for AI agents
GEMINI_API_KEY=your_google_ai_api_key

# MongoDB Atlas
MONGODB_URI=your_mongoDB_connection_url_here

# Optional (for training data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Optional (for news sentiment)
NEWSAPI_KEY=your_newsapi_key
```

---

## 🚀 Running Locally

### Start Backend Server

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start FastAPI server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
python -m uvicorn src.api.app:app --reload --port 8000
```

**Verify**:

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Start Frontend (when implemented)

```bash
cd frontend
npm install
npm run dev
```

**Access**: http://localhost:3000

---

## 📦 Docker Deployment

### Backend Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - MONGODB_URI=${MONGODB_URI}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
```

### Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## ☁️ Cloud Deployment Options

### Option 1: Railway (Recommended for FYP)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Pros**: Free tier, easy setup, automatic deployments

### Option 2: Render

1. Connect GitHub repository
2. Create Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables

### Option 3: AWS (Production)

```yaml
# AWS ECS Task Definition (simplified)
containerDefinitions:
  - name: emetix-api
    image: your-ecr-repo/emetix:latest
    portMappings:
      - containerPort: 8000
    memory: 1024
    cpu: 512
    environment:
      - name: GEMINI_API_KEY
        valueFrom: arn:aws:secretsmanager:...
      - name: MONGODB_URI
        valueFrom: arn:aws:secretsmanager:...
```

---

## 🔧 Production Configuration

### Uvicorn Workers

```bash
# Production with multiple workers
uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --access-log \
    --log-level info
```

### Gunicorn (Alternative)

```bash
gunicorn src.api.app:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### NGINX Reverse Proxy

```nginx
# /etc/nginx/sites-available/emetix
server {
    listen 80;
    server_name api.emetix.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🔒 Security Checklist

### Pre-Production

- [ ] Remove `allow_origins=["*"]` in CORS, specify allowed domains
- [ ] Enable HTTPS with SSL certificate
- [ ] Add rate limiting (FastAPI-limiter)
- [ ] Set secure environment variables (not in code)
- [ ] Enable logging and monitoring
- [ ] Add authentication (if needed)

### Environment Security

```python
# config/settings.py - Production checks
import os

if os.getenv("ENV") == "production":
    # Validate required variables
    assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY required"
    assert os.getenv("MONGODB_URI"), "MONGODB_URI required"

    # Disable debug features
    DEBUG = False
    LOG_LEVEL = "WARNING"
```

---

## 📊 Monitoring

### Health Checks

```python
# Already implemented at /health
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Logging

```python
# config/logging_config.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log')
        ]
    )
```

### Metrics (Future)

Consider adding:

- Prometheus metrics endpoint
- Request duration tracking
- Error rate monitoring
- Model inference time

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        run: railway up
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

---

## 🛠️ Troubleshooting

### Common Issues

| Issue                          | Solution                                             |
| ------------------------------ | ---------------------------------------------------- |
| `ModuleNotFoundError: fastapi` | Install: `pip install fastapi uvicorn`               |
| `CUDA not available`           | Install CUDA toolkit 11.8+, verify with `nvidia-smi` |
| `GEMINI_API_KEY missing`       | Add to `.env` file                                   |
| `MongoDB connection failed`    | Verify `MONGODB_URI` and network access              |
| Port 8000 in use               | Use different port: `--port 8001`                    |
| Slow first request             | Model loading on startup, consider preloading        |

### Logs Location

```
logs/
├── app.log           # Application logs
├── uvicorn.log       # Server logs
└── error.log         # Error logs
```

---

## 📋 Quick Reference

### Essential Commands

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Start server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Check model status
python scripts/evaluate/quick_model_test.py

# Collect fresh data
python scripts/quick_start_data_collection.py
```

### API Quick Test

```bash
# Health
curl http://localhost:8000/health

# Watchlist
curl "http://localhost:8000/api/screener/watchlist?n=5"

# Single stock
curl http://localhost:8000/api/screener/stock/AAPL
```

---

## 🗄️ Database Integration (MongoDB Atlas)

### Why MongoDB Atlas

| Feature          | MongoDB Atlas                     | Firebase | PlanetScale | Raw PostgreSQL   |
| ---------------- | --------------------------------- | -------- | ----------- | ---------------- |
| Free Tier        | Generous (512MB, unlimited reads) | Limited  | Limited     | Requires hosting |
| Document Store   | ✅ Native JSON                    | ✅       | ❌ SQL      | ❌ SQL           |
| No Auth Required | ✅ Session-based                  | ❌       | ❌          | ❌               |
| Python Driver    | ✅ pymongo                        | ✅       | ✅          | ✅               |
| Serverless       | ✅ Atlas Serverless               | ✅       | ✅          | ❌               |
| Global Clusters  | ✅ Multi-region                   | ✅       | ✅          | Manual           |

**Recommendation**: MongoDB Atlas for free tier, flexible schema, and no auth complexity.

### MongoDB Atlas Setup

1. Create project at [cloud.mongodb.com](https://cloud.mongodb.com)
2. Create a free M0 cluster (512MB)
3. Add database user with read/write permissions
4. Whitelist IP addresses (or allow 0.0.0.0/0 for dev)
5. Get connection string from Connect > Drivers
6. Add to `.env`:

```env
MONGODB_URI=your_mongoDB_connection_url_here
MONGODB_DATABASE=emetix
```

### Database Collections

| Collection        | Purpose                            | Storage      |
| ----------------- | ---------------------------------- | ------------ |
| **watchlists**    | User-created watchlist collections | MongoDB      |
| **strategies**    | Saved screening strategies         | MongoDB      |
| **education**     | Educational content                | MongoDB      |
| **risk_profiles** | N/A - stored client-side           | localStorage |

### Python Client

```bash
pip install pymongo
```

See `src/data/mongo_client.py` for complete implementation.

### Data Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT-SIDE (Browser)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  localStorage                                        │   │
│  │  ├── emetix_profile_id: "uuid-xxx"                   │   │
│  │  ├── emetix_risk_profile: { full profile JSON }      │   │
│  │  └── theme: "dark" | "light"                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ API Calls
┌─────────────────────────────────────────────────────────────┐
│                    SERVER-SIDE (MongoDB Atlas)              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Collections:                                        │   │
│  │  ├── watchlists (session_id indexed)                 │   │
│  │  ├── strategies (public/private)                     │   │
│  │  └── education (static content)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Why No User Auth?**

- FYP scope focuses on ML + Risk Framework, not auth infrastructure
- Risk profiles are personal and private (localStorage)
- Watchlists use session-based ownership (no login required)
- Reduces complexity and deployment costs

---

## 🌐 Production Hosting Stack

### Recommended Free-Tier Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vercel)                        │
│                    Next.js 14 + Tailwind                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Render)                         │
│                    FastAPI + ML Models                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Database (MongoDB Atlas)                 │
│               Document Store (No Auth Required)             │
└─────────────────────────────────────────────────────────────┘
```

### Free Tier Limits (Sufficient for FYP)

| Service           | Free Limit                     | Expected Usage  |
| ----------------- | ------------------------------ | --------------- |
| **MongoDB Atlas** | 512MB storage, unlimited reads | ~50MB, ~10K/mo  |
| **Render**        | 750 hrs/mo, 512MB RAM          | ~720 hrs, 256MB |
| **Vercel**        | 100GB bandwidth, 1M requests   | ~10GB, ~50K     |

All within free tier - **$0/month for MVP**.

### Environment Variables Summary

**Backend (.env)**:

```env
# Required
GEMINI_API_KEY=your_google_ai_api_key

# Database (MongoDB Atlas)
MONGODB_URI=your_mongoDB_connection_url_here
MONGODB_DATABASE=emetix

# Optional
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
CORS_ORIGINS=https://your-frontend.vercel.app
```

**Frontend (.env.local)**:

```env
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
```

---

_Documentation complete. Return to [README](./README.md)_
