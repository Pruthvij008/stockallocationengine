# Investify — Stock Allocation Engine

A portfolio optimization web app. Users enter an investment amount, period, and
risk tolerance; a machine-learning service selects stocks and computes a
**max-Sharpe-ratio** allocation, which the UI renders as charts and tables.

## Architecture

| Part | Stack | Location |
| --- | --- | --- |
| **Frontend** | React 19 + Vite + Tailwind CSS 4 + Chart.js | `frontend/` |
| **Backend API** | Node.js + Express 5 + MongoDB (Mongoose) + JWT | `backend/` |
| **ML service** | Python + Flask + scikit-learn + scipy + yfinance | `backend/fly.py` |

Request flow: **React → Express (`/api/portfolio/*`) → Flask (`/predict`, `/summary`)**.

## Prerequisites

- Node.js 20+
- Python 3.10+
- MongoDB running locally (or a connection string)

## Setup

### 1. Environment files

Copy the examples and fill in values:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

### 2. Install dependencies

```bash
# From the repo root — installs root, backend, and frontend
npm run install:all

# Python ML service
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Start the three processes (in separate terminals, or use the root `dev` script
for the two Node processes):

```bash
# 1. Flask ML service (port 5000)
cd backend && python fly.py

# 2 + 3. Express API (8080) and Vite frontend (5173) together
npm run dev
```

Then open http://localhost:5173.

## Deployment (Render)

The repo ships a `render.yaml` Blueprint that provisions all three pieces:

| Service | Type | Notes |
| --- | --- | --- |
| `investify-web` | Static Site | React build — always on, never sleeps |
| `investify-api` | Web Service (Node) | Express proxy |
| `investify-ml` | Web Service (Python) | Flask + gunicorn |

Steps:

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo. Render reads `render.yaml` and
   creates the three services, wiring the cross-service URLs automatically.
3. Open the `investify-web` URL.

**Health checks:** `GET /api/health` (Express) and `GET /health` (Flask).

**Keep-alive:** free web services sleep after ~15 min idle. Each backend
self-pings when running, but to guarantee they stay awake, add a free uptime
monitor (e.g. [cron-job.org](https://cron-job.org) or UptimeRobot) that hits
`https://investify-api.onrender.com/api/health` every ~10 minutes — that endpoint
also warms the Flask service.

> First backtest load trains the walk-forward models (~90s) and is cached for the
> day. The free 512 MB instances are enough for a demo but can be tight.

## Notes

- The Flask service reads `backend/stock_data.csv` and `backend/sp500_selected_stocks.csv`
  and fetches live data via `yfinance`.
- `backend/flyprev.py` is an earlier ML variant that uses the FRED API; set
  `FRED_API_KEY` in `backend/.env` if you want to run it.
