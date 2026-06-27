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

## Notes

- The Flask service reads `backend/stock_data.csv` and `backend/sp500_selected_stocks.csv`
  and fetches live data via `yfinance`.
- `backend/flyprev.py` is an earlier ML variant that uses the FRED API; set
  `FRED_API_KEY` in `backend/.env` if you want to run it.
