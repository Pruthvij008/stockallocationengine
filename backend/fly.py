import datetime

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

app = Flask(__name__)
app.json.sort_keys = False  # preserve insertion order (Indian universe first)

# Annual risk-free rate used for Sharpe ratios / CAPM.
RISK_FREE_RATE = 0.017

# In-memory price cache so we only hit yfinance once per day, not per request.
_PRICE_CACHE = {"date": None, "adj_close": None}


def _universe_tickers():
    """The investable universe, taken from the columns of the bundled CSV."""
    header = pd.read_csv("stock_data.csv", header=[0, 1], index_col=0, nrows=1)
    return sorted({ticker for _, ticker in header.columns})


def get_price_data():
    """Live adjusted-close prices for the universe, cached for the day."""
    today = datetime.date.today()
    if _PRICE_CACHE["date"] == today and _PRICE_CACHE["adj_close"] is not None:
        return _PRICE_CACHE["adj_close"]

    tickers = _universe_tickers()
    data = yf.download(tickers, start="2020-01-01", auto_adjust=False, progress=False)
    adj_close = data["Adj Close"]
    # Drop tickers that returned no data (delisted / renamed symbols).
    adj_close = adj_close.dropna(axis=1, how="all")

    _PRICE_CACHE["date"] = today
    _PRICE_CACHE["adj_close"] = adj_close
    return adj_close


def _ann_risk_return(returns_df):
    summary = returns_df.agg(["mean", "std"]).T
    summary.columns = ["Return", "Risk"]
    summary.Return = summary.Return * 252
    summary.Risk = summary.Risk * np.sqrt(252)
    return summary


def _portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    weights = np.array(weights)
    port_return = np.sum(mean_returns * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio


def _select_stocks(daily_returns, risk_tolerance, num_stocks):
    """Pick `num_stocks` names: best Sharpe within the requested risk band,
    favouring low mutual correlation for diversification.

    This intentionally replaces the previous RandomForest approach, which
    leaked the label (derived from `returns`) into the features and so produced
    a meaningless ranking.
    """
    ann_return = daily_returns.mean() * 252
    ann_risk = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_risk

    metrics = pd.DataFrame({"Return": ann_return, "Risk": ann_risk, "Sharpe": sharpe})
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()

    # Bucket stocks into low/medium/high by their annualized volatility.
    low_threshold = metrics["Risk"].quantile(0.33)
    high_threshold = metrics["Risk"].quantile(0.67)

    def categorize(risk):
        if risk <= low_threshold:
            return "low"
        if risk >= high_threshold:
            return "high"
        return "medium"

    metrics["RiskCategory"] = metrics["Risk"].apply(categorize)

    pool = metrics[metrics["RiskCategory"] == risk_tolerance]
    if len(pool) < num_stocks:
        pool = metrics  # relax if the band doesn't have enough names
    pool = pool.sort_values("Sharpe", ascending=False)

    corr = daily_returns.corr()
    selected = []
    for ticker in pool.index:
        if len(selected) >= num_stocks:
            break
        if all(abs(corr.loc[ticker, s]) <= 0.6 for s in selected):
            selected.append(ticker)

    # If the correlation filter was too strict, top up by Sharpe.
    if len(selected) < num_stocks:
        for ticker in pool.index:
            if ticker not in selected:
                selected.append(ticker)
            if len(selected) >= num_stocks:
                break

    return selected


def _optimize_max_sharpe(ret):
    noa = ret.shape[1]
    mean_returns = ret.mean()
    cov_matrix = ret.cov()

    def neg_sharpe(weights):
        return -_portfolio_statistics(weights, mean_returns, cov_matrix)[2]

    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(noa))
    initial_guess = noa * [1.0 / noa]

    result = minimize(neg_sharpe, initial_guess, method="SLSQP", bounds=bounds, constraints=cons)
    return result.x, mean_returns, cov_matrix


def _fetch_stock_info(tickers):
    """Fundamentals for each ticker. Yahoo's `.info` is flaky, so a failure for
    one ticker must not break the whole prediction."""
    stock_info = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:
            print(f"Could not fetch info for {ticker}: {exc}")
            info = {}
        stock_info[ticker] = {
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Summary": info.get("longBusinessSummary"),
            "Market_Cap": info.get("marketCap"),
            "PE_Ratio": info.get("trailingPE"),
            "Divident_Yield": info.get("dividendYield"),
        }
    return stock_info


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json() or {}

        investment_amount = float(body.get("investment_amount") or 0)
        investment_period = float(body.get("investment_period") or 0)
        risk_tolerance = str(body.get("risk_tolerance") or "medium").lower()
        expected_return = float(body.get("expected_return") or 0)
        num_stocks = int(float(body.get("num_stocks") or 6))
        num_stocks = max(2, min(num_stocks, 12))

        # 1. Live prices + daily returns
        prices = get_price_data()
        daily_returns = prices.pct_change().dropna(how="all")

        # 2. Select diversified, high-Sharpe stocks in the requested risk band
        selected = _select_stocks(daily_returns, risk_tolerance, num_stocks)
        if len(selected) < 2:
            return jsonify({"error": "Not enough stocks available to build a portfolio"}), 400

        selected_prices = prices[selected].dropna()
        ret = selected_prices.pct_change().dropna()

        # 3. Optimize max-Sharpe weights
        opt_weights, mean_returns, cov_matrix = _optimize_max_sharpe(ret)
        ret["MP"] = ret[selected].dot(opt_weights)
        opt_port_return, opt_port_volatility, opt_sharpe_ratio = _portfolio_statistics(
            opt_weights, mean_returns, cov_matrix
        )

        # 4. Per-stock + portfolio summary
        summary = _ann_risk_return(ret)
        summary["sharpe"] = (summary["Return"] - RISK_FREE_RATE) / summary["Risk"]

        percentage_weights = opt_weights * 100
        portfolio_data = {
            "Optimized_Portfolio_Weights_in_Percentage": [
                {"Ticker": stock, "Weight": f"{weight:.2f}%"}
                for stock, weight in zip(selected, percentage_weights)
            ]
        }

        annualized_return_percentage = summary["Return"] * 100
        annualized_risk_percentage = summary["Risk"] * 100
        annual_data = {
            "Annualized_Return_and_Risk": [
                {
                    "Ticker": ticker,
                    "Return": f"{ann_ret:.2f}%",
                    "Risk": f"{risk:.2f}%",
                    "Sharpe": f"{sharpe:.2f}",
                }
                for ticker, ann_ret, risk, sharpe in zip(
                    summary.index,
                    annualized_return_percentage,
                    annualized_risk_percentage,
                    summary["sharpe"],
                )
            ]
        }

        # 5. Simple projection figures (kept for the UI cards)
        investment_per_stock = investment_amount / len(selected)
        investment_value_after_period = investment_amount * (
            1 + opt_port_return * (investment_period if investment_period else 1)
        )

        stock_info = _fetch_stock_info(selected)

        result = {
            "top_stock_tickers": selected,
            "num_stocks": len(selected),
            "data_as_of": str(prices.index[-1].date()),
            "investment_amount": investment_amount,
            "investment_per_stock": investment_per_stock,
            "investment_value_after_period": investment_value_after_period,
            "optimized_portfolio_sharper_ratio": opt_sharpe_ratio,
            "optimized_portfolio_weights_in_percentage": portfolio_data,
            "optimized_portfolio_annualized_return": opt_port_return * 100,
            "annualized_return_and_risk_in_percentage": annual_data,
            "stock_info": stock_info,
        }

        return jsonify(result)

    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return jsonify({"error": str(exc)}), 500


_SUMMARY_CACHE = {"date": None, "data": None}


def _fetch_index_returns(tickers, start="2014-01-01"):
    """Daily returns of the first benchmark index that downloads, else None."""
    for ticker in tickers:
        try:
            px = yf.download(ticker, start=start, auto_adjust=False, progress=False)["Adj Close"]
            if isinstance(px, pd.DataFrame):
                px = px.squeeze("columns")
            if px is not None and len(px) > 0:
                returns = px.pct_change().dropna()
                returns.index = pd.to_datetime(returns.index)
                return returns
        except Exception as exc:
            print(f"Index download failed for {ticker}: {exc}")
    return None


def _summary_with_capm(daily_returns, market_col):
    """Annualized return/risk/Sharpe plus CAPM beta/alpha vs `market_col`."""
    summary = _ann_risk_return(daily_returns)
    summary["sharpe"] = (summary["Return"] - RISK_FREE_RATE) / summary["Risk"]
    summary["TotalRisk_var"] = np.power(summary.Risk, 2)

    summary["SystRisk_var"] = np.nan
    summary["UnsystRisk_var"] = np.nan
    summary["beta"] = np.nan
    summary["capm_ret"] = np.nan
    summary["alpha"] = np.nan

    if market_col in daily_returns.columns and daily_returns[market_col].notna().any():
        COV = daily_returns.cov() * 252
        summary["SystRisk_var"] = COV[market_col]
        summary["UnsystRisk_var"] = summary["TotalRisk_var"] - summary["SystRisk_var"]
        market_var = summary.loc[market_col, "SystRisk_var"]
        market_ret = summary.loc[market_col, "Return"]
        summary["beta"] = summary.SystRisk_var / market_var
        summary["capm_ret"] = RISK_FREE_RATE + (market_ret - RISK_FREE_RATE) * summary.beta
        summary["alpha"] = summary.Return - summary.capm_ret

    return summary


def _clean_to_dict(summary):
    summary = summary.replace([np.inf, -np.inf], np.nan)
    return summary.astype(object).where(pd.notnull(summary), None).to_dict(orient="index")


def load_and_process_data():
    today = datetime.date.today()
    if _SUMMARY_CACHE["date"] == today and _SUMMARY_CACHE["data"] is not None:
        return _SUMMARY_CACHE["data"]

    result = {}

    # Indian universe (the stocks we actually allocate) benchmarked vs Nifty 50.
    try:
        in_returns = get_price_data().pct_change().dropna(how="all")
        nifty = _fetch_index_returns(["^NSEI"])
        if nifty is not None:
            in_returns = in_returns.join(nifty.rename("^NSEI"), how="left")
        result.update(_clean_to_dict(_summary_with_capm(in_returns, "^NSEI")))
    except Exception as exc:
        print(f"Indian summary failed: {exc}")

    # US sample benchmarked vs the S&P 500.
    try:
        st = pd.read_csv("sp500_selected_stocks.csv", header=0, index_col=0, parse_dates=True)
        st.index = pd.to_datetime(st.index)
        us_returns = st.pct_change().dropna()
        gspc = _fetch_index_returns(["^GSPC", "SPY"])
        if gspc is not None:
            us_returns = us_returns.join(gspc.rename("^GSPC"), how="left")
        result.update(_clean_to_dict(_summary_with_capm(us_returns, "^GSPC")))
    except Exception as exc:
        print(f"US summary failed: {exc}")

    _SUMMARY_CACHE["date"] = today
    _SUMMARY_CACHE["data"] = result
    return result


@app.route("/summary", methods=["GET"])
def get_summary():
    return jsonify(load_and_process_data())


@app.route("/meta", methods=["GET"])
def get_meta():
    """Coverage info about the data we currently hold (for the UI)."""
    try:
        prices = get_price_data()
        return jsonify({
            "source": "Yahoo Finance",
            "library": "yfinance",
            "frequency": "Daily adjusted close",
            "refresh": "Fetched live and cached once per day",
            "universe": "NSE-listed Indian equities",
            "benchmarks": ["^NSEI (Nifty 50)", "^GSPC (S&P 500)"],
            "tickers": int(prices.shape[1]),
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "trading_days": int(prices.shape[0]),
        })
    except Exception as exc:
        print(f"Meta failed: {exc}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
