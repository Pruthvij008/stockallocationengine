"""
Out-of-sample, walk-forward machine-learning backtest.

This module is intentionally separate from `fly.py`. `fly.py` runs a
*deterministic* mean-variance (Markowitz) optimiser; this one layers a real
supervised-learning model on top and proves it out of sample.

The story it tells (and the one to tell in an interview):

  * Every month we build cross-sectional features for each stock using ONLY
    information available up to that date — trailing momentum over several
    horizons and trailing volatility. No future data ever touches the features.
  * The label is the stock's return over the NEXT rebalance period. A sample's
    label is therefore only known one period later, so it can never leak into a
    prediction made today.
  * We walk forward through time. At each rebalance date the model is trained
    *only* on periods whose labels have already been realised, then it ranks the
    universe and we equal-weight the top-K names for the coming month.
  * The resulting equity curve is fully out of sample and is compared against
    simply buying and holding the Nifty 50.

This directly answers the most common critique of the deterministic engine —
"you optimised in-sample, of course it looks good." Here the model never sees
the period it is judged on.
"""

import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

backtest_bp = Blueprint("backtest", __name__)

# Trailing windows (in trading days) used to build the predictive features.
MOMENTUM_WINDOWS = {
    "mom_1m": 21,
    "mom_3m": 63,
    "mom_6m": 126,
    "mom_12m": 252,
}
VOLATILITY_WINDOWS = {
    "vol_1m": 21,
    "vol_3m": 63,
}
FEATURE_NAMES = list(MOMENTUM_WINDOWS) + list(VOLATILITY_WINDOWS)

PERIODS_PER_YEAR = 12  # monthly rebalancing
MIN_TRAIN_PERIODS = 12  # warm-up: don't trade until the model has a year of labels
MIN_TRAIN_ROWS = 60  # require a reasonable sample before trusting the model

# Cache results for the day, keyed by the request parameters, since a backtest
# trains dozens of models and is far too slow to recompute on every page load.
_BACKTEST_CACHE = {}


def _rebalance_dates(index):
    """Last trading day of each calendar month in the price index."""
    s = pd.Series(index, index=index)
    last_per_month = s.groupby([index.year, index.month]).last()
    return pd.DatetimeIndex(last_per_month.values)


def _build_panels(prices, rebal):
    """For every rebalance date, a (ticker x features+label) DataFrame.

    Features use only data up to the date; the label is the forward return to
    the *next* rebalance date, so it is unknown until one period later.
    """
    daily_ret = prices.pct_change()

    features = {}
    for name, w in MOMENTUM_WINDOWS.items():
        features[name] = prices.pct_change(w).loc[rebal]
    for name, w in VOLATILITY_WINDOWS.items():
        features[name] = (daily_ret.rolling(w).std() * np.sqrt(252)).loc[rebal]

    # Period (rebalance-to-rebalance) return, attributed to the *start* date.
    rebal_prices = prices.loc[rebal]
    period_returns = rebal_prices.pct_change().shift(-1)

    panels = {}
    for d in rebal:
        panel = pd.DataFrame({f: features[f].loc[d] for f in FEATURE_NAMES})
        panel["y"] = period_returns.loc[d]
        panels[d] = panel
    return panels, period_returns


def _make_model(model_name):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        )
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _metrics(period_rets, risk_free_rate):
    """Total return, CAGR, annualised Sharpe, max drawdown, win rate."""
    rets = np.asarray(period_rets, dtype=float)
    equity = np.cumprod(1 + rets)
    total_return = float(equity[-1] - 1) if len(equity) else 0.0

    years = len(rets) / PERIODS_PER_YEAR if len(rets) else 0
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else 0.0

    ann_ret = float(np.mean(rets) * PERIODS_PER_YEAR)
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(rets) > 1 else 0.0
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    running_max = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
    drawdowns = equity / running_max - 1
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

    win_rate = float(np.mean(rets > 0)) if len(rets) else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_return": ann_ret,
        "volatility": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def _run_backtest(top_k=10, model_name="gbr"):
    # Lazy import to avoid a circular import with fly.py.
    from fly import get_price_data, _fetch_index_returns, RISK_FREE_RATE

    prices = get_price_data().dropna(axis=1, how="all")
    rebal = _rebalance_dates(prices.index)
    panels, period_returns = _build_panels(prices, rebal)

    # Benchmark: buy & hold the Nifty 50 over the same rebalance grid.
    nifty_ret = _fetch_index_returns(["^NSEI"])
    if nifty_ret is not None:
        nifty_level = (1 + nifty_ret).cumprod().reindex(prices.index, method="ffill")
        nifty_period = nifty_level.loc[rebal].pct_change().shift(-1)
    else:
        nifty_period = pd.Series(index=rebal, dtype=float)

    curve = []
    strat_rets, bench_rets = [], []
    strat_equity, bench_equity = 1.0, 1.0
    last_model = None
    last_feature_cols = FEATURE_NAMES

    # Walk forward: trade from MIN_TRAIN_PERIODS until the second-to-last date
    # (the last date has no realised forward return to score against).
    for i in range(MIN_TRAIN_PERIODS, len(rebal) - 1):
        d = rebal[i]

        # Train only on periods whose labels are already realised (j <= i-1).
        train = pd.concat([panels[rebal[j]] for j in range(i)]).dropna()
        if len(train) < MIN_TRAIN_ROWS:
            continue

        model = _make_model(model_name)
        model.fit(train[FEATURE_NAMES], train["y"])
        last_model = model

        current = panels[d].dropna(subset=FEATURE_NAMES)
        if current.empty:
            continue

        current = current.assign(pred=model.predict(current[FEATURE_NAMES]))
        picks = current.sort_values("pred", ascending=False).head(top_k)

        realized = picks["y"].dropna()
        strat_ret = float(realized.mean()) if len(realized) else 0.0
        bench_ret = float(nifty_period.loc[d]) if d in nifty_period.index and pd.notna(nifty_period.loc[d]) else 0.0

        strat_rets.append(strat_ret)
        bench_rets.append(bench_ret)
        strat_equity *= 1 + strat_ret
        bench_equity *= 1 + bench_ret

        curve.append({
            "date": str(pd.Timestamp(d).date()),
            "strategy": round(strat_equity, 4),
            "benchmark": round(bench_equity, 4),
        })

    if not curve:
        raise ValueError("Not enough history to run a walk-forward backtest.")

    # Latest model's view: what it would hold today and which features matter.
    last_panel = panels[rebal[-1]].dropna(subset=FEATURE_NAMES)
    holdings = []
    if last_model is not None and not last_panel.empty:
        scored = last_panel.assign(pred=last_model.predict(last_panel[FEATURE_NAMES]))
        holdings = [
            {"ticker": t.replace(".NS", ""), "score": round(float(p), 4)}
            for t, p in scored.sort_values("pred", ascending=False)
            .head(top_k)["pred"].items()
        ]

    importance = []
    if last_model is not None:
        importance = sorted(
            (
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in zip(last_feature_cols, last_model.feature_importances_)
            ),
            key=lambda x: x["importance"],
            reverse=True,
        )

    return {
        "as_of": str(pd.Timestamp(rebal[-1]).date()),
        "start": curve[0]["date"],
        "end": curve[-1]["date"],
        "model": "GradientBoostingRegressor" if model_name != "rf" else "RandomForestRegressor",
        "rebalance": "monthly",
        "top_k": top_k,
        "n_periods": len(curve),
        "universe_size": int(prices.shape[1]),
        "features": FEATURE_NAMES,
        "method": (
            "Walk-forward, out-of-sample. Each month the model is trained only on "
            "periods with already-realised labels, then ranks the universe and "
            "equal-weights the top-K names for the next month."
        ),
        "curve": curve,
        "strategy": _metrics(strat_rets, RISK_FREE_RATE),
        "benchmark": _metrics(bench_rets, RISK_FREE_RATE),
        "feature_importance": importance,
        "current_holdings": holdings,
    }


@backtest_bp.route("/backtest", methods=["GET"])
def backtest():
    try:
        top_k = int(float(request.args.get("top_k") or 10))
        top_k = max(3, min(top_k, 25))
        model_name = (request.args.get("model") or "gbr").lower()
        if model_name not in ("gbr", "rf"):
            model_name = "gbr"

        cache_key = (datetime.date.today(), top_k, model_name)
        if cache_key in _BACKTEST_CACHE:
            return jsonify(_BACKTEST_CACHE[cache_key])

        result = _run_backtest(top_k=top_k, model_name=model_name)
        _BACKTEST_CACHE[cache_key] = result
        return jsonify(result)
    except Exception as exc:
        print(f"Backtest failed: {exc}")
        return jsonify({"error": str(exc)}), 500
