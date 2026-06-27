import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import Navbar from "./Navbar";
import { API_URL } from "../api";
import {
  FaArrowTrendUp,
  FaDatabase,
  FaScaleBalanced,
  FaFilter,
  FaSliders,
  FaChartPie,
  FaServer,
  FaCloudArrowDown,
  FaBrain,
  FaRobot,
  FaCodeBranch,
  FaClockRotateLeft,
} from "react-icons/fa6";

const fmtPct = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : (Number(v) * 100).toFixed(1) + "%";
const fmtNum = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : Number(v).toFixed(2);

const steps = [
  {
    icon: FaDatabase,
    title: "1. Live market data",
    text: "Pulls daily prices for 140+ NSE-listed stocks from Yahoo Finance, refreshed every day — never stale.",
  },
  {
    icon: FaArrowTrendUp,
    title: "2. Risk & return",
    text: "For each stock we compute annualized return, volatility (risk) and the Sharpe ratio (return earned per unit of risk).",
  },
  {
    icon: FaScaleBalanced,
    title: "3. Risk matching",
    text: "Stocks are bucketed into Low / Medium / High by volatility, and we keep only those that match the risk level you choose.",
  },
  {
    icon: FaFilter,
    title: "4. Smart selection",
    text: "We rank the matching stocks by Sharpe ratio and pick the number you asked for, skipping highly-correlated names so the basket stays diversified.",
  },
  {
    icon: FaSliders,
    title: "5. Optimization",
    text: "A Sequential Least-Squares optimizer (SLSQP) finds the exact weights that maximize the portfolio's Sharpe ratio — the best risk-adjusted mix.",
  },
  {
    icon: FaChartPie,
    title: "6. Your portfolio",
    text: "You get the allocation, a projected growth curve, and per-stock risk/return metrics — all in one view.",
  },
];

const guide = [
  "Open the Predictions page.",
  'Enter your amount — shorthand works: type 5L, 2.5Cr, 1M, 500K or 1B.',
  "Set how long you'll stay invested, in years.",
  "Pick your risk tolerance and how many stocks you want.",
  "Hit Calculate to see your optimized allocation and projected returns.",
];

const Home = () => {
  const navigate = useNavigate();
  const [meta, setMeta] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [btError, setBtError] = useState(false);

  useEffect(() => {
    axios
      .get(`${API_URL}/api/portfolio/meta`)
      .then((res) => setMeta(res.data))
      .catch(() => setMeta(null));

    // Pull the latest walk-forward ML backtest so the home page can show the
    // model's real, out-of-sample performance. First run trains dozens of
    // models and can take ~90s; it's cached server-side after that.
    axios
      .get(`${API_URL}/api/portfolio/backtest`, { params: { top_k: 10, model: "gbr" } })
      .then((res) => {
        if (res.data?.error) setBtError(true);
        else setBacktest(res.data);
      })
      .catch(() => setBtError(true));
  }, []);

  const btBeats =
    backtest &&
    Number(backtest.strategy?.total_return) > Number(backtest.benchmark?.total_return);
  const btMultiple =
    backtest && Number(backtest.benchmark?.total_return) !== 0
      ? (
          (1 + Number(backtest.strategy?.total_return)) /
          (1 + Number(backtest.benchmark?.total_return))
        ).toFixed(1)
      : null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <Navbar />

      {/* Hero */}
      <section className="relative overflow-hidden bg-customBlack-100 pt-28 pb-20 text-white">
        <div className="absolute -right-24 -top-24 h-96 w-96 rounded-full bg-customGreen-100/20 blur-3xl" />
        <div className="absolute -left-24 bottom-0 h-80 w-80 rounded-full bg-customGreen-100/10 blur-3xl" />
        <div className="relative mx-auto flex max-w-6xl flex-col items-center gap-8 px-6 md:flex-row md:justify-between">
          <div className="max-w-xl">
            <span className="inline-block rounded-full bg-customGreen-100/15 px-4 py-1 text-sm font-semibold text-customGreen-100">
              Smart portfolio allocation
            </span>
            <h1 className="mt-5 text-4xl font-extrabold leading-tight md:text-5xl">
              Build an optimized stock portfolio in{" "}
              <span className="text-customGreen-100">seconds</span>.
            </h1>
            <p className="mt-5 text-lg text-slate-300">
              Investify uses Modern Portfolio Theory and live market data to pick
              a diversified set of stocks and find the weights that give you the
              best return for the risk you're comfortable with.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <button
                onClick={() => navigate("/prediction")}
                className="rounded-xl bg-customGreen-100 px-7 py-3 font-semibold text-white shadow-lg transition hover:scale-105 hover:bg-green-600"
              >
                Get started
              </button>
              <button
                onClick={() => navigate("/performance")}
                className="rounded-xl border border-white/20 px-7 py-3 font-semibold text-white transition hover:bg-white/10"
              >
                View performance
              </button>
            </div>
          </div>

          <div className="grid w-full max-w-sm grid-cols-2 gap-4">
            {[
              { k: "140+", v: "NSE stocks" },
              { k: "Daily", v: "Live data" },
              { k: "Max", v: "Sharpe ratio" },
              { k: "2–12", v: "Stocks you pick" },
            ].map((s) => (
              <div
                key={s.v}
                className="rounded-2xl border border-white/10 bg-white/5 p-5 text-center backdrop-blur"
              >
                <div className="text-2xl font-extrabold text-customGreen-100">
                  {s.k}
                </div>
                <div className="mt-1 text-sm text-slate-300">{s.v}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="text-center">
          <h2 className="text-3xl font-bold">How it works</h2>
          <p className="mx-auto mt-3 max-w-2xl text-slate-600">
            Under the hood it's the Markowitz mean-variance framework — select
            good stocks, then optimize their weights for the best risk-adjusted
            return.
          </p>
        </div>

        <div className="mt-10 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {steps.map(({ icon: Icon, title, text }) => (
            <div key={title} className="iv-card iv-card-hover p-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-customGreen-100/10 text-xl text-customGreen-100">
                <Icon />
              </div>
              <h3 className="mt-4 text-lg font-semibold">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                {text}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* The math, briefly */}
      <section className="bg-white py-16">
        <div className="mx-auto max-w-6xl px-6">
          <div className="iv-card grid gap-8 p-8 md:grid-cols-2 md:p-10">
            <div>
              <h2 className="text-2xl font-bold">The idea in one line</h2>
              <p className="mt-4 leading-relaxed text-slate-600">
                Every stock has a <strong>return</strong> and a{" "}
                <strong>risk</strong> (how much its price swings). The{" "}
                <strong>Sharpe ratio</strong> measures return earned per unit of
                risk. Investify maximizes the Sharpe ratio of the{" "}
                <em>whole portfolio</em> — so you're never just chasing returns,
                you're chasing the <em>best return for your risk</em>.
              </p>
              <p className="mt-4 leading-relaxed text-slate-600">
                Because stocks that move together add little diversification, we
                also avoid picking highly-correlated names — spreading your money
                across genuinely different bets.
              </p>
            </div>
            <div className="flex flex-col justify-center gap-4 rounded-2xl bg-slate-900 p-8 text-slate-100">
              <div className="font-mono text-sm text-slate-400">
                // maximized for the portfolio
              </div>
              <div className="font-mono text-lg">
                Sharpe ={" "}
                <span className="text-customGreen-100">
                  (Return − RiskFree)
                </span>{" "}
                / <span className="text-amber-300">Risk</span>
              </div>
              <div className="font-mono text-sm text-slate-400">
                subject to: weights ≥ 0 and Σ weights = 1
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Under the hood */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="text-center">
          <h2 className="text-3xl font-bold">What's running behind the scenes</h2>
          <p className="mx-auto mt-3 max-w-3xl text-slate-600">
            No black-box, no opaque "trained model." Investify uses{" "}
            <strong>Modern Portfolio Theory</strong> (Harry Markowitz, 1952) — the
            same mean-variance optimization used across the finance industry.
            Every number below is reproducible from public price data.
          </p>
        </div>

        <div className="mt-10 space-y-5">
          <TechRow
            step="Data pipeline"
            detail="Daily adjusted-close prices for 140+ NSE stocks are pulled from Yahoo Finance (yfinance), from 2020 to today, and cached once per day. Adjusted close accounts for splits and dividends so returns are accurate."
          />
          <TechRow
            step="Per-stock metrics"
            formula="Return = mean(dailyReturns) × 252    ·    Risk = std(dailyReturns) × √252"
            detail="We annualize each stock's average daily return and its volatility (standard deviation). 252 is the number of trading days in a year."
          />
          <TechRow
            step="Sharpe ratio"
            formula="Sharpe = (Return − RiskFreeRate) / Risk"
            detail="The core score: how much return a stock earns per unit of risk taken. Higher is better. We use this to rank and to optimize."
          />
          <TechRow
            step="Risk bucketing & selection"
            detail="Stocks are split into Low / Medium / High by volatility quantiles (33rd & 67th percentile). We keep the bucket you chose, rank by Sharpe, and greedily pick your number of stocks — skipping any pair with correlation above 0.6 so the basket is genuinely diversified."
          />
          <TechRow
            step="Weight optimization (the finance algorithm)"
            formula="maximize  Sharpe(w)   subject to   Σ wᵢ = 1,   wᵢ ≥ 0"
            detail="Given the chosen stocks, a Sequential Least-Squares Programming (SLSQP) optimizer from SciPy searches the weight combinations to maximize the whole portfolio's Sharpe ratio. Weights sum to 100% and can't go negative (no short-selling)."
          />
          <TechRow
            step="Performance page (CAPM)"
            formula="β = Cov(stock, market) / Var(market)    ·    α = Return − [Rf + β(Rₘ − Rf)]"
            detail="On the Performance page we also compute each stock's Beta (sensitivity to the market) and Alpha (excess return beyond what its Beta predicts), benchmarked against the index — classic Capital Asset Pricing Model metrics."
          />
        </div>

        <p className="mx-auto mt-8 max-w-3xl text-center text-sm text-slate-500">
          Earlier versions tried a machine-learning classifier to rank stocks, but
          it leaked its own target into its features (inflating accuracy without
          real skill), so we replaced it with this transparent, well-established
          quantitative method.
        </p>
      </section>

      {/* Where the data comes from */}
      <section className="bg-white py-16">
        <div className="mx-auto max-w-6xl px-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold">Where the data comes from</h2>
            <p className="mx-auto mt-3 max-w-3xl text-slate-600">
              Everything is built on free, public market data — no paid feeds, no
              hidden sources. Here's exactly how it's pulled.
            </p>
          </div>

          <div className="mt-10 grid gap-6 md:grid-cols-2">
            <div className="iv-card p-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-customGreen-100/10 text-xl text-customGreen-100">
                <FaCloudArrowDown />
              </div>
              <h3 className="mt-4 text-lg font-semibold">
                Source: Yahoo Finance via <code>yfinance</code>
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                We use the open-source <code>yfinance</code> Python library, which
                reads from Yahoo Finance. A single batch call —{" "}
                <code className="text-customGreen-100">
                  yf.download(tickers, start="2020-01-01")
                </code>{" "}
                — fetches daily price history for all 140+ symbols at once. We
                take the <strong>Adjusted Close</strong>, which already accounts
                for stock splits and dividends, so returns are accurate.
              </p>
            </div>

            <div className="iv-card p-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-customGreen-100/10 text-xl text-customGreen-100">
                <FaServer />
              </div>
              <h3 className="mt-4 text-lg font-semibold">
                Live, then cached for speed
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                Prices are pulled <strong>live</strong> and cached once per day,
                so you always get the latest close without waiting on a download
                every click. Company fundamentals (sector, P/E, market cap) come
                from <code>yf.Ticker(symbol).info</code>, and we benchmark against
                the <strong>Nifty 50</strong> (^NSEI) and <strong>S&amp;P 500</strong>{" "}
                (^GSPC).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Markowitz, in our code */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="text-center">
          <h2 className="text-3xl font-bold">
            Markowitz mean-variance optimization — in our code
          </h2>
          <p className="mx-auto mt-3 max-w-3xl text-slate-600">
            We don't "train" a statistical model. We solve Harry Markowitz's
            mean-variance problem directly: given the selected stocks, find the
            weight mix on the <strong>efficient frontier</strong> with the highest
            risk-adjusted return (the <strong>tangency portfolio</strong>).
          </p>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          <div className="iv-card p-6">
            <h3 className="text-lg font-semibold">The math</h3>
            <ul className="mt-3 space-y-3 text-sm text-slate-600">
              <li>
                <strong>Expected returns</strong> μ = annualized mean of daily
                returns for each stock.
              </li>
              <li>
                <strong>Covariance matrix</strong> Σ = how the stocks move
                together (annualized).
              </li>
              <li>
                <strong>Portfolio return</strong> = wᵀμ &nbsp;·&nbsp;{" "}
                <strong>Portfolio variance</strong> = wᵀΣw
              </li>
              <li>
                We <strong>maximize the Sharpe ratio</strong> of the whole
                portfolio — the point where the capital line is tangent to the
                efficient frontier.
              </li>
            </ul>
          </div>

          <div className="iv-card overflow-hidden p-0">
            <div className="bg-slate-900 p-6 text-sm text-slate-100">
              <div className="text-slate-400">// fly.py — what the code does</div>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
{`port_return     = wᵀ·μ · 252
port_volatility = √(wᵀ·Σ·w · 252)
sharpe          = (port_return − Rf) / port_volatility

# maximize Sharpe  ⇒  minimize (−Sharpe)
minimize(neg_sharpe, w0,
         method="SLSQP",
         bounds = 0 ≤ wᵢ ≤ 1,      # no short-selling
         constraints = Σ wᵢ = 1)   # fully invested`}
              </pre>
              <div className="mt-3 text-slate-400">
                // solver: SciPy SLSQP (Sequential Least-Squares Programming)
              </div>
            </div>
          </div>
        </div>

        <p className="mx-auto mt-6 max-w-3xl text-center text-sm text-slate-500">
          In code this lives in{" "}
          <code>_portfolio_statistics()</code> and{" "}
          <code>_optimize_max_sharpe()</code> — the optimizer iterates weight
          combinations until it converges on the maximum-Sharpe portfolio.
        </p>
      </section>

      {/* Two engines: deterministic vs machine learning */}
      <section className="bg-white py-16">
        <div className="mx-auto max-w-6xl px-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold">Two engines, one app</h2>
            <p className="mx-auto mt-3 max-w-3xl text-slate-600">
              Investify now runs <strong>two complementary methods</strong>. The
              original <strong>deterministic optimizer</strong> powers the
              Prediction &amp; Performance pages. A new{" "}
              <strong>machine-learning strategy</strong> powers the Backtest page —
              and is validated <em>out-of-sample</em>, the gold standard for
              proving a model has real skill.
            </p>
          </div>

          <div className="mt-10 grid gap-6 lg:grid-cols-2">
            {/* Deterministic engine */}
            <div className="iv-card p-7">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-amber-100 text-lg text-amber-600">
                  <FaScaleBalanced />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Deterministic engine</h3>
                  <p className="text-xs font-medium text-slate-500">
                    Prediction &amp; Performance pages
                  </p>
                </div>
              </div>
              <ul className="mt-5 space-y-2.5 text-sm text-slate-600">
                <li>• <strong>Method:</strong> Markowitz mean-variance optimization (MPT).</li>
                <li>• <strong>Algorithm:</strong> SciPy SLSQP solver, max-Sharpe weights.</li>
                <li>• <strong>Answers:</strong> "Given these stocks, what's the optimal mix <em>right now?</em>"</li>
                <li>• <strong>Nature:</strong> closed-form, reproducible, no training.</li>
                <li>• <strong>Honest limit:</strong> it's an <em>in-sample</em> fit — it describes the past, it doesn't predict.</li>
              </ul>
              <button
                onClick={() => navigate("/performance")}
                className="mt-6 text-sm font-semibold text-customGreen-100 hover:underline"
              >
                See per-stock metrics →
              </button>
            </div>

            {/* ML engine */}
            <div className="iv-card p-7 ring-2 ring-customGreen-100/25">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-customGreen-100/15 text-lg text-customGreen-100">
                  <FaBrain />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Machine-learning strategy</h3>
                  <p className="text-xs font-medium text-slate-500">Backtest page</p>
                </div>
              </div>
              <ul className="mt-5 space-y-2.5 text-sm text-slate-600">
                <li>• <strong>Method:</strong> supervised learning that ranks stocks by predicted next-month return.</li>
                <li>• <strong>Algorithm:</strong> Gradient Boosting (or Random Forest) regressor.</li>
                <li>• <strong>Answers:</strong> "Which stocks will <em>outperform next month?</em>"</li>
                <li>• <strong>Nature:</strong> retrained <strong>walk-forward</strong>, evaluated <strong>out-of-sample</strong>.</li>
                <li>• <strong>Why it's credible:</strong> the model never sees the period it's scored on — no leakage.</li>
              </ul>
              <button
                onClick={() => navigate("/backtest")}
                className="mt-6 text-sm font-semibold text-customGreen-100 hover:underline"
              >
                Open the backtest →
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ML model + live out-of-sample performance */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="text-center">
          <span className="inline-block rounded-full bg-customGreen-100/15 px-4 py-1 text-sm font-semibold text-customGreen-100">
            ✨ Machine learning · out-of-sample
          </span>
          <h2 className="mt-4 text-3xl font-bold">
            The ML model — and how it actually performs
          </h2>
          <p className="mx-auto mt-3 max-w-3xl text-slate-600">
            Everything an interviewer would ask about the model: what it is, what
            it learns from, how it's validated, and the real numbers it produced
            on data it never trained on.
          </p>
        </div>

        {/* Model spec */}
        <div className="mt-10 grid gap-6 lg:grid-cols-3">
          <div className="iv-card p-6">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-customGreen-100/10 text-lg text-customGreen-100">
              <FaRobot />
            </div>
            <h3 className="mt-4 text-lg font-semibold">The model</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              A <strong>Gradient Boosting Regressor</strong> (scikit-learn) — an
              ensemble of shallow decision trees — predicts each stock's{" "}
              <strong>next-month return</strong>. A{" "}
              <strong>Random Forest</strong> is selectable as an alternative.
              Each month we go long the top-ranked names, equal-weighted.
            </p>
          </div>
          <div className="iv-card p-6">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-customGreen-100/10 text-lg text-customGreen-100">
              <FaCodeBranch />
            </div>
            <h3 className="mt-4 text-lg font-semibold">The features (no leakage)</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Six cross-sectional signals per stock, all from{" "}
              <em>past data only</em>: momentum over{" "}
              <strong>1, 3, 6 and 12 months</strong> and volatility over{" "}
              <strong>1 and 3 months</strong>. The label — next month's return —
              is unknown at prediction time, so it can't leak in.
            </p>
          </div>
          <div className="iv-card p-6">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-customGreen-100/10 text-lg text-customGreen-100">
              <FaClockRotateLeft />
            </div>
            <h3 className="mt-4 text-lg font-semibold">The validation</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              <strong>Walk-forward</strong>: at every monthly rebalance the model
              is retrained only on periods whose outcomes are already known, then
              tested on the <em>next</em>, unseen month. This is the honest way to
              prove skill — and it's exactly what the old leaky classifier failed.
            </p>
          </div>
        </div>

        {/* Live out-of-sample results */}
        <div className="iv-card mt-6 p-7">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-slate-900">
              Out-of-sample results
            </h3>
            {backtest && (
              <span className="text-xs font-medium text-slate-500">
                {backtest.n_periods} monthly rebalances · {backtest.start} →{" "}
                {backtest.end} · {backtest.universe_size} stocks
              </span>
            )}
          </div>

          {btError ? (
            <p className="mt-6 text-sm text-slate-500">
              Live backtest unavailable right now (start the Flask service to load
              it). See the <button onClick={() => navigate("/backtest")} className="font-semibold text-customGreen-100 hover:underline">Backtest page</button> for full results.
            </p>
          ) : !backtest ? (
            <div className="mt-6 flex items-center gap-3 text-sm text-slate-500">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-customGreen-100 border-t-transparent" />
              Training walk-forward models on the full history — first load can take
              ~90 seconds, then it's cached…
            </div>
          ) : (
            <>
              <div className="mt-3 rounded-xl bg-slate-50 p-4 text-sm text-slate-700">
                Over {backtest.n_periods} out-of-sample months, the ML strategy
                returned{" "}
                <strong className="text-slate-900">
                  {fmtPct(backtest.strategy.total_return)}
                </strong>{" "}
                vs the Nifty 50's{" "}
                <strong className="text-slate-900">
                  {fmtPct(backtest.benchmark.total_return)}
                </strong>
                {btBeats && btMultiple ? (
                  <> — about <strong className="text-customGreen-100">{btMultiple}× the index</strong>, on data the model never trained on.</>
                ) : (
                  <>.</>
                )}
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <PerfBlock
                  title="ML strategy"
                  accent="text-customGreen-100"
                  highlight
                  m={backtest.strategy}
                />
                <PerfBlock
                  title="Nifty 50 (buy & hold)"
                  accent="text-slate-700"
                  m={backtest.benchmark}
                />
              </div>

              <p className="mt-5 text-xs leading-relaxed text-slate-500">
                <strong>A note on "accuracy":</strong> this is a ranking/regression
                model, so the honest scorecard isn't classification accuracy — it's
                out-of-sample portfolio performance. <strong>Win rate</strong> (the
                share of months the strategy finished positive) is shown as a
                directional proxy. Caveats: long-only, monthly rebalance, no
                transaction costs — a research demonstrator, not a live trading
                system.
              </p>
            </>
          )}
        </div>
      </section>

      {/* User guide */}
      <section className="mx-auto max-w-4xl px-6 py-16">
        <h2 className="text-center text-3xl font-bold">Quick start guide</h2>
        <ol className="mt-10 space-y-4">
          {guide.map((g, i) => (
            <li key={i} className="iv-card flex items-start gap-4 p-5">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-customGreen-100 font-bold text-white">
                {i + 1}
              </span>
              <span className="pt-1 text-slate-700">{g}</span>
            </li>
          ))}
        </ol>
        <div className="mt-10 text-center">
          <button
            onClick={() => navigate("/prediction")}
            className="rounded-xl bg-customBlack-100 px-8 py-3 font-semibold text-white shadow-lg transition hover:scale-105 hover:bg-black"
          >
            Build my portfolio →
          </button>
        </div>
      </section>

      {/* Current data coverage */}
      <section className="bg-customBlack-100 py-12 text-white">
        <div className="mx-auto max-w-5xl px-6">
          <div className="text-center">
            <h2 className="text-2xl font-bold">Data we currently hold</h2>
            <p className="mt-2 text-slate-400">
              Live snapshot of the dataset powering your portfolio right now.
            </p>
          </div>
          {meta && !meta.error ? (
            <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <CoverageStat label="Date range" value={`${meta.start} → ${meta.end}`} />
              <CoverageStat label="Trading days" value={meta.trading_days?.toLocaleString()} />
              <CoverageStat label="Stocks tracked" value={meta.tickers} />
              <CoverageStat label="Source" value={meta.source} />
            </div>
          ) : (
            <p className="mt-8 text-center text-slate-400">
              {meta?.error
                ? "Could not load data coverage right now."
                : "Loading data coverage…"}
            </p>
          )}
          {meta && !meta.error && (
            <p className="mt-6 text-center text-sm text-slate-400">
              {meta.frequency} · {meta.refresh} · benchmarked against{" "}
              {(meta.benchmarks || []).join(" & ")}
            </p>
          )}
        </div>
      </section>

      <footer className="border-t border-slate-200 bg-white py-6 text-center text-sm text-slate-500">
        Investify · Educational demo · Not investment advice.
      </footer>
    </div>
  );
};

const CoverageStat = ({ label, value }) => (
  <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-center">
    <div className="text-lg font-bold text-customGreen-100">{value ?? "—"}</div>
    <div className="mt-1 text-sm text-slate-400">{label}</div>
  </div>
);

const PerfBlock = ({ title, m, accent, highlight }) => (
  <div className={`rounded-2xl border p-5 ${highlight ? "border-customGreen-100/30 bg-customGreen-100/5" : "border-slate-200 bg-white"}`}>
    <p className={`text-sm font-semibold ${accent}`}>{title}</p>
    <div className="mt-3 grid grid-cols-3 gap-3 text-center">
      <PerfStat label="Total" value={fmtPct(m.total_return)} />
      <PerfStat label="CAGR" value={fmtPct(m.cagr)} />
      <PerfStat label="Sharpe" value={fmtNum(m.sharpe)} />
      <PerfStat label="Max DD" value={fmtPct(m.max_drawdown)} negative />
      <PerfStat label="Volatility" value={fmtPct(m.volatility)} />
      <PerfStat label="Win rate" value={fmtPct(m.win_rate)} />
    </div>
  </div>
);

const PerfStat = ({ label, value, negative }) => (
  <div>
    <p className={`text-base font-bold ${negative ? "text-red-500" : "text-slate-900"}`}>
      {value}
    </p>
    <p className="mt-0.5 text-[11px] text-slate-500">{label}</p>
  </div>
);

const TechRow = ({ step, formula, detail }) => (
  <div className="iv-card p-6">
    <div className="flex flex-col gap-2">
      <h3 className="text-lg font-semibold text-slate-900">{step}</h3>
      {formula && (
        <code className="block overflow-x-auto rounded-lg bg-slate-900 px-4 py-3 text-sm text-customGreen-100">
          {formula}
        </code>
      )}
      <p className="text-sm leading-relaxed text-slate-600">{detail}</p>
    </div>
  </div>
);

export default Home;
