import { useEffect, useMemo, useRef, useState } from "react";
import Navbar from "./Navbar";
import axios from "axios";
import Loader, { ProgressLoader } from "./loader";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  Filler,
} from "chart.js";
import { API_URL } from "../api";
import { useTheme } from "../useTheme";
import { formatDate, formatDateShort } from "../utils/format";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  Filler
);

const pct = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : (Number(v) * 100).toFixed(1) + "%";
const num = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : Number(v).toFixed(2);

const FEATURE_LABELS = {
  mom_1m: "1-month momentum",
  mom_3m: "3-month momentum",
  mom_6m: "6-month momentum",
  mom_12m: "12-month momentum",
  vol_1m: "1-month volatility",
  vol_3m: "3-month volatility",
};

const POLL_MS = 4000;

const Backtest = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [topK, setTopK] = useState(10);
  const [model, setModel] = useState("gbr");
  const { theme } = useTheme();
  const timerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    setProgress(0);

    // The backtest runs in the background on the server; it answers 202 with a
    // progress figure until the result is ready, so poll until we get a 200.
    const fetchOnce = () => {
      axios
        .get(`${API_URL}/api/portfolio/backtest`, {
          params: { top_k: topK, model },
          validateStatus: (s) => s === 200 || s === 202,
        })
        .then((res) => {
          if (cancelled) return;
          if (res.status === 202) {
            setProgress(Number(res.data?.progress) || 0);
            timerRef.current = setTimeout(fetchOnce, POLL_MS);
            return;
          }
          if (res.data?.error) setError(true);
          else setData(res.data);
          setLoading(false);
        })
        .catch((e) => {
          if (cancelled) return;
          console.log("Backtest request failed", e);
          setError(true);
          setLoading(false);
        });
    };

    fetchOnce();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [topK, model]);

  const isDark = theme === "dark";
  const gridColor = isDark ? "rgba(148,163,184,0.16)" : "rgba(15,23,42,0.08)";
  const tickColor = isDark ? "#a3aec6" : "#475569";

  const chart = useMemo(() => {
    if (!data?.curve) return null;
    return {
      labels: data.curve.map((p) => formatDateShort(p.date)),
      datasets: [
        {
          label: "ML strategy",
          data: data.curve.map((p) => p.strategy),
          borderColor: "#08bc54",
          backgroundColor: isDark
            ? "rgba(8, 188, 84, 0.14)"
            : "rgba(8, 188, 84, 0.08)",
          fill: true,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 2.5,
        },
        {
          label: "Nifty 50 (buy & hold)",
          data: data.curve.map((p) => p.benchmark),
          borderColor: isDark ? "#94a3b8" : "#64748b",
          borderDash: [6, 4],
          fill: false,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    };
  }, [data, isDark]);

  const chartOptions = useMemo(
    () => ({
      plugins: {
        legend: { position: "top", labels: { color: tickColor, boxWidth: 12 } },
        tooltip: {
          mode: "index",
          intersect: false,
          callbacks: {
            title: (items) => {
              const i = items?.[0]?.dataIndex;
              return i != null && data?.curve?.[i]
                ? formatDate(data.curve[i].date)
                : "";
            },
            label: (ctx) =>
              `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(2)}×`,
          },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 6, autoSkip: true, color: tickColor },
          grid: { display: false },
        },
        y: {
          title: {
            display: true,
            text: "Growth of ₹1 invested",
            color: tickColor,
          },
          ticks: { callback: (v) => `${v}×`, color: tickColor },
          grid: { color: gridColor },
        },
      },
      interaction: { mode: "index", intersect: false },
      maintainAspectRatio: false,
    }),
    [tickColor, gridColor, data]
  );

  const beatsBenchmark =
    data &&
    Number(data.strategy?.total_return) > Number(data.benchmark?.total_return);
  const maxImportance =
    data?.feature_importance?.reduce((m, f) => Math.max(m, f.importance), 0) || 1;

  return (
    <div className="iv-page min-h-screen">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 pt-24 pb-16 sm:px-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 rounded-full bg-customGreen-100/10 px-3 py-1 text-xs font-semibold text-customGreen-100">
              ✨ Machine learning · out-of-sample
            </div>
            <h1 className="iv-heading text-2xl font-bold sm:text-3xl">
              ML strategy backtest
            </h1>
            <p className="iv-muted mt-2 max-w-3xl text-sm sm:text-base">
              A gradient-boosting model ranks the universe each month using only
              past data, and we equal-weight its top picks for the month ahead.
              The model is <strong>retrained walk-forward</strong> and never sees
              the period it's judged on — so this curve is genuinely{" "}
              <strong>out-of-sample</strong>, not an in-sample fit. We compare it
              against simply buying and holding the Nifty 50.
            </p>
          </div>

          {/* Controls */}
          <div className="flex flex-row gap-4 lg:flex-col lg:gap-3">
            <div className="flex-1 lg:flex-none">
              <label className="iv-subtle mb-1 block text-xs font-semibold uppercase tracking-wide">
                Holdings
              </label>
              <div className="iv-surface-muted flex rounded-xl p-1">
                {[5, 10, 15].map((k) => (
                  <button
                    key={k}
                    onClick={() => setTopK(k)}
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold transition sm:px-4 ${
                      topK === k
                        ? "bg-white text-slate-900 shadow dark:bg-slate-700 dark:text-white"
                        : "iv-muted"
                    }`}
                  >
                    Top {k}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex-1 lg:flex-none">
              <label className="iv-subtle mb-1 block text-xs font-semibold uppercase tracking-wide">
                Model
              </label>
              <div className="iv-surface-muted flex rounded-xl p-1">
                {[
                  { key: "gbr", label: "Boosting" },
                  { key: "rf", label: "Forest" },
                ].map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setModel(m.key)}
                    className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold transition sm:px-4 ${
                      model === m.key
                        ? "bg-white text-slate-900 shadow dark:bg-slate-700 dark:text-white"
                        : "iv-muted"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {error ? (
          <div className="iv-card mt-8 p-10 text-center text-red-500">
            Couldn't run the backtest. Please try again in a moment.
          </div>
        ) : loading || !data ? (
          <div className="iv-card mt-8 p-8 sm:p-12">
            {progress > 0 ? (
              <ProgressLoader
                label="Training walk-forward models"
                progress={progress}
                sub="Each month is trained on past data only. This runs once and is then cached for the day."
              />
            ) : (
              <Loader
                label="Starting the backtest…"
                sub="Waking the ML service and loading price history."
              />
            )}
          </div>
        ) : (
          <>
            {/* Headline verdict */}
            <div
              className={`mt-8 rounded-2xl border p-5 ${
                beatsBenchmark
                  ? "border-customGreen-100/30 bg-customGreen-100/5"
                  : "border-amber-300/40 bg-amber-50 dark:bg-amber-500/10"
              }`}
            >
              <p className="iv-muted text-sm">
                Over {data.n_periods} monthly rebalances (
                <span className="iv-nowrap">
                  {formatDate(data.start)} → {formatDate(data.end)}
                </span>
                ), the ML strategy returned{" "}
                <strong className="iv-heading">
                  {pct(data.strategy.total_return)}
                </strong>{" "}
                vs the Nifty 50's{" "}
                <strong className="iv-heading">
                  {pct(data.benchmark.total_return)}
                </strong>
                {beatsBenchmark
                  ? " — it beat the index."
                  : " — it trailed the index."}
              </p>
            </div>

            {/* Metric comparison */}
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <MetricCard
                title="ML strategy"
                accent="text-customGreen-100"
                m={data.strategy}
                highlight
              />
              <MetricCard
                title="Nifty 50 (buy & hold)"
                accent="iv-heading"
                m={data.benchmark}
              />
            </div>

            {/* Equity curve */}
            <div className="iv-card mt-6 p-4 sm:p-6">
              <h3 className="iv-heading text-lg font-semibold">Equity curve</h3>
              <p className="iv-muted mb-4 text-sm">
                Growth of ₹1 invested at the start, rebalanced monthly. Both
                lines start at 1× on{" "}
                <span className="iv-nowrap">{formatDate(data.start)}</span>.
              </p>
              <div className="h-[260px] sm:h-[380px]">
                {chart && <Line data={chart} options={chartOptions} />}
              </div>
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              {/* Feature importance */}
              <div className="iv-card p-5 sm:p-6">
                <h3 className="iv-heading text-lg font-semibold">
                  What the model learned
                </h3>
                <p className="iv-muted mb-4 text-sm">
                  Feature importances from the final trained model — which
                  signals drove the stock ranking.
                </p>
                <div className="space-y-3">
                  {data.feature_importance?.map((f) => (
                    <div key={f.feature}>
                      <div className="flex justify-between text-sm">
                        <span className="iv-muted">
                          {FEATURE_LABELS[f.feature] || f.feature}
                        </span>
                        <span className="iv-heading font-semibold">
                          {(f.importance * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="iv-surface-muted mt-1 h-2 rounded-full">
                        <div
                          className="h-2 rounded-full bg-customGreen-100"
                          style={{
                            width: `${(f.importance / maxImportance) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Current holdings */}
              <div className="iv-card p-5 sm:p-6">
                <h3 className="iv-heading text-lg font-semibold">
                  What it would hold today
                </h3>
                <p className="iv-muted mb-4 text-sm">
                  The model's current top {data.top_k} picks, ranked by predicted
                  forward return.
                </p>
                <div className="flex flex-wrap gap-2">
                  {data.current_holdings?.map((h, i) => (
                    <span
                      key={h.ticker}
                      className="iv-surface-muted iv-heading inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm font-medium"
                    >
                      <span className="iv-subtle text-xs">#{i + 1}</span>
                      {h.ticker}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Model evaluation metrics */}
            {data.model_metrics && (
              <div className="iv-card mt-6 p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="iv-heading text-lg font-semibold">
                    Model evaluation — out-of-sample
                  </h3>
                  <span className="iv-subtle text-xs font-medium">
                    scored on{" "}
                    {data.model_metrics.n_predictions?.toLocaleString()} held-out
                    predictions
                  </span>
                </div>
                <p className="iv-muted mb-5 mt-1 text-sm">
                  The standard supervised-learning scorecard, computed on
                  predictions the model never trained on. Because this is a
                  return-ranking problem, three families of metrics matter.
                </p>

                <div className="grid gap-4 lg:grid-cols-3">
                  <EvalGroup
                    title="Regression error"
                    subtitle="How close the predicted return is"
                    rows={[
                      ["RMSE", pct(data.model_metrics.rmse)],
                      ["MAE", pct(data.model_metrics.mae)],
                      ["R²", num(data.model_metrics.r2)],
                    ]}
                  />
                  <EvalGroup
                    title="Direction (up / down)"
                    subtitle="Predicting the sign of next-month return"
                    rows={[
                      ["Accuracy", pct(data.model_metrics.accuracy)],
                      ["Precision", pct(data.model_metrics.precision)],
                      ["Recall", pct(data.model_metrics.recall)],
                      ["F1 score", num(data.model_metrics.f1)],
                      ["Base rate", pct(data.model_metrics.base_rate_up), true],
                    ]}
                  />
                  <EvalGroup
                    title="Ranking quality"
                    subtitle="What actually matters for trading"
                    rows={[
                      [
                        "Information Coeff.",
                        Number.isFinite(
                          Number(data.model_metrics.information_coefficient)
                        )
                          ? Number(
                              data.model_metrics.information_coefficient
                            ).toFixed(4)
                          : "—",
                      ],
                      ["IC hit rate", pct(data.model_metrics.ic_hit_rate)],
                      ["Precision@K", pct(data.model_metrics.precision_at_k)],
                    ]}
                  />
                </div>

                <div className="iv-muted mt-5 rounded-lg bg-amber-50 p-4 text-xs leading-relaxed dark:bg-amber-500/10">
                  <strong>How to read this (the honest version):</strong> the
                  pointwise accuracy is modest — R² is slightly negative and the
                  Information Coefficient is near zero, which is{" "}
                  <em>completely normal</em> for monthly stock returns (they're
                  mostly noise; published factor models live around IC
                  0.02–0.05). The strong equity curve above comes largely from a
                  persistent <strong>momentum / low-volatility tilt</strong> in a
                  rising market, not from precise per-stock forecasting. Showing
                  both scorecards — and not hiding the weak ones — is the point:
                  it's an honest demonstrator of the full ML pipeline, not a
                  claim of alpha.
                </div>
              </div>
            )}

            {/* Methodology */}
            <div className="iv-card mt-6 p-5 sm:p-6">
              <h3 className="iv-heading text-lg font-semibold">How this works</h3>
              <div className="iv-muted mt-3 grid gap-4 text-sm sm:grid-cols-2">
                <Step n="1" title="Features (no leakage)">
                  Each month we compute trailing momentum (1, 3, 6, 12-month) and
                  volatility (1, 3-month) for every stock — using only data up to
                  that date.
                </Step>
                <Step n="2" title="Label = future return">
                  The target is each stock's return over the next month, which is
                  unknown at prediction time, so it can't leak into the features.
                </Step>
                <Step n="3" title="Walk-forward training">
                  At every rebalance the model ({data.model}) is retrained only on
                  periods whose labels have already been realised.
                </Step>
                <Step n="4" title="Out-of-sample test">
                  We equal-weight the top {data.top_k} predicted names and measure
                  the realised next-month return — the model never sees it in
                  advance.
                </Step>
              </div>
              <p className="iv-subtle iv-surface-muted mt-4 rounded-lg p-3 text-xs">
                <strong>Honest caveats:</strong> {data.universe_size} NSE stocks,
                no transaction costs or slippage, long-only, monthly rebalance.
                This is a research demonstrator — it shows the modelling is sound
                out of sample, not a live trading system.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const MetricCard = ({ title, m, accent, highlight }) => (
  <div
    className={`iv-card p-5 sm:p-6 ${
      highlight ? "ring-2 ring-customGreen-100/20" : ""
    }`}
  >
    <p className={`text-sm font-semibold ${accent}`}>{title}</p>
    <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
      <Metric label="Total return" value={pct(m.total_return)} />
      <Metric label="CAGR" value={pct(m.cagr)} />
      <Metric label="Sharpe ratio" value={num(m.sharpe)} />
      <Metric label="Max drawdown" value={pct(m.max_drawdown)} negative />
      <Metric label="Volatility p.a." value={pct(m.volatility)} />
      <Metric label="Win rate" value={pct(m.win_rate)} />
    </div>
  </div>
);

const Metric = ({ label, value, negative }) => (
  <div>
    <p className="iv-subtle text-xs">{label}</p>
    <p
      className={`mt-0.5 text-lg font-bold sm:text-xl ${
        negative ? "text-red-500" : "iv-heading"
      }`}
    >
      {value}
    </p>
  </div>
);

const EvalGroup = ({ title, subtitle, rows }) => (
  <div className="iv-surface-muted rounded-2xl p-5">
    <p className="iv-heading text-sm font-semibold">{title}</p>
    <p className="iv-subtle mt-0.5 text-xs">{subtitle}</p>
    <div className="mt-4 space-y-2.5">
      {rows.map(([label, value, muted]) => (
        <div key={label} className="flex items-center justify-between text-sm">
          <span className={muted ? "iv-subtle" : "iv-muted"}>{label}</span>
          <span className={`font-bold ${muted ? "iv-subtle" : "iv-heading"}`}>
            {value}
          </span>
        </div>
      ))}
    </div>
  </div>
);

const Step = ({ n, title, children }) => (
  <div className="flex gap-3">
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-customBlack-100 text-xs font-bold text-white dark:bg-customGreen-100">
      {n}
    </div>
    <div>
      <p className="iv-heading font-semibold">{title}</p>
      <p className="mt-0.5">{children}</p>
    </div>
  </div>
);

export default Backtest;
