import React, { useState } from "react";
import Navbar from "./Navbar";
import { Pie, Line } from "react-chartjs-2";
import axios from "axios";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
} from "chart.js";
import { API_URL } from "../api";
import {
  sanitizeAmountInput,
  parseAmount,
  formatINR,
  formatCompactINR,
  formatNum,
  formatDate,
} from "../utils/format";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler
);

const PALETTE = (n) =>
  Array.from({ length: n }, (_, i) => `hsl(${Math.round((360 * i) / n)}, 65%, 55%)`);

const LoadingSpinner = () => (
  <div className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-[color:var(--iv-page)]/85 backdrop-blur">
    <div className="h-16 w-16 animate-spin rounded-full border-4 border-customGreen-100 border-t-transparent" />
    <p className="text-lg font-medium iv-heading">
      Crunching live market data & optimizing your portfolio…
    </p>
  </div>
);

const Prediction = () => {
  const [predictionData, setPredictionData] = useState(null);
  const [amountText, setAmountText] = useState("");
  const [amtFocused, setAmtFocused] = useState(false);
  const [period, setPeriod] = useState(5);
  const [risk, setRisk] = useState("");
  const [numStocks, setNumStocks] = useState(6);
  const [graphType, setGraphType] = useState("allocation");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const parsedAmount = parseAmount(amountText);

  const onAmountChange = (e) => setAmountText(sanitizeAmountInput(e.target.value));
  const onAmountBlur = () => {
    setAmtFocused(false);
    // Normalize "1k" / "2.5cr" to the plain resolved number for clean editing.
    if (parsedAmount > 0) setAmountText(String(parsedAmount));
  };
  const onAmountKeyDown = (e) => {
    if (e.key === "Enter") e.target.blur();
  };
  // While editing show the raw text; once done show comma-grouped (1,000).
  const amountDisplay = amtFocused
    ? amountText
    : parsedAmount > 0
    ? parsedAmount.toLocaleString("en-IN")
    : amountText;

  const calculateStock = async () => {
    if (!parsedAmount || parsedAmount <= 0) {
      setErr("Please enter a valid investment amount.");
      return;
    }
    if (!risk) {
      setErr("Please select a risk tolerance.");
      return;
    }
    if (!period || period <= 0) {
      setErr("Please enter an investment period in years.");
      return;
    }

    setLoading(true);
    setErr("");
    setPredictionData(null);
    try {
      const response = await axios.post(`${API_URL}/api/portfolio/prediction`, {
        investment_amount: parsedAmount,
        investment_period: period,
        risk_tolerance: risk,
        expected_return: 15,
        num_stocks: numStocks,
      });
      const data = response?.data?.data;
      if (!data) {
        setErr("Could not generate a portfolio. Please try again.");
      } else {
        setPredictionData(data);
      }
    } catch (error) {
      console.log("Prediction failed", error);
      setErr("Something went wrong fetching your portfolio. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // ---- Derived projection numbers ----
  const principal = predictionData ? Number(predictionData.investment_amount) : 0;
  const annReturn = predictionData
    ? Number(predictionData.optimized_portfolio_annualized_return)
    : 0;
  const futureValue = principal * Math.pow(1 + annReturn / 100, period);
  const totalGains = futureValue - principal;

  const weights =
    predictionData?.optimized_portfolio_weights_in_percentage
      ?.Optimized_Portfolio_Weights_in_Percentage || [];
  const rows =
    predictionData?.annualized_return_and_risk_in_percentage
      ?.Annualized_Return_and_Risk || [];
  const stockRows = rows.filter((r) => r.Ticker !== "MP");
  const overall = rows.find((r) => r.Ticker === "MP");

  const pieData = {
    labels: weights.map((w) => w.Ticker),
    datasets: [
      {
        data: weights.map((w) => parseFloat(w.Weight)),
        backgroundColor: PALETTE(weights.length),
        borderColor: "#fff",
        borderWidth: 2,
      },
    ],
  };

  const pieOptions = {
    plugins: {
      legend: { position: "right", labels: { boxWidth: 14, font: { size: 12 } } },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.label}: ${ctx.parsed.toFixed(2)}%`,
        },
      },
    },
    maintainAspectRatio: false,
  };

  const years = Array.from({ length: period }, (_, i) => i + 1);
  const lineData = {
    labels: years.map((y) => `Year ${y}`),
    datasets: [
      {
        label: "Projected value",
        data: years.map((y) => principal * Math.pow(1 + annReturn / 100, y)),
        borderColor: "#08bc54",
        backgroundColor: "rgba(8,188,84,0.12)",
        fill: true,
        tension: 0.3,
        pointBackgroundColor: "#08bc54",
      },
    ],
  };

  const lineOptions = {
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: (ctx) => ` ${formatINR(ctx.parsed.y)}` },
      },
    },
    scales: {
      y: {
        ticks: { callback: (v) => formatCompactINR(v) },
      },
    },
    maintainAspectRatio: false,
  };

  const inputClass =
    "iv-heading mt-2 block w-full rounded-xl border-2 border-[color:var(--iv-border)] bg-[color:var(--iv-surface-muted)] px-4 py-3 outline-none transition focus:border-customGreen-100 focus:bg-[color:var(--iv-surface)]";

  return (
    <div className="iv-page min-h-screen">
      <Navbar />
      {loading && <LoadingSpinner />}

      <div className="mx-auto max-w-6xl px-4 sm:px-6 pt-24 pb-16">
        <h1 className="iv-heading text-2xl font-bold sm:text-3xl">Build your portfolio</h1>
        <p className="mt-2 iv-muted">
          Enter your details and we'll allocate a diversified, risk-matched
          basket of stocks using live data.
        </p>

        {/* Input form */}
        <div className="iv-card mt-8 p-6 md:p-8">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {/* Amount */}
            <div>
              <label className="block text-sm font-semibold iv-heading">
                Investment amount
              </label>
              <input
                type="text"
                inputMode="decimal"
                value={amountDisplay}
                onChange={onAmountChange}
                onFocus={() => setAmtFocused(true)}
                onBlur={onAmountBlur}
                onKeyDown={onAmountKeyDown}
                placeholder="e.g. 5L, 2.5Cr, 1M"
                className={inputClass}
              />
              <p className="mt-1 h-5 text-xs iv-muted">
                {amountText
                  ? `= ${formatINR(parsedAmount)}`
                  : "Type k=thousand, l=lakh, m=million, c=crore, b=billion"}
              </p>
            </div>

            {/* Period */}
            <div>
              <label className="block text-sm font-semibold iv-heading">
                Investment period
              </label>
              <div className="relative">
                <input
                  type="number"
                  min="1"
                  max="40"
                  value={period}
                  onChange={(e) => setPeriod(Number(e.target.value))}
                  className={inputClass + " pr-16"}
                />
                <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium iv-subtle">
                  years
                </span>
              </div>
              <p className="mt-1 h-5 text-xs iv-muted">
                How long you'll stay invested.
              </p>
            </div>

            {/* Risk */}
            <div>
              <label className="block text-sm font-semibold iv-heading">
                Risk tolerance
              </label>
              <select
                value={risk}
                onChange={(e) => setRisk(e.target.value)}
                className={inputClass}
              >
                <option value="">Select…</option>
                <option value="low">Low — steadier, lower swings</option>
                <option value="medium">Medium — balanced</option>
                <option value="high">High — bigger swings & upside</option>
              </select>
              <p className="mt-1 h-5 text-xs iv-muted">
                Sets how volatile your stocks can be.
              </p>
            </div>

            {/* Num stocks */}
            <div>
              <label className="block text-sm font-semibold iv-heading">
                Number of stocks
              </label>
              <select
                value={numStocks}
                onChange={(e) => setNumStocks(Number(e.target.value))}
                className={inputClass}
              >
                {Array.from({ length: 11 }, (_, i) => i + 2).map((n) => (
                  <option key={n} value={n}>
                    {n} stocks
                  </option>
                ))}
              </select>
              <p className="mt-1 h-5 text-xs iv-muted">
                How many names to spread across.
              </p>
            </div>
          </div>

          <div className="mt-6 flex flex-col items-start gap-3 sm:flex-row sm:items-center">
            <button
              onClick={calculateStock}
              disabled={loading}
              className="rounded-xl bg-customGreen-100 px-8 py-3 font-semibold text-white shadow transition hover:bg-green-600 disabled:opacity-60"
            >
              Calculate
            </button>
            {err && <p className="text-sm font-medium text-red-500">{err}</p>}
          </div>
        </div>

        {/* Results */}
        {predictionData && (
          <>
            {predictionData.data_as_of && (
              <p className="mt-6 text-sm iv-muted">
                Live market data as of{" "}
                <span className="iv-nowrap font-medium iv-heading">
                  {formatDate(predictionData.data_as_of)}
                </span>
              </p>
            )}

            {/* Stat cards */}
            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard label="Total invested" value={formatINR(principal)} />
              <StatCard
                label={`Projected gains (${period}y)`}
                value={formatINR(totalGains)}
                accent="text-customGreen-100"
              />
              <StatCard label="Future value" value={formatINR(futureValue)} />
              <StatCard
                label="Portfolio return p.a."
                value={`${formatNum(annReturn)}%`}
                sub={`Sharpe ${formatNum(
                  predictionData.optimized_portfolio_sharper_ratio
                )}`}
              />
            </div>

            {/* Chart + holdings */}
            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <div className="iv-card p-6">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-lg font-semibold iv-heading">
                    {graphType === "allocation"
                      ? "Allocation"
                      : "Projected growth"}
                  </h3>
                  <div className="flex rounded-lg iv-surface-muted p-1 text-sm">
                    {["allocation", "growth"].map((t) => (
                      <button
                        key={t}
                        onClick={() => setGraphType(t)}
                        className={`rounded-md px-3 py-1 capitalize transition ${
                          graphType === t
                            ? "iv-surface iv-heading font-semibold shadow"
                            : "iv-muted"
                        }`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="h-[340px]">
                  {graphType === "allocation" ? (
                    <Pie data={pieData} options={pieOptions} />
                  ) : (
                    <Line data={lineData} options={lineOptions} />
                  )}
                </div>
              </div>

              {/* Holdings table */}
              <div className="iv-card overflow-hidden p-6">
                <h3 className="mb-4 text-lg font-semibold iv-heading">
                  Holdings
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[color:var(--iv-border)] text-left text-xs uppercase tracking-wide iv-muted">
                        <th className="py-2 pr-2">Stock</th>
                        <th className="py-2 px-2 text-right">Weight</th>
                        <th className="py-2 px-2 text-right">Return</th>
                        <th className="py-2 px-2 text-right">Risk</th>
                        <th className="py-2 pl-2 text-right">Sharpe</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stockRows.map((r) => {
                        const w = weights.find((x) => x.Ticker === r.Ticker);
                        return (
                          <tr
                            key={r.Ticker}
                            className="border-b border-[color:var(--iv-border)] hover:bg-[color:var(--iv-surface-muted)]"
                          >
                            <td className="py-2 pr-2 font-medium iv-heading">
                              {r.Ticker.replace(".NS", "")}
                            </td>
                            <td className="py-2 px-2 text-right font-semibold text-customGreen-100">
                              {w ? w.Weight : "—"}
                            </td>
                            <td className="py-2 px-2 text-right">{r.Return}</td>
                            <td className="py-2 px-2 text-right">{r.Risk}</td>
                            <td className="py-2 pl-2 text-right">{r.Sharpe}</td>
                          </tr>
                        );
                      })}
                      {overall && (
                        <tr className="iv-surface-muted font-semibold">
                          <td className="py-2 pr-2">Portfolio</td>
                          <td className="py-2 px-2 text-right">100%</td>
                          <td className="py-2 px-2 text-right">
                            {overall.Return}
                          </td>
                          <td className="py-2 px-2 text-right">{overall.Risk}</td>
                          <td className="py-2 pl-2 text-right">
                            {overall.Sharpe}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Stock info */}
            {predictionData.stock_info && (
              <div className="mt-6">
                <h3 className="mb-4 text-lg font-semibold iv-heading">
                  About these companies
                </h3>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {Object.entries(predictionData.stock_info).map(
                    ([ticker, info]) => (
                      <div key={ticker} className="iv-card iv-card-hover p-5">
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <h4 className="text-base font-bold iv-heading">
                              {ticker.replace(".NS", "")}
                            </h4>
                            {info.Name && (
                              <p className="iv-subtle truncate text-xs">
                                {info.Name}
                              </p>
                            )}
                          </div>
                          {info.Sector && (
                            <span className="shrink-0 rounded-full bg-customGreen-100/10 px-2 py-0.5 text-xs font-medium text-customGreen-100">
                              {info.Sector}
                            </span>
                          )}
                        </div>
                        <dl className="mt-3 space-y-1 text-sm iv-muted">
                          <Row k="Industry" v={info.Industry || "—"} />
                          <Row
                            k="Market cap"
                            v={
                              info.Market_Cap
                                ? formatCompactINR(info.Market_Cap)
                                : "—"
                            }
                          />
                          <Row k="P/E ratio" v={formatNum(info.PE_Ratio)} />
                          <Row
                            k="Dividend yield"
                            v={formatNum(info.Divident_Yield)}
                          />
                        </dl>
                        {info.source === "snapshot" && info.as_of && (
                          <p className="iv-subtle mt-3 text-[11px]">
                            Fundamentals as of {formatDate(info.as_of)} — the live
                            feed blocks requests from hosted servers.
                          </p>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {!predictionData && !loading && (
          <div className="iv-card mt-8 flex flex-col items-center justify-center gap-2 p-16 text-center iv-subtle">
            <p className="text-lg">Your optimized portfolio will appear here.</p>
            <p className="text-sm">
              Fill in the form above and hit Calculate.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const StatCard = ({ label, value, sub, accent = "iv-heading" }) => (
  <div className="iv-card p-5">
    <p className="text-sm iv-muted">{label}</p>
    <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
    {sub && <p className="mt-1 text-xs iv-subtle">{sub}</p>}
  </div>
);

const Row = ({ k, v }) => (
  <div className="flex justify-between gap-4">
    <dt className="iv-subtle">{k}</dt>
    <dd className="text-right font-medium iv-heading">{v}</dd>
  </div>
);

export default Prediction;
