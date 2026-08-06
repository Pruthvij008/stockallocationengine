import { useEffect, useMemo, useState } from "react";
import Navbar from "./Navbar";
import axios from "axios";
import { SkeletonStats, SkeletonChart, SkeletonTable } from "./loader";
import { Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  Title,
} from "chart.js";
import { API_URL } from "../api";
import { useTheme } from "../useTheme";

ChartJS.register(LinearScale, PointElement, Tooltip, Legend, Title);

const pct = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : (Number(v) * 100).toFixed(2) + "%";
const num = (v) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : Number(v).toFixed(2);

const BENCHMARK = {
  indian: { key: "^NSEI", label: "Nifty 50" },
  us: { key: "^GSPC", label: "S&P 500" },
};

// `hide` marks columns dropped on narrow screens so the table stays readable.
const COLUMNS = [
  { label: "Stock", key: "ticker", type: "str", align: "left" },
  { label: "Return", key: "Return", type: "num" },
  { label: "Risk", key: "Risk", type: "num" },
  { label: "Sharpe", key: "sharpe", type: "num" },
  { label: "Beta", key: "beta", type: "num", hide: "sm" },
  { label: "Alpha", key: "alpha", type: "num", hide: "md" },
  { label: "CAPM ret.", key: "capm_ret", type: "num", hide: "lg" },
];

const hideClass = (hide) =>
  hide === "sm"
    ? "hidden sm:table-cell"
    : hide === "md"
    ? "hidden md:table-cell"
    : hide === "lg"
    ? "hidden lg:table-cell"
    : "";

const sharpeColor = (s) =>
  s >= 1 ? "#08bc54" : s >= 0.5 ? "#f59e0b" : "#ef4444";

const Performance = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [market, setMarket] = useState("indian");
  const [sort, setSort] = useState({ key: "sharpe", dir: "desc" });
  const [visibleRows, setVisibleRows] = useState(15);
  const { theme } = useTheme();

  useEffect(() => {
    axios
      .get(`${API_URL}/api/portfolio/summary`)
      .then((res) => setData(res.data))
      .catch((e) => {
        console.log("An error occurred", e);
        setError(true);
      });
  }, []);

  const bench = BENCHMARK[market];
  const isDark = theme === "dark";
  const gridColor = isDark ? "rgba(148,163,184,0.16)" : "rgba(15,23,42,0.08)";
  const tickColor = isDark ? "#a3aec6" : "#475569";

  // Rows for the selected market.
  const rows = useMemo(() => {
    if (!data) return [];
    const keys = Object.keys(data).filter((k) => {
      const isIndian = k.endsWith(".NS") || k === "^NSEI";
      return market === "indian" ? isIndian : !isIndian;
    });
    return keys.map((k) => ({
      ticker: k,
      display:
        k === bench.key ? `${bench.label} (benchmark)` : k.replace(".NS", ""),
      isMarket: k === bench.key,
      ...data[k],
    }));
  }, [data, market, bench]);

  const sortedRows = useMemo(() => {
    const arr = [...rows];
    arr.sort((a, b) => {
      // keep the benchmark pinned to the top
      if (a.isMarket) return -1;
      if (b.isMarket) return 1;
      let av = a[sort.key];
      let bv = b[sort.key];
      if (sort.key === "ticker") {
        av = a.display;
        bv = b.display;
        return sort.dir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      av = av === null || av === undefined ? -Infinity : Number(av);
      bv = bv === null || bv === undefined ? -Infinity : Number(bv);
      return sort.dir === "asc" ? av - bv : bv - av;
    });
    return arr;
  }, [rows, sort]);

  const stocks = rows.filter((r) => !r.isMarket);
  const stats = useMemo(() => {
    if (!stocks.length) return null;
    const avg = (f) =>
      stocks.reduce((s, r) => s + (Number(r[f]) || 0), 0) / stocks.length;
    const top = stocks.reduce((best, r) =>
      Number(r.sharpe) > Number(best.sharpe) ? r : best
    );
    return {
      count: stocks.length,
      avgReturn: avg("Return"),
      avgRisk: avg("Risk"),
      topName: top.display,
      topSharpe: top.sharpe,
    };
  }, [stocks]);

  // Scatter: risk (x) vs return (y), colored by Sharpe.
  const scatter = useMemo(() => {
    const pts = stocks
      .filter((r) => r.Return != null && r.Risk != null)
      .map((r) => ({
        x: Number(r.Risk) * 100,
        y: Number(r.Return) * 100,
        ticker: r.display,
        sharpe: Number(r.sharpe),
      }));
    return {
      datasets: [
        {
          label: "Stocks",
          data: pts,
          pointBackgroundColor: pts.map((p) => sharpeColor(p.sharpe)),
          pointRadius: 5,
          pointHoverRadius: 8,
        },
      ],
    };
  }, [stocks]);

  const scatterOptions = useMemo(
    () => ({
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              return `${p.ticker}: return ${p.y.toFixed(1)}%, risk ${p.x.toFixed(
                1
              )}%, Sharpe ${p.sharpe.toFixed(2)}`;
            },
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Risk — annualized volatility (%)",
            color: tickColor,
          },
          ticks: { color: tickColor },
          grid: { color: gridColor },
        },
        y: {
          title: {
            display: true,
            text: "Return — annualized (%)",
            color: tickColor,
          },
          ticks: { color: tickColor },
          grid: { color: gridColor },
        },
      },
      maintainAspectRatio: false,
    }),
    [tickColor, gridColor]
  );

  const onSort = (col) => {
    if (col.key === sort.key) {
      setSort((s) => ({ ...s, dir: s.dir === "asc" ? "desc" : "asc" }));
    } else {
      setSort({ key: col.key, dir: col.type === "str" ? "asc" : "desc" });
    }
  };

  return (
    <div className="iv-page min-h-screen">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 pt-24 pb-16 sm:px-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 className="iv-heading text-2xl font-bold sm:text-3xl">
              Stock performance
            </h1>
            <p className="iv-muted mt-2 max-w-2xl text-sm sm:text-base">
              Risk-and-return metrics from the live data we pull. Indian (NSE)
              stocks are benchmarked against the Nifty 50, the US sample against
              the S&amp;P 500. <strong>Beta</strong> = market sensitivity,{" "}
              <strong>Alpha</strong> = excess return, <strong>Sharpe</strong> =
              return per unit of risk.
            </p>
          </div>
          {/* Market filter */}
          <div className="iv-surface-muted flex shrink-0 rounded-xl p-1">
            {["indian", "us"].map((m) => (
              <button
                key={m}
                onClick={() => {
                  setMarket(m);
                  setVisibleRows(15);
                }}
                className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold capitalize transition sm:flex-none ${
                  market === m
                    ? "bg-white text-slate-900 shadow dark:bg-slate-700 dark:text-white"
                    : "iv-muted"
                }`}
              >
                {m === "indian" ? "🇮🇳 Indian" : "🇺🇸 US"}
              </button>
            ))}
          </div>
        </div>

        {error ? (
          <div className="iv-card mt-8 p-10 text-center text-red-500">
            Couldn't load performance data.
          </div>
        ) : !data ? (
          <div className="mt-8 space-y-6">
            <SkeletonStats />
            <SkeletonChart />
            <SkeletonTable />
          </div>
        ) : (
          <>
            {/* Summary stats */}
            {stats && (
              <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Stocks tracked" value={stats.count} />
                <Stat label="Avg. return p.a." value={pct(stats.avgReturn)} />
                <Stat label="Avg. risk p.a." value={pct(stats.avgRisk)} />
                <Stat
                  label="Best Sharpe"
                  value={num(stats.topSharpe)}
                  sub={stats.topName}
                  accent="text-customGreen-100"
                />
              </div>
            )}

            {/* Scatter chart */}
            <div className="iv-card mt-6 p-4 sm:p-6">
              <h3 className="iv-heading text-lg font-semibold">
                Risk vs. return
              </h3>
              <p className="iv-muted mb-4 text-sm">
                Each dot is a stock — up = higher return, right = higher risk.
                Green dots have the best risk-adjusted return (Sharpe ≥ 1).
              </p>
              <div className="h-[260px] sm:h-[360px]">
                <Scatter data={scatter} options={scatterOptions} />
              </div>
            </div>

            {/* Sortable table */}
            <div className="iv-card mt-6 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-customBlack-100 text-xs uppercase tracking-wide text-slate-200">
                      {COLUMNS.map((col) => {
                        const activeSort = sort.key === col.key;
                        return (
                          <th
                            key={col.key}
                            onClick={() => onSort(col)}
                            className={`cursor-pointer select-none whitespace-nowrap px-3 py-3 transition hover:bg-white/10 sm:px-4 ${
                              col.align === "left" ? "text-left" : "text-right"
                            } ${hideClass(col.hide)}`}
                          >
                            {col.label}
                            <span className="ml-1 text-customGreen-100">
                              {activeSort
                                ? sort.dir === "asc"
                                  ? "▲"
                                  : "▼"
                                : "⇅"}
                            </span>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRows.slice(0, visibleRows).map((r, i) => (
                      <tr
                        key={r.ticker}
                        className={`border-b border-[color:var(--iv-border)] ${
                          r.isMarket
                            ? "bg-amber-50 font-semibold dark:bg-amber-500/10"
                            : i % 2
                            ? "iv-surface-muted"
                            : ""
                        } hover:bg-customGreen-100/5`}
                      >
                        <td className="iv-heading whitespace-nowrap px-3 py-3 text-left font-medium sm:px-4">
                          {r.display}
                        </td>
                        <td className="iv-muted px-3 py-3 text-right sm:px-4">
                          {pct(r.Return)}
                        </td>
                        <td className="iv-muted px-3 py-3 text-right sm:px-4">
                          {pct(r.Risk)}
                        </td>
                        <td className="px-3 py-3 text-right font-semibold sm:px-4">
                          <span
                            className={
                              Number(r.sharpe) >= 1
                                ? "text-customGreen-100"
                                : "iv-muted"
                            }
                          >
                            {num(r.sharpe)}
                          </span>
                        </td>
                        <td
                          className={`iv-muted px-3 py-3 text-right sm:px-4 ${hideClass(
                            "sm"
                          )}`}
                        >
                          {num(r.beta)}
                        </td>
                        <td
                          className={`px-3 py-3 text-right sm:px-4 ${hideClass(
                            "md"
                          )} ${
                            r.alpha > 0
                              ? "text-customGreen-100"
                              : r.alpha < 0
                              ? "text-red-500"
                              : "iv-muted"
                          }`}
                        >
                          {pct(r.alpha)}
                        </td>
                        <td
                          className={`iv-muted px-3 py-3 text-right sm:px-4 ${hideClass(
                            "lg"
                          )}`}
                        >
                          {pct(r.capm_ret)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {visibleRows < sortedRows.length && (
                <div className="border-t border-[color:var(--iv-border)] p-4 text-center">
                  <button
                    onClick={() => setVisibleRows((n) => n + 15)}
                    className="rounded-lg bg-customBlack-100 px-6 py-2 font-medium text-white transition hover:bg-black dark:bg-customGreen-100 dark:hover:bg-green-600"
                  >
                    Show more ({sortedRows.length - visibleRows} left)
                  </button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const Stat = ({ label, value, sub, accent = "iv-heading" }) => (
  <div className="iv-card p-5">
    <p className="iv-subtle text-sm">{label}</p>
    <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
    {sub && <p className="iv-subtle mt-1 text-xs">{sub}</p>}
  </div>
);

export default Performance;
