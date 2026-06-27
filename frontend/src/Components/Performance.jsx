import { useEffect, useMemo, useState } from "react";
import Navbar from "./Navbar";
import axios from "axios";
import Loader from "./loader";
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

const COLUMNS = [
  { label: "Stock", key: "ticker", type: "str", align: "left" },
  { label: "Return", key: "Return", type: "num", fmt: pct },
  { label: "Risk", key: "Risk", type: "num", fmt: pct },
  { label: "Sharpe", key: "sharpe", type: "num", fmt: num },
  { label: "Beta", key: "beta", type: "num", fmt: num },
  { label: "Alpha", key: "alpha", type: "num", fmt: pct },
  { label: "CAPM ret.", key: "capm_ret", type: "num", fmt: pct },
];

const sharpeColor = (s) =>
  s >= 1 ? "#08bc54" : s >= 0.5 ? "#f59e0b" : "#ef4444";

const Performance = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [market, setMarket] = useState("indian");
  const [sort, setSort] = useState({ key: "sharpe", dir: "desc" });
  const [visibleRows, setVisibleRows] = useState(15);

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

  // Rows for the selected market.
  const rows = useMemo(() => {
    if (!data) return [];
    const keys = Object.keys(data).filter((k) => {
      const isIndian = k.endsWith(".NS") || k === "^NSEI";
      return market === "indian" ? isIndian : !isIndian;
    });
    return keys.map((k) => ({
      ticker: k,
      display: k === bench.key ? `${bench.label} (benchmark)` : k.replace(".NS", ""),
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
    const avg = (f) => stocks.reduce((s, r) => s + (Number(r[f]) || 0), 0) / stocks.length;
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

  const scatterOptions = {
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
        title: { display: true, text: "Risk — annualized volatility (%)" },
      },
      y: {
        title: { display: true, text: "Return — annualized (%)" },
      },
    },
    maintainAspectRatio: false,
  };

  const onSort = (col) => {
    if (col.key === sort.key) {
      setSort((s) => ({ ...s, dir: s.dir === "asc" ? "desc" : "asc" }));
    } else {
      setSort({ key: col.key, dir: col.type === "str" ? "asc" : "desc" });
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <div className="mx-auto max-w-6xl px-6 pt-24 pb-16">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Stock performance</h1>
            <p className="mt-2 max-w-2xl text-slate-600">
              Risk-and-return metrics from the live data we pull. Indian (NSE)
              stocks are benchmarked against the Nifty 50, the US sample against
              the S&amp;P 500. <strong>Beta</strong> = market sensitivity,{" "}
              <strong>Alpha</strong> = excess return, <strong>Sharpe</strong> =
              return per unit of risk.
            </p>
          </div>
          {/* Market filter */}
          <div className="flex rounded-xl bg-slate-200 p-1">
            {["indian", "us"].map((m) => (
              <button
                key={m}
                onClick={() => {
                  setMarket(m);
                  setVisibleRows(15);
                }}
                className={`rounded-lg px-4 py-2 text-sm font-semibold capitalize transition ${
                  market === m
                    ? "bg-white text-slate-900 shadow"
                    : "text-slate-500"
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
          <div className="iv-card mt-8 p-16">
            <Loader />
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
            <div className="iv-card mt-6 p-6">
              <h3 className="text-lg font-semibold text-slate-800">
                Risk vs. return
              </h3>
              <p className="mb-4 text-sm text-slate-500">
                Each dot is a stock — up = higher return, right = higher risk.
                Green dots have the best risk-adjusted return (Sharpe ≥ 1).
              </p>
              <div className="h-[360px]">
                <Scatter data={scatter} options={scatterOptions} />
              </div>
            </div>

            {/* Sortable table */}
            <div className="iv-card mt-6">
              <table className="w-full text-sm">
                <thead className="sticky top-16 z-20">
                  <tr className="bg-customBlack-100 text-xs uppercase tracking-wide text-slate-200">
                    {COLUMNS.map((col, i) => {
                      const activeSort = sort.key === col.key;
                      return (
                        <th
                          key={col.key}
                          onClick={() => onSort(col)}
                          className={`cursor-pointer select-none px-4 py-3 transition hover:bg-white/10 ${
                            col.align === "left" ? "text-left" : "text-right"
                          } ${i === 0 ? "rounded-tl-2xl" : ""} ${
                            i === COLUMNS.length - 1 ? "rounded-tr-2xl" : ""
                          }`}
                        >
                          {col.label}
                          <span className="ml-1 text-customGreen-100">
                            {activeSort ? (sort.dir === "asc" ? "▲" : "▼") : "⇅"}
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
                      className={`border-b border-slate-100 ${
                        r.isMarket
                          ? "bg-amber-50 font-semibold"
                          : i % 2
                          ? "bg-slate-50/60"
                          : "bg-white"
                      } hover:bg-customGreen-100/5`}
                    >
                      <td className="px-4 py-3 text-left font-medium text-slate-800">
                        {r.display}
                      </td>
                      <td className="px-4 py-3 text-right">{pct(r.Return)}</td>
                      <td className="px-4 py-3 text-right">{pct(r.Risk)}</td>
                      <td
                        className="px-4 py-3 text-right font-semibold"
                        style={{
                          color: Number(r.sharpe) >= 1 ? "#08bc54" : undefined,
                        }}
                      >
                        {num(r.sharpe)}
                      </td>
                      <td className="px-4 py-3 text-right">{num(r.beta)}</td>
                      <td
                        className={`px-4 py-3 text-right ${
                          r.alpha > 0
                            ? "text-customGreen-100"
                            : r.alpha < 0
                            ? "text-red-500"
                            : ""
                        }`}
                      >
                        {pct(r.alpha)}
                      </td>
                      <td className="px-4 py-3 text-right">{pct(r.capm_ret)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {visibleRows < sortedRows.length && (
                <div className="border-t border-slate-100 p-4 text-center">
                  <button
                    onClick={() => setVisibleRows((n) => n + 15)}
                    className="rounded-lg bg-customBlack-100 px-6 py-2 font-medium text-white transition hover:bg-black"
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

const Stat = ({ label, value, sub, accent = "text-slate-900" }) => (
  <div className="iv-card p-5">
    <p className="text-sm text-slate-500">{label}</p>
    <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
    {sub && <p className="mt-1 text-xs text-slate-400">{sub}</p>}
  </div>
);

export default Performance;
