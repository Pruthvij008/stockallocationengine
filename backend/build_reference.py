"""Generate `stock_reference.json`, a bundled fallback for company fundamentals.

Yahoo's `.info` endpoint refuses requests from datacenter IPs, so the hosted
service gets nothing back and the UI shows empty company cards. This script is
run locally (where the endpoint works) and its output is committed, giving the
app something accurate to fall back on.

Sector / industry / name are effectively static. Market cap, P/E and dividend
yield are point-in-time, so they're stamped with the date they were captured
and the UI labels them as a snapshot.

Usage:  python build_reference.py
"""

import datetime
import json

import pandas as pd
import yfinance as yf

OUT = "stock_reference.json"


def universe():
    header = pd.read_csv("stock_data.csv", header=[0, 1], index_col=0, nrows=1)
    return sorted({ticker for _, ticker in header.columns})


def main():
    tickers = universe()
    print(f"Fetching fundamentals for {len(tickers)} tickers…")

    out = {}
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).info or {}
        except Exception as exc:
            print(f"  [{i}/{len(tickers)}] {t}: FAILED ({exc})")
            continue

        if not info.get("sector") and not info.get("marketCap"):
            print(f"  [{i}/{len(tickers)}] {t}: no data")
            continue

        out[t] = {
            "Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Summary": (info.get("longBusinessSummary") or "")[:600] or None,
            "Market_Cap": info.get("marketCap"),
            "PE_Ratio": info.get("trailingPE"),
            "Divident_Yield": info.get("dividendYield"),
        }
        if i % 20 == 0:
            print(f"  [{i}/{len(tickers)}] ok")

    payload = {
        "captured": str(datetime.date.today()),
        "source": "Yahoo Finance via yfinance",
        "data": out,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"Wrote {OUT}: {len(out)}/{len(tickers)} tickers")


if __name__ == "__main__":
    main()
