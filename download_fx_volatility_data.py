#!/usr/bin/env python3
"""Download FX intraday OHLC data for Ladder Pricing volatility calibration.

Usage:
  pip install -U yfinance pandas
  python download_fx_volatility_data.py
  python download_fx_volatility_data.py --ticker EURSEK=X --period 30d --interval 5m
  python download_fx_volatility_data.py --ticker EURSEK=X --period 7d --interval 1m

Upload the resulting ZIP file to ChatGPT. No calibration happens in this script.
"""
import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="EURSEK=X", help="Yahoo Finance ticker (default EURSEK=X)")
    parser.add_argument("--period", default="30d", help="Historical period, e.g. 30d (5m), 7d (1m)")
    parser.add_argument("--interval", default="5m", help="Bar frequency, e.g. 1m, 5m, 15m")
    parser.add_argument("--timezone", default="Europe/Copenhagen", help="Trading session time zone")
    parser.add_argument("--session-start", default="07:45", help="Local session start HH:MM")
    parser.add_argument("--session-end", default="17:15", help="Local session end HH:MM")
    parser.add_argument("--output", default="fx_volatility_data.zip", help="Output ZIP path")
    args = parser.parse_args()

    if args.session_start >= args.session_end:
        parser.error("Session start must precede session end.")

    print(f"Downloading {args.ticker}: {args.period} at {args.interval} resolution ...")
    try:
        df = yf.download(
            args.ticker, period=args.period, interval=args.interval,
            auto_adjust=False, progress=False, threads=False,
            prepost=False,
        )
    except Exception as exc:
        sys.exit(f"Download failed: {exc}")

    if df is None or df.empty:
        sys.exit("No data returned. Try --period 7d --interval 1m or --period 30d --interval 5m.")

    # yfinance versions sometimes return a MultiIndex even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda x: str(x).strip().lower().replace(" ", "_"))
    if not isinstance(df.index, pd.DatetimeIndex):
        sys.exit("Unexpected Yahoo Finance response: index is not timestamps.")
    if df.index.tz is None:
        sys.exit("Yahoo returned timezone-naive timestamps; cannot safely assign trading sessions.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index.tz_convert(args.timezone)
    df.index.name = "timestamp_local"

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing price columns: {missing}")
    df = df[required + [c for c in ["volume"] if c in df.columns]].copy()
    df = df.dropna(subset=["close"])
    all_bars = df.copy()
    # Time of each bar is its opening timestamp, in local trading timezone.
    local_clock = df.index.strftime("%H:%M")
    weekdays = df.index.dayofweek < 5
    df = df[(local_clock >= args.session_start) & (local_clock < args.session_end) & weekdays].copy()
    if df.empty:
        sys.exit("No bars in the chosen session. Check time zone and session settings.")

    df.insert(0, "timestamp_utc", df.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    df.insert(1, "session_date", df.index.strftime("%Y-%m-%d"))
    df = df.reset_index()
    # Local ISO timestamps keep DST offset; avoids ambiguity across DST transitions.
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    metadata = {
        "ticker": args.ticker,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "period_requested": args.period,
        "interval": args.interval,
        "timezone": args.timezone,
        "session_start_inclusive": args.session_start,
        "session_end_exclusive": args.session_end,
        "rows_all": len(all_bars),
        "rows_session": len(df),
        "first_bar_local": df["timestamp_local"].iloc[0],
        "last_bar_local": df["timestamp_local"].iloc[-1],
        "session_days": int(df["session_date"].nunique()),
        "source": "Yahoo Finance via yfinance",
        "notes": "Intraday OHLC bars. No overnight returns should be calculated; calculate returns within each session_date only. Yahoo intraday history is limited.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fx_intraday_session.csv", df.to_csv(index=False))
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {out.resolve()} | {len(df):,} bars over {metadata['session_days']} sessions")
    print(f"Range: {metadata['first_bar_local']} -> {metadata['last_bar_local']}")
    print("Upload this ZIP file in the conversation.")


if __name__ == "__main__":
    main()
