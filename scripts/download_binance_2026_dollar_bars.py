#!/usr/bin/env python3
"""Download Binance Vision COIN-M BTCUSD_PERP aggTrades and build $500,000 Dollar Bars parquet."""
import argparse
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _month_range(start_ym, end_ym):
    return [str(p) for p in pd.period_range(start_ym, end_ym, freq='M')]


def fetch_month(ym, archive_dir, sym='BTCUSD_PERP', market='futures/cm'):
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    name = f'{sym}-aggTrades-{ym}.zip'
    zpath = archive_dir / name
    url = f'https://data.binance.vision/data/{market}/monthly/aggTrades/{sym}/{name}'
    if zpath.exists():
        print(f'  [cache] {name} ({zpath.stat().st_size / 1e6:.1f} MB)', flush=True)
        return zpath
    print(f'  [download] {url} ...', flush=True)
    part = zpath.with_suffix('.zip.part')
    try:
        urllib.request.urlretrieve(url, part)
        part.replace(zpath)
        print(f'  [done] {name} ({zpath.stat().st_size / 1e6:.1f} MB)', flush=True)
        return zpath
    except urllib.error.HTTPError as exc:
        part.unlink(missing_ok=True)
        if exc.code == 404:
            print(f'  [skip 404] {name} not available yet', flush=True)
            return None
        raise


def _read_aggtrade_chunks(zpath, chunksize=2_000_000):
    cols = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'transact_time', 'is_buyer_maker']
    with zipfile.ZipFile(zpath) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
        with zf.open(csv_name) as fh:
            head = fh.readline().decode('utf-8', 'ignore').lower()
        has_header = 'price' in head or 'transact_time' in head
        with zf.open(csv_name) as fh:
            yield from pd.read_csv(
                fh, header=0 if has_header else None,
                names=None if has_header else cols,
                chunksize=chunksize,
                usecols=['agg_trade_id', 'price', 'quantity', 'transact_time', 'is_buyer_maker'] if has_header else [0, 1, 2, 5, 6],
            )


def convert_to_dollar_bars(files, outfile, threshold=500000.0, contract_usd=100.0):
    bars = []
    total = 0
    last_ts_ns = -1
    cur_open = np.nan; cur_high = -np.inf; cur_low = np.inf
    cur_open_ts = -1
    cur_overshoot = 0.0; cur_usd = 0.0; cur_signed = 0.0; cur_buys = 0.0; cur_trades = 0

    for path in files:
        monthly = 0
        for chunk in _read_aggtrade_chunks(path):
            chunk.columns = [c.strip().lower() for c in chunk.columns]
            q = pd.DataFrame({
                'price': pd.to_numeric(chunk['price'], errors='coerce'),
                'qty': pd.to_numeric(chunk['quantity'], errors='coerce'),
                'ts_ms': pd.to_numeric(chunk['transact_time'], errors='coerce'),
                'is_buyer_maker': chunk['is_buyer_maker'].astype(str).str.lower().isin(['true', '1']),
            }).dropna()
            q = q[(q.price > 0) & (q.qty > 0)]
            if q.empty:
                continue
            monthly += len(q); total += len(q)

            prices = q['price'].to_numpy(dtype='float64')
            qtys = q['qty'].to_numpy(dtype='float64')
            ts_ns_arr = q['ts_ms'].to_numpy(dtype='int64') * 1_000_000
            is_sell = q['is_buyer_maker'].to_numpy(dtype=bool)
            usd_arr = qtys * contract_usd
            sign_arr = np.where(is_sell, -1.0, 1.0)
            signed_arr = sign_arr * usd_arr
            buy_arr = np.where(sign_arr > 0, usd_arr, 0.0)
            n_chunk = len(prices)

            cum_usd = cur_overshoot + np.cumsum(usd_arr, dtype='float64')
            k_max = int(cum_usd[-1] // threshold)
            if k_max == 0:
                if cur_trades == 0:
                    cur_open = float(prices[0]); cur_open_ts = int(ts_ns_arr[0])
                cur_high = max(cur_high, float(prices.max()))
                cur_low = min(cur_low, float(prices.min()))
                cur_usd += float(usd_arr.sum()); cur_signed += float(signed_arr.sum())
                cur_buys += float(buy_arr.sum()); cur_trades += n_chunk
                cur_overshoot = float(cum_usd[-1])
            else:
                levels = threshold * np.arange(1, k_max + 1, dtype='float64')
                close_ix = np.unique(np.searchsorted(cum_usd, levels, side='left'))
                close_ix = close_ix[close_ix < n_chunk]
                starts = np.r_[0, close_ix[:-1] + 1]
                end = int(close_ix[-1]) + 1
                b_open = prices[starts]; b_close = prices[close_ix]
                b_high = np.maximum.reduceat(prices[:end], starts); b_low = np.minimum.reduceat(prices[:end], starts)
                b_usd = np.add.reduceat(usd_arr[:end], starts); b_signed = np.add.reduceat(signed_arr[:end], starts)
                b_buys = np.add.reduceat(buy_arr[:end], starts)
                b_trades = (close_ix - starts + 1).astype(np.int64)
                b_open_ts = ts_ns_arr[starts]; b_close_ts = ts_ns_arr[close_ix]
                if cur_trades > 0:
                    b_open[0] = cur_open; b_high[0] = max(cur_high, float(b_high[0])); b_low[0] = min(cur_low, float(b_low[0]))
                    b_usd[0] += cur_usd; b_signed[0] += cur_signed; b_buys[0] += cur_buys
                    b_trades[0] += cur_trades; b_open_ts[0] = cur_open_ts
                for j in range(len(close_ix)):
                    ct = int(b_close_ts[j])
                    if ct <= last_ts_ns:
                        ct = last_ts_ns + 1
                    last_ts_ns = ct
                    dur = max((ct - int(b_open_ts[j])) / 1e9, 1e-3)
                    bars.append((ct, float(b_open[j]), float(b_high[j]), float(b_low[j]), float(b_close[j]),
                                 float(b_usd[j]), float(b_signed[j]), float(b_buys[j]), int(b_trades[j]), dur))
                cur_overshoot = float(cum_usd[-1] - np.floor(cum_usd[close_ix[-1]] / threshold) * threshold)
                if end < n_chunk:
                    cur_open = float(prices[end]); cur_high = float(prices[end:].max()); cur_low = float(prices[end:].min())
                    cur_open_ts = int(ts_ns_arr[end]); cur_usd = float(usd_arr[end:].sum())
                    cur_signed = float(signed_arr[end:].sum()); cur_buys = float(buy_arr[end:].sum())
                    cur_trades = int(n_chunk - end)
                else:
                    cur_open = np.nan; cur_high = -np.inf; cur_low = np.inf
                    cur_open_ts = -1; cur_usd = 0.0; cur_signed = 0.0; cur_buys = 0.0; cur_trades = 0
        print(f'  Converted {path.name}: aggTrades={monthly:,} -> total dollarBars={len(bars):,}', flush=True)

    df = pd.DataFrame(bars, columns=[
        'timestamp_ns', 'open', 'high', 'low', 'close',
        'volume', 'signed_notional_usd', 'buy_notional_usd', 'trade_count', 'duration_sec'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp_ns'], unit='ns', utc=True)
    df['notional_usd'] = df['volume']
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'notional_usd',
             'signed_notional_usd', 'buy_notional_usd', 'trade_count', 'duration_sec']]
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'Saved {len(df):,} Dollar Bars ({total:,} aggTrades) to {out}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-ym', type=str, default='2025-11')
    parser.add_argument('--end-ym', type=str, default='2026-10')
    parser.add_argument('--archive-dir', type=str, default='data_cache/binance_cm_aggTrades')
    parser.add_argument('--out', type=str, default='data_cache/BTCUSD_DOLLAR_500K_2026.parquet')
    args = parser.parse_args()

    t0 = time.time()
    months = _month_range(args.start_ym, args.end_ym)
    print(f'Fetching {len(months)} months ({args.start_ym} -> {args.end_ym}) from Binance Vision...', flush=True)
    files = []
    for ym in months:
        zp = fetch_month(ym, args.archive_dir)
        if zp is not None:
            files.append(zp)
    if not files:
        raise RuntimeError('No archives downloaded')
    print(f'Converting {len(files)} monthly archives to $500,000 Dollar Bars...', flush=True)
    convert_to_dollar_bars(files, args.out)
    print(f'All done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
