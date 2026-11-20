#!/usr/bin/env python3
"""Fetch MT5 BTCUSDc ticks, convert to proxy Dollar Bars, and run V8 realistic backtest."""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v8_btc_dollar as bt


MT5_PATH = r'C:\Program Files\MetaTrader 5\terminal64.exe'


def fetch_mt5_ticks(symbol='BTCUSDc', days=5):
    if not mt5.initialize(path=MT5_PATH):
        raise RuntimeError(f'MT5 initialize failed: {mt5.last_error()}')
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f'symbol_select({symbol}) failed: {mt5.last_error()}')
        si = mt5.symbol_info(symbol)
        ai = mt5.account_info()
        tick = mt5.symbol_info_tick(symbol)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        chunks = []
        cur = start
        while cur < end:
            nxt = min(cur + timedelta(hours=12), end)
            t = mt5.copy_ticks_range(symbol, cur, nxt, mt5.COPY_TICKS_ALL)
            if t is not None and len(t) > 0:
                chunks.append(t)
            cur = nxt
        if not chunks:
            raise RuntimeError(f'No ticks returned for {symbol}')
        ticks = np.concatenate(chunks)

        live_spread_usd = float(tick.ask - tick.bid) if (tick and tick.ask > 0 and tick.bid > 0) else float(si.spread * si.point)
        specs = {
            'symbol': symbol,
            'server': ai.server if ai else bt.BROKER_CFG['server'],
            'account_type': ai.name if ai else bt.BROKER_CFG['account_type'],
            'initial_balance_usc': float(ai.balance) if ai else bt.BROKER_CFG['initial_balance_usc'],
            'leverage': int(ai.leverage) if ai else bt.BROKER_CFG['leverage'],
            'margin_call_pct': float(ai.margin_so_call) if ai else bt.BROKER_CFG['margin_call_pct'],
            'stop_out_pct': float(ai.margin_so_so) if ai else bt.BROKER_CFG['stop_out_pct'],
            'spread_usd': live_spread_usd,
            'contract_size': float(si.trade_contract_size),
            'volume_min': float(si.volume_min),
            'volume_step': float(si.volume_step),
            'volume_max': float(si.volume_max),
            'point': float(si.point),
        }
        return ticks, specs
    finally:
        mt5.shutdown()


def ticks_to_proxy_dollar_bars(ticks, point=0.01, target_sec_per_bar=4.34):
    t_msc = ticks['time_msc'].astype(np.int64)
    bid = ticks['bid'].astype(np.float64)
    ask = ticks['ask'].astype(np.float64)
    valid = (bid > 0) & (ask > 0) & (ask >= bid)
    t_msc, bid, ask = t_msc[valid], bid[valid], ask[valid]

    mid = 0.5 * (bid + ask)
    dm = np.diff(mid, prepend=mid[0])
    # Lee-Ready tick rule: forward-fill zero price changes
    sign = np.sign(dm)
    if sign[0] == 0:
        sign[0] = 1.0
    nz = sign != 0
    idx = np.where(nz, np.arange(len(sign)), 0)
    np.maximum.accumulate(idx, out=idx)
    sign = sign[idx]

    # Activity-weighted proxy notional per tick
    tick_notional = 1000.0 * (1.0 + 0.25 * np.abs(dm) / max(point, 1e-6))
    dt_s = np.diff(t_msc, prepend=t_msc[0]) / 1000.0
    active_sec = max(float(dt_s[dt_s <= 60.0].sum()), 1.0)
    target_bars = max(int(active_sec / target_sec_per_bar), 100)
    threshold = float(tick_notional.sum() / target_bars)

    # Cut bars when cumulative proxy notional reaches threshold
    cs = np.cumsum(tick_notional)
    bar_ids = np.floor(cs / threshold).astype(np.int64)
    # Find bar start/end indices
    change = np.flatnonzero(np.diff(bar_ids, prepend=-1) != 0)
    starts = change
    ends = np.r_[change[1:], len(mid)]

    n_bars = len(starts)
    ts_end = t_msc[ends - 1]
    ts_prev = np.r_[t_msc[0], ts_end[:-1]]
    dur_sec = np.clip((ts_end - ts_prev) / 1000.0, 0.05, 3600.0)

    # Vectorized OHLC & flow aggregation via reduceat
    o_arr = mid[starts]
    c_arr = mid[ends - 1]
    hi_arr = np.maximum.reduceat(mid, starts)
    lo_arr = np.minimum.reduceat(mid, starts)
    cnt_arr = (ends - starts).astype(np.float64)
    notional_arr = np.add.reduceat(tick_notional, starts)
    signed_arr = np.add.reduceat(tick_notional * sign, starts)
    buy_arr = np.add.reduceat(np.where(sign > 0, tick_notional, 0.0), starts)

    df = pd.DataFrame({
        'open': o_arr,
        'high': hi_arr,
        'low': lo_arr,
        'close': c_arr,
        'bid': bid[ends - 1],
        'ask': ask[ends - 1],
        'volume': cnt_arr,
        'notional_usd': notional_arr,
        'signed_notional_usd': signed_arr,
        'buy_notional_usd': buy_arr,
        'trade_count': cnt_arr,
        'duration_sec': dur_sec,
    }, index=pd.to_datetime(ts_end, unit='ms', utc=True))
    df.index.name = 'timestamp'
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDc')
    parser.add_argument('--days', type=int, default=5)
    parser.add_argument('--warmup-days', type=int, default=2)
    args = parser.parse_args()

    t0 = time.time()
    print(f'Fetching {args.days}d of MT5 ticks for {args.symbol}...', flush=True)
    ticks, specs = fetch_mt5_ticks(args.symbol, days=args.days)
    print(f'Fetched {len(ticks)} ticks in {time.time() - t0:.1f}s. Building proxy Dollar Bars...', flush=True)

    df_bars = ticks_to_proxy_dollar_bars(ticks, point=specs['point'], target_sec_per_bar=4.34)
    print(f"Built {len(df_bars)} proxy Dollar Bars | avg duration={df_bars['duration_sec'].mean():.2f}s | avg ticks/bar={df_bars['trade_count'].mean():.1f}", flush=True)

    # Reuse run_inference from backtest_mt5
    import backtest_mt5 as bm
    pred_df, probs_all = bm.run_inference(df_bars, f'{args.symbol}_PROXY_DB', warmup_days=args.warmup_days)

    bh_ret = (pred_df['close'].iloc[-1] / pred_df['close'].iloc[0] - 1.0) * 100.0
    print(f'\n=== {args.symbol} PROXY DOLLAR-BAR BACKTEST ({pred_df["timestamp"].iloc[0]} -> {pred_df["timestamp"].iloc[-1]}) ===')
    print(f'Samples: {len(pred_df)} | Buy & Hold Return: {bh_ret:.2f}%')
    print(f'Prob mean (p_short / p_wait / p_long): {probs_all.mean(0).round(4).tolist()}')
    print(f'Prob max: short={probs_all[:,0].max():.4f}, wait={probs_all[:,1].max():.4f}, long={probs_all[:,2].max():.4f}')

    results = bt.run_realistic_backtest(
        pred_df,
        horizon_bars=96,
        spread_usd=specs['spread_usd'],
        ping_min_ms=30.0,
        ping_max_ms=80.0,
        ping_seed=42,
        slippage_bps=0.0,
        commission_per_lot=0.0,
        leverage=specs['leverage'],
        initial_balance_usc=specs['initial_balance_usc'],
        margin_call_pct=specs['margin_call_pct'],
        stop_out_pct=specs['stop_out_pct'],
        contract_size=specs['contract_size'],
        volume_min=specs['volume_min'],
        volume_step=specs['volume_step'],
        volume_max=specs['volume_max'],
        label=f'{args.symbol.upper()}_TICKDB',
    )

    out_path = Path(f"models/v8_btc_dollar/{args.symbol.lower()}_mt5_tickdb_backtest.json")
    out_json = {
        'cohort': f'mt5_{args.symbol.lower()}_tick_proxy_dollar_bars',
        'start': str(pred_df['timestamp'].iloc[0]),
        'end': str(pred_df['timestamp'].iloc[-1]),
        'ticks': int(len(ticks)),
        'proxy_bars': int(len(df_bars)),
        'samples': len(pred_df),
        'buy_and_hold_return_pct': bh_ret,
        'broker_specs': specs,
        'backtests': results,
    }
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f'\nSaved tick proxy Dollar-Bar backtest to {out_path} (total {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
