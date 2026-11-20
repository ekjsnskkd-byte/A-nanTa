#!/usr/bin/env python3
"""Fetch M1 history + live broker specs for any MT5 symbol and run V8 realistic backtest."""
import argparse
import json
import sys
import time
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v8_btc_dollar as bt


MT5_PATH = r'C:\Program Files\MetaTrader 5\terminal64.exe'


def fetch_mt5_symbol(symbol, bars=100000):
    if not mt5.initialize(path=MT5_PATH):
        raise RuntimeError(f'MT5 initialize failed: {mt5.last_error()}')
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f'symbol_select({symbol}) failed: {mt5.last_error()}')
        si = mt5.symbol_info(symbol)
        ai = mt5.account_info()
        tick = mt5.symbol_info_tick(symbol)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f'copy_rates_from_pos({symbol}) returned empty: {mt5.last_error()}')

        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.sort_values('timestamp').set_index('timestamp')
        df['volume'] = pd.to_numeric(df['tick_volume'], errors='coerce')
        df['duration_sec'] = 60.0
        half_sp = (df['spread'].to_numpy(dtype=np.float64) * si.point) * 0.5
        df['bid'] = df['close'] - half_sp
        df['ask'] = df['close'] + half_sp

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
            'digits': int(si.digits),
        }
        return df, specs
    finally:
        mt5.shutdown()


def run_inference(df_all, symbol, ckpt_path='models/v8_btc_dollar/ananta_v8_btc_dollar.pt', warmup_days=14):
    t0 = time.time()
    torch.set_num_threads(4)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mc = ckpt['model_config']
    temp = float(ckpt['calibration_temperature'])
    med = pd.Series(ckpt['med'])
    iqr = pd.Series(ckpt['iqr'])
    mtf_scalers = {k: (pd.Series(v['med']), pd.Series(v['iqr'])) for k, v in ckpt['mtf_scalers'].items()}

    model = bt.UniversalQuantModel(
        features=mc['features'], hidden=mc['hidden'], nexperts=mc['nexperts'],
        topk=mc['topk'], nh=mc['nh'], transformer_layers=mc['layers'],
        attention_heads=mc['heads'], gru_layers=mc['gru_layers'],
        expert_width=mc['expert_width'], ret_scale=float(ckpt['ret_scale']),
        rv_scale=float(ckpt['rv_scale']),
    )
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()

    eval_start = df_all.index.min() + pd.DateOffset(days=warmup_days)
    print(f'MT5 {symbol} M1 bars: {len(df_all)} ({df_all.index.min()} -> {df_all.index.max()})', flush=True)
    print(f'Eval start (after {warmup_days}d warmup): {eval_start}', flush=True)

    f = bt.causal_features(df_all)
    z = bt.causal_feature_normalize(f, med, iqr)
    dates = df_all.index.to_numpy(dtype='datetime64[ns]')

    mtf_data = {}
    for tf_name, tf_freq in bt.CFG['mtf_timeframes'].items():
        tf = bt.build_closed_timeframe_features(df_all, tf_freq)
        tf_med, tf_iqr = mtf_scalers[tf_name]
        tf_z = bt.causal_feature_normalize(
            tf, tf_med, tf_iqr,
            gap_seconds=max(float(bt.CFG['max_gap_seconds']), pd.Timedelta(tf_freq).total_seconds() * 2.0),
        )
        tf_valid = np.isfinite(tf[['ret1', 'vol12', 'vol48']].to_numpy()).all(1).astype('bool')
        tf_times = tf.index.to_numpy('datetime64[ns]')
        stops = np.searchsorted(tf_times, dates, side='right').astype('int32')
        mtf_data[tf_name] = (tf_z, tf_valid, stops)

    lb = bt.CFG['lookback']
    w = bt.CFG['mtf_window']
    eval_np = eval_start.tz_localize(None).to_datetime64()
    core_ok = np.isfinite(f[['ret1', 'vol12', 'vol48']].to_numpy()).all(1)
    run_bad = np.convolve((~core_ok).astype('int32'), np.ones(lb, dtype='int32'), mode='full')[:len(dates)]
    valid_idx = np.where((dates >= eval_np) & (np.arange(len(dates)) >= lb - 1) & (run_bad == 0))[0]
    print(f'Inference samples: {len(valid_idx)} (prep {time.time() - t0:.1f}s)', flush=True)

    batch_size = 512
    probs_list, uq_list = [], []
    t_inf = time.time()
    with torch.no_grad():
        for start_i in range(0, len(valid_idx), batch_size):
            batch_ids = valid_idx[start_i:start_i + batch_size]
            B = len(batch_ids)
            x_b = np.stack([z[idx - lb + 1:idx + 1] for idx in batch_ids], axis=0)
            mtf_b = np.zeros((B, len(bt.TF_NAMES), w, mc['features']), dtype=np.float32)
            mtf_m = np.zeros((B, len(bt.TF_NAMES), w), dtype=bool)
            for j, tf_name in enumerate(bt.TF_NAMES):
                tf_z, tf_valid, stops = mtf_data[tf_name]
                for bi, idx in enumerate(batch_ids):
                    stop = int(stops[idx])
                    s_j = max(0, stop - w)
                    ln = stop - s_j
                    if ln > 0:
                        mtf_b[bi, j, -ln:] = tf_z[s_j:stop]
                        mtf_m[bi, j, -ln:] = tf_valid[s_j:stop]
            out = model(torch.from_numpy(x_b), mtf=torch.from_numpy(mtf_b), mtf_mask=torch.from_numpy(mtf_m))
            probs_list.append(F.softmax(out['decision'].float() / temp, dim=-1).numpy())
            uq_list.append(out['action_quantiles'].float().numpy()[..., 1])
            if (start_i // batch_size) % 30 == 0:
                print(f'  Inference {start_i + B}/{len(valid_idx)} ({time.time() - t_inf:.1f}s)', flush=True)

    probs_all = np.concatenate(probs_list, axis=0)
    uq_all = np.concatenate(uq_list, axis=0)
    sel_df = df_all.iloc[valid_idx]
    lr_all = np.log(df_all['close']).diff()
    vol_all = lr_all.rolling(48, min_periods=24).std().clip(lower=1e-8).iloc[valid_idx].fillna(1e-4)

    pred_df = pd.DataFrame({
        'timestamp': sel_df.index,
        'open': sel_df['open'].to_numpy(dtype=np.float64),
        'close': sel_df['close'].to_numpy(dtype=np.float64),
        'high': sel_df['high'].to_numpy(dtype=np.float64),
        'low': sel_df['low'].to_numpy(dtype=np.float64),
        'duration_sec': sel_df['duration_sec'].to_numpy(dtype=np.float64),
        'local_vol': vol_all.to_numpy(dtype=np.float64),
        'p_short': probs_all[:, 0], 'p_wait': probs_all[:, 1], 'p_long': probs_all[:, 2],
        'u_short': uq_all[:, 0], 'u_wait': uq_all[:, 1], 'u_long': uq_all[:, 2],
    })
    return pred_df, probs_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDc')
    parser.add_argument('--bars', type=int, default=100000)
    parser.add_argument('--warmup-days', type=int, default=14)
    args = parser.parse_args()

    t0 = time.time()
    df_all, specs = fetch_mt5_symbol(args.symbol, bars=args.bars)
    pred_df, probs_all = run_inference(df_all, args.symbol, warmup_days=args.warmup_days)

    bh_ret = (pred_df['close'].iloc[-1] / pred_df['close'].iloc[0] - 1.0) * 100.0
    print(f'\n=== {args.symbol} MT5 BACKTEST SUMMARY ({pred_df["timestamp"].iloc[0]} -> {pred_df["timestamp"].iloc[-1]}) ===')
    print(f'Samples: {len(pred_df)} | Buy & Hold Return: {bh_ret:.2f}%')
    print(f'Prob mean (p_short / p_wait / p_long): {probs_all.mean(0).round(4).tolist()}')
    print(f'Prob max: short={probs_all[:,0].max():.4f}, wait={probs_all[:,1].max():.4f}, long={probs_all[:,2].max():.4f}')
    print(f"Broker specs: Spread=${specs['spread_usd']:.4f}, Ping=30-80ms (mostly 30-40ms), LV=1:{specs['leverage']}, MinLot={specs['volume_min']}, Bal={specs['initial_balance_usc']:.2f} USC")

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
        label=f'{args.symbol.upper()}_MT5',
    )

    out_path = Path(f"models/v8_btc_dollar/{args.symbol.lower()}_mt5_realistic_backtest.json")
    out_json = {
        'cohort': f'mt5_{args.symbol.lower()}_m1_realistic',
        'start': str(pred_df['timestamp'].iloc[0]),
        'end': str(pred_df['timestamp'].iloc[-1]),
        'samples': len(pred_df),
        'buy_and_hold_return_pct': bh_ret,
        'broker_specs': specs,
        'backtests': results,
    }
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f'\nSaved {args.symbol} backtest results to {out_path} (total {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
