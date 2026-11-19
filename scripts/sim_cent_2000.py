#!/usr/bin/env python3
"""Simulate Cent account (2,000 USC capital, 1:2000 max leverage, Stop-Out @ 20% margin level)
using models/v8_btc_dollar/sealed_oos_preds.parquet with realistic execution (0 fee, $0.80 spread, 60ms ping, 0.3bps slip)."""
import numpy as np
import pandas as pd

df = pd.read_parquet('models/v8_btc_dollar/sealed_oos_preds.parquet')
o = df['open'].to_numpy(dtype=np.float64)
c = df['close'].to_numpy(dtype=np.float64)
hi = df['high'].to_numpy(dtype=np.float64)
lo = df['low'].to_numpy(dtype=np.float64)
lv = df['local_vol'].to_numpy(dtype=np.float64)
dur = df['duration_sec'].to_numpy(dtype=np.float64)
ps = df['p_short'].to_numpy(dtype=np.float64)
pl = df['p_long'].to_numpy(dtype=np.float64)
ul = df['u_long'].to_numpy(dtype=np.float64)
us = df['u_short'].to_numpy(dtype=np.float64)

PING_S = 0.060
HALF_SPREAD = 0.40
SLIP_FRAC = 0.3e-4
MARGIN_SIGNAL = 0.05
MAX_LEV = 2000.0
STOP_OUT_MARGIN_LEVEL = 0.20  # 20% margin level stop-out
START_BAL = 2000.0  # 2,000 USC (Cent)


def exec_entry(idx, is_long):
    rem = PING_S
    k = idx + 1
    n_rep = len(c)
    while k < n_rep - 1 and dur[k] < rem:
        rem -= dur[k]
        k += 1
    frac = min(rem / max(dur[k], 1e-3), 1.0)
    mid = o[k] + frac * (c[k] - o[k])
    fill = (mid + HALF_SPREAD + mid * SLIP_FRAC) if is_long else (mid - HALF_SPREAD - mid * SLIP_FRAC)
    return k, fill


def sim_fixed(hb, thresh, eff_lev):
    bal = START_BAL
    peak = START_BAL
    max_dd = 0.0
    trades = 0
    wins = 0
    stop_outs = 0
    i = 0
    n_rep = len(c)
    while i + hb + 5 < n_rep and bal > 1.0:
        go_l = pl[i] >= thresh and (pl[i] - ps[i]) >= MARGIN_SIGNAL and ul[i] >= us[i]
        go_s = ps[i] >= thresh and (ps[i] - pl[i]) >= MARGIN_SIGNAL and us[i] >= ul[i]
        if not (go_l or go_s):
            i += 1
            continue
        ent_k, entry_p = exec_entry(i, go_l)
        used_margin = bal * (eff_lev / MAX_LEV)
        stop_out_loss_frac = (bal - STOP_OUT_MARGIN_LEVEL * used_margin) / bal
        max_adverse_move = stop_out_loss_frac / eff_lev

        exit_k = min(ent_k + hb, n_rep - 1)
        liquidated = False
        for k in range(1, hb + 1):
            idx_k = ent_k + k
            if idx_k >= n_rep:
                break
            adv = ((entry_p - lo[idx_k]) / entry_p) if go_l else ((hi[idx_k] - entry_p) / entry_p)
            if adv >= max_adverse_move:
                liquidated = True
                exit_k = idx_k
                break

        if liquidated:
            bal = max(0.0, STOP_OUT_MARGIN_LEVEL * used_margin)
            stop_outs += 1
        else:
            _, exit_p = exec_entry(exit_k - 1, not go_l)
            raw_ret = ((exit_p / entry_p) - 1.0) if go_l else (1.0 - (exit_p / entry_p))
            pnl = bal * eff_lev * raw_ret
            bal = max(0.0, bal + pnl)
            if raw_ret > 0:
                wins += 1
        trades += 1
        peak = max(peak, bal)
        max_dd = min(max_dd, (bal - peak) / peak)
        i = exit_k
    return bal, (bal / START_BAL - 1) * 100, max_dd * 100, trades, wins / max(1, trades), stop_outs


def sim_tb(hb, thresh, min_edge, risk_pct=None, eff_lev=None):
    bal = START_BAL
    peak = START_BAL
    max_dd = 0.0
    trades = 0
    wins = 0
    stop_outs = 0
    i = 0
    n_rep = len(c)
    while i + hb + 5 < n_rep and bal > 1.0:
        up_r = max(1.0 * lv[i] * np.sqrt(hb), 5e-4)
        go_l = up_r >= min_edge and pl[i] >= thresh and (pl[i] - ps[i]) >= MARGIN_SIGNAL and ul[i] >= us[i]
        go_s = up_r >= min_edge and ps[i] >= thresh and (ps[i] - pl[i]) >= MARGIN_SIGNAL and us[i] >= ul[i]
        if not (go_l or go_s):
            i += 1
            continue
        ent_k, entry_p = exec_entry(i, go_l)
        sl_dist = 1.0 - np.exp(-up_r) if go_l else np.exp(up_r) - 1.0
        if risk_pct is not None:
            lev = min(risk_pct / max(sl_dist, 1e-4), MAX_LEV)
        else:
            lev = eff_lev

        used_margin = bal * (lev / MAX_LEV)
        stop_out_loss_frac = (bal - STOP_OUT_MARGIN_LEVEL * used_margin) / bal
        max_adverse_move = stop_out_loss_frac / lev

        tp = entry_p * np.exp(up_r) if go_l else entry_p * np.exp(-up_r)
        sl = entry_p * np.exp(-up_r) if go_l else entry_p * np.exp(up_r)
        k_exit = hb
        exit_p = c[min(ent_k + hb, n_rep - 1)]
        hit_type = 'timeout'
        liquidated = False

        for k in range(1, hb + 1):
            idx_k = ent_k + k
            if idx_k >= n_rep:
                break
            adv = ((entry_p - lo[idx_k]) / entry_p) if go_l else ((hi[idx_k] - entry_p) / entry_p)
            if adv >= max_adverse_move:
                liquidated = True
                k_exit = k
                break
            if go_l:
                if lo[idx_k] <= sl:
                    k_exit = k; exit_p = sl - HALF_SPREAD - sl * SLIP_FRAC; hit_type = 'sl'; break
                if hi[idx_k] >= tp:
                    k_exit = k; exit_p = tp; hit_type = 'tp'; break
            else:
                if hi[idx_k] >= sl:
                    k_exit = k; exit_p = sl + HALF_SPREAD + sl * SLIP_FRAC; hit_type = 'sl'; break
                if lo[idx_k] <= tp:
                    k_exit = k; exit_p = tp; hit_type = 'tp'; break

        if liquidated:
            bal = max(0.0, STOP_OUT_MARGIN_LEVEL * used_margin)
            stop_outs += 1
        else:
            if hit_type == 'timeout':
                _, exit_p = exec_entry(min(ent_k + k_exit - 1, n_rep - 2), not go_l)
            raw_ret = ((exit_p / entry_p) - 1.0) if go_l else (1.0 - (exit_p / entry_p))
            bal = max(0.0, bal + bal * lev * raw_ret)
            if raw_ret > 0:
                wins += 1
        trades += 1
        peak = max(peak, bal)
        max_dd = min(max_dd, (bal - peak) / peak)
        i = ent_k + k_exit
    return bal, (bal / START_BAL - 1) * 100, max_dd * 100, trades, wins / max(1, trades), stop_outs


print('=== CENT ACCOUNT SIMULATION (Start = 2,000 USC | Max Leverage = 1:2000 | Stop-Out = 20% Margin) ===')
print('\n--- 1. Fixed-Horizon (h=24, Thresh=0.35) by Effective Leverage ---')
for lev in (1, 5, 10, 20, 50, 100, 200, 500, 2000):
    bal, ret, dd, n, wr, so = sim_fixed(24, 0.35, lev)
    print(f'EffLev={lev:4d}x | Final={bal:10.2f} USC | Ret={ret:+8.2f}% | MaxDD={dd:7.2f}% | N={n:3d} | Win={wr:.1%} | StopOuts={so}')

print('\n--- 2. Fixed-Horizon (h=96, Thresh=0.35) by Effective Leverage ---')
for lev in (1, 5, 10, 20, 50, 100, 200, 500, 2000):
    bal, ret, dd, n, wr, so = sim_fixed(96, 0.35, lev)
    print(f'EffLev={lev:4d}x | Final={bal:10.2f} USC | Ret={ret:+8.2f}% | MaxDD={dd:7.2f}% | N={n:3d} | Win={wr:.1%} | StopOuts={so}')

print('\n--- 3. Triple-Barrier (h=96, tp>=25bp, Thresh=0.45) by Risk % per Trade (1:2000 margin) ---')
for rp in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35):
    bal, ret, dd, n, wr, so = sim_tb(96, 0.45, 0.0025, risk_pct=rp)
    print(f'Risk={rp*100:4.1f}%/trade | Final={bal:10.2f} USC | Ret={ret:+8.2f}% | MaxDD={dd:7.2f}% | N={n:3d} | Win={wr:.1%} | StopOuts={so}')

print('\n--- 4. Triple-Barrier (h=96, tp>=0bp, Thresh=0.45) by Risk % per Trade (1:2000 margin) ---')
for rp in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35):
    bal, ret, dd, n, wr, so = sim_tb(96, 0.45, 0.0, risk_pct=rp)
    print(f'Risk={rp*100:4.1f}%/trade | Final={bal:10.2f} USC | Ret={ret:+8.2f}% | MaxDD={dd:7.2f}% | N={n:3d} | Win={wr:.1%} | StopOuts={so}')
