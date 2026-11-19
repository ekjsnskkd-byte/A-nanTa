#!/usr/bin/env python3
"""Multi-Head Quantitative Strategy Backtest on 2,000 USC Cent Account (1:2000 leverage).
Uses ALL model heads: decision, action_quantiles, direction, quantiles, barrier,
rv, excursions (MFE/MAE), jump, pattern_detail, regime, path_quantiles."""
import numpy as np
import pandas as pd

df = pd.read_parquet('models/v8_btc_dollar/sealed_oos_full_preds.parquet'
                     if __import__('pathlib').Path('models/v8_btc_dollar/sealed_oos_full_preds.parquet').exists()
                     else 'models/v8_btc_dollar/sealed_oos_preds.parquet')

o = df['open'].to_numpy(dtype=np.float64)
c = df['close'].to_numpy(dtype=np.float64)
hi = df['high'].to_numpy(dtype=np.float64)
lo = df['low'].to_numpy(dtype=np.float64)
lv = df['local_vol'].to_numpy(dtype=np.float64)
dur = df['duration_sec'].to_numpy(dtype=np.float64)

ps = df['p_short'].to_numpy(dtype=np.float64)
pl = df['p_long'].to_numpy(dtype=np.float64)
us = df['u_short'].to_numpy(dtype=np.float64)
ul = df['u_long'].to_numpy(dtype=np.float64)

# Extra heads
d24_up = df['dir24_up'].to_numpy(dtype=np.float64)
d24_dn = df['dir24_dn'].to_numpy(dtype=np.float64)
d96_up = df['dir96_up'].to_numpy(dtype=np.float64)
d96_dn = df['dir96_dn'].to_numpy(dtype=np.float64)
r96_q50 = df['ret96_q50'].to_numpy(dtype=np.float64)
bar_l = df['bar96_long'].to_numpy(dtype=np.float64)
bar_s = df['bar96_short'].to_numpy(dtype=np.float64)
rv96 = np.sqrt(np.maximum(df['rv96_pred'].to_numpy(dtype=np.float64), 1e-10))
mfe96 = np.clip(df['mfe96_pred'].to_numpy(dtype=np.float64), 5e-4, 0.05)
mae96 = np.clip(np.abs(df['mae96_pred'].to_numpy(dtype=np.float64)), 5e-4, 0.05)
jump96 = df['jump96_prob'].to_numpy(dtype=np.float64)
fb96 = df['fb96_prob'].to_numpy(dtype=np.float64)
cont96 = df['cont96_prob'].to_numpy(dtype=np.float64)
reg_trend = df['reg_trend'].to_numpy(dtype=np.float64)
reg_trans = df['reg_transition'].to_numpy(dtype=np.float64)
path_q50 = df['path96_q50'].to_numpy(dtype=np.float64)

PING_S = 0.060
HALF_SPREAD = 0.40
SLIP_FRAC = 0.3e-4
MAX_LEV = 2000.0
STOP_OUT_MARGIN_LEVEL = 0.20
START_BAL = 2000.0


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


def print_head_summary():
    print('=== EXTRA HEADS DIAGNOSTICS (Sealed OOS) ===')
    for name, arr in [
        ('rv96_vol_bps', rv96 * 1e4),
        ('mfe96_bps', mfe96 * 1e4),
        ('mae96_bps', mae96 * 1e4),
        ('jump96_prob', jump96),
        ('fb96_prob', fb96),
        ('cont96_prob', cont96),
        ('reg_trend', reg_trend),
        ('reg_transition', reg_trans),
        ('bar96_long', bar_l),
        ('bar96_short', bar_s),
        ('ret96_q50_bps', r96_q50 * 1e4),
        ('path96_q50_bps', path_q50 * 1e4),
    ]:
        print(f'  {name:16s}: mean={arr.mean():.4f} | p10={np.quantile(arr,0.1):.4f} | p50={np.quantile(arr,0.5):.4f} | p90={np.quantile(arr,0.9):.4f}')


def run_multihead(
    name,
    thresh=0.40,
    margin=0.05,
    use_consensus=False,
    use_risk_filters=False,
    use_dynamic_tp_sl=False,
    risk_pct=0.05,
    hb=96,
):
    bal = START_BAL
    peak = START_BAL
    max_dd = 0.0
    trades = []
    l_cnt = s_cnt = l_win = s_win = stop_outs = 0
    i = 0
    n_rep = len(c)

    while i + hb + 5 < n_rep and bal > 1.0:
        go_l = pl[i] >= thresh and (pl[i] - ps[i]) >= margin and ul[i] >= us[i]
        go_s = ps[i] >= thresh and (ps[i] - pl[i]) >= margin and us[i] >= ul[i]
        if not (go_l or go_s):
            i += 1
            continue

        # Multi-head consensus filter: barrier + direction + path/quantile agreement
        if use_consensus:
            if go_l:
                if not (bar_l[i] > bar_s[i] and d96_up[i] >= d96_dn[i] and path_q50[i] >= 0 and r96_q50[i] >= 0):
                    i += 1
                    continue
            else:
                if not (bar_s[i] > bar_l[i] and d96_dn[i] >= d96_up[i] and path_q50[i] <= 0 and r96_q50[i] <= 0):
                    i += 1
                    continue

        # Regime / Jump / False-breakout filter (avoid bottom/top 15-20% riskiest bars)
        if use_risk_filters:
            if jump96[i] > jump96_cut or fb96[i] > fb96_cut or reg_trans[i] > trans_cut:
                i += 1
                continue

        # Dynamic TP/SL from model's predicted MFE/MAE & RV vs baseline local_vol
        if use_dynamic_tp_sl:
            # Use model's predicted excursion & RV; enforce minimum 25bps edge
            tp_r = float(np.clip(max(mfe96[i], rv96[i] * 1.1), 0.0025, 0.04))
            sl_r = float(np.clip(max(mae96[i], rv96[i] * 0.9), 0.0020, 0.03))
        else:
            tp_r = max(1.0 * lv[i] * np.sqrt(hb), 5e-4)
            if tp_r < 0.0025:
                i += 1
                continue
            sl_r = tp_r

        ent_k, entry_p = exec_entry(i, go_l)
        sl_dist = 1.0 - np.exp(-sl_r) if go_l else np.exp(sl_r) - 1.0
        lev = min(risk_pct / max(sl_dist, 1e-4), MAX_LEV)

        used_margin = bal * (lev / MAX_LEV)
        stop_out_loss_frac = (bal - STOP_OUT_MARGIN_LEVEL * used_margin) / bal
        max_adverse_move = stop_out_loss_frac / lev

        tp = entry_p * np.exp(tp_r) if go_l else entry_p * np.exp(-tp_r)
        sl = entry_p * np.exp(-sl_r) if go_l else entry_p * np.exp(sl_r)
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
            trades.append(-1.0)
            stop_outs += 1
        else:
            if hit_type == 'timeout':
                _, exit_p = exec_entry(min(ent_k + k_exit - 1, n_rep - 2), not go_l)
            raw_ret = ((exit_p / entry_p) - 1.0) if go_l else (1.0 - (exit_p / entry_p))
            pnl_frac = lev * raw_ret
            bal = max(0.0, bal * (1.0 + pnl_frac))
            trades.append(pnl_frac)
            if go_l:
                l_cnt += 1
                if raw_ret > 0: l_win += 1
            else:
                s_cnt += 1
                if raw_ret > 0: s_win += 1
        peak = max(peak, bal)
        max_dd = min(max_dd, (bal - peak) / peak)
        i = ent_k + k_exit

    arr = np.asarray(trades, dtype=np.float64)
    win = float((arr > 0).mean()) if len(arr) else 0.0
    pf = float(arr[arr > 0].sum() / max(abs(arr[arr < 0].sum()), 1e-12)) if len(arr) else 0.0
    sh = float(arr.mean() / max(arr.std(), 1e-12)) if len(arr) > 1 else 0.0
    print(f'{name:45s} | N={len(arr):3d} (L={l_cnt}/{l_win/max(1,l_cnt):.2f}, S={s_cnt}/{s_win/max(1,s_cnt):.2f}) | Win={win:.1%} | PF={pf:.2f} | SR={sh:.2f} | Final={bal:8.2f} USC ({(bal/START_BAL-1)*100:+6.2f}%) | DD={max_dd*100:6.2f}%')


print_head_summary()
jump96_cut = float(np.quantile(jump96, 0.85))
fb96_cut = float(np.quantile(fb96, 0.85))
trans_cut = float(np.quantile(reg_trans, 0.85))

print('\n=== MULTI-HEAD ABLATION (Cent 2,000 USC | lv 1:2000 | Risk 5%/trade | 0 Fee, $0.80 Spread, 60ms Ping) ===')
run_multihead('1. Baseline (Decision only, T=0.40)', thresh=0.40, use_consensus=False, use_risk_filters=False, use_dynamic_tp_sl=False, risk_pct=0.05)
run_multihead('2. +Multi-Head Consensus (T=0.40)', thresh=0.40, use_consensus=True, use_risk_filters=False, use_dynamic_tp_sl=False, risk_pct=0.05)
run_multihead('3. +Consensus +RiskFilters (T=0.40)', thresh=0.40, use_consensus=True, use_risk_filters=True, use_dynamic_tp_sl=False, risk_pct=0.05)
run_multihead('4. +ALL HEADS (+Dynamic MFE/MAE TP/SL) T=0.40', thresh=0.40, use_consensus=True, use_risk_filters=True, use_dynamic_tp_sl=True, risk_pct=0.05)
run_multihead('5. +ALL HEADS (+Dynamic MFE/MAE TP/SL) T=0.35', thresh=0.35, use_consensus=True, use_risk_filters=True, use_dynamic_tp_sl=True, risk_pct=0.05)
run_multihead('6. +ALL HEADS (+Dynamic MFE/MAE TP/SL) T=0.45', thresh=0.45, use_consensus=True, use_risk_filters=True, use_dynamic_tp_sl=True, risk_pct=0.05)
run_multihead('7. +ALL HEADS T=0.40 Risk=10%/trade', thresh=0.40, use_consensus=True, use_risk_filters=True, use_dynamic_tp_sl=True, risk_pct=0.10)
run_multihead('8. +ALL HEADS T=0.35 Risk=10%/trade', thresh=0.35, use_consensus=True, use_risk_filters=True, use_dynamic_tp_sl=True, risk_pct=0.10)
