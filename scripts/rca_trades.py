#!/usr/bin/env python3
"""Deep Trade Root-Cause Analysis (RCA) tool for V8 Triple-Barrier + Policy backtests."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_v8_btc_dollar as bt


def _agg_group(df_g):
    n = len(df_g)
    if n == 0:
        return {'N': 0}
    wins = df_g['net_usc'] > 0
    losses = df_g['net_usc'] < 0
    gp = float(df_g.loc[wins, 'net_usc'].sum())
    gl = float(abs(df_g.loc[losses, 'net_usc'].sum()))
    pf = gp / max(gl, 1e-12) if gl > 0 else (99.99 if gp > 0 else 0.0)
    return {
        'N': int(n),
        'Win%': round(float(wins.mean() * 100.0), 1),
        'PF': round(pf, 2),
        'Net_USC': round(float(df_g['net_usc'].sum()), 2),
        'Avg_USC': round(float(df_g['net_usc'].mean()), 2),
        'Avg_Hold_m': round(float(df_g['hold_min'].mean()), 1),
        'Avg_MFE_bp': round(float(df_g['mfe_bps'].mean()), 1),
        'Avg_MAE_bp': round(float(df_g['mae_bps'].mean()), 1),
    }


def _classify_loss_cause(row):
    if row['net_usc'] >= 0:
        return 'WIN_OR_BE'
    if row['exit_reason'] == 'soft_cut_hv' or row['reg_high_vol'] >= 0.5 or row['jump96_prob'] >= 0.45:
        return 'HIGH_VOL_SPIKE'
    if row['mfe_bps'] >= 0.25 * row['up_r_bps']:
        return 'REVERSAL_AFTER_PROFIT'
    if row['mfe_bps'] < 5.0 and row['hold_bars'] < 48:
        return 'INSTANT_WRONG_DIR'
    if row['exit_reason'] == 'stale_cut' or row['hold_bars'] >= 48:
        return 'STALE_CHOP_BLEED'
    return 'ADVERSE_DRIFT_TO_SL'


def build_trade_df(pred_df, trade_log, spread_usd):
    ts = pred_df['timestamp'].to_numpy()
    hi = pred_df['high'].to_numpy(dtype=np.float64)
    lo = pred_df['low'].to_numpy(dtype=np.float64)
    half_sp = spread_usd * 0.5

    rows = []
    for tid, tr in enumerate(trade_log, 1):
        si = tr['sig_idx']
        ei = tr['ent_idx']
        xi = tr['exit_idx']
        is_long = tr['side'] == 'LONG'
        t_ent = pd.Timestamp(ts[ei])
        t_exit = pd.Timestamp(ts[xi])
        hold_min = max((t_exit - t_ent).total_seconds() / 60.0, 0.0)

        sl_hi = hi[ei:xi + 1]
        sl_lo = lo[ei:xi + 1]
        if is_long:
            fav_max = float((sl_hi - half_sp).max())
            adv_min = float((sl_lo - half_sp).min())
            mfe_bps = np.log(max(fav_max, 1e-9) / tr['entry_p']) * 1e4
            mae_bps = np.log(max(adv_min, 1e-9) / tr['entry_p']) * 1e4
        else:
            fav_min = float((sl_lo + half_sp).min())
            adv_max = float((sl_hi + half_sp).max())
            mfe_bps = np.log(tr['entry_p'] / max(fav_min, 1e-9)) * 1e4
            mae_bps = np.log(tr['entry_p'] / max(adv_max, 1e-9)) * 1e4

        p_short = float(pred_df['p_short'].iloc[si])
        p_long = float(pred_df['p_long'].iloc[si])
        p_wait = float(pred_df['p_wait'].iloc[si]) if 'p_wait' in pred_df else 0.0
        p_dir = p_long if is_long else p_short
        p_opp = p_short if is_long else p_long

        row = {
            'trade_id': tid,
            'entry_time': str(t_ent),
            'exit_time': str(t_exit),
            'month': t_ent.strftime('%Y-%m'),
            'dow': t_ent.strftime('%a'),
            'utc_hour': int(t_ent.hour),
            'session': (
                '00-04_AsiaEarly' if t_ent.hour < 4 else
                '04-08_AsiaLate' if t_ent.hour < 8 else
                '08-12_LondonAM' if t_ent.hour < 12 else
                '12-16_LondonNY' if t_ent.hour < 16 else
                '16-20_NYPM' if t_ent.hour < 20 else
                '20-24_LateNY'
            ),
            'side': tr['side'],
            'exit_reason': tr['exit_reason'],
            'after_sl_bump': tr['after_sl_bump'],
            'hold_bars': tr['hold_bars'],
            'extended_bars': tr['extended_bars'],
            'hold_min': hold_min,
            'lots': tr['lots'],
            'entry_p': tr['entry_p'],
            'exit_p': tr['exit_p'],
            'net_usc': tr['net_usc'],
            'acct_ret_pct': tr['acct_ret'] * 100.0,
            'bal_after': tr['bal_after'],
            'dd_pct': tr['dd_pct'],
            'up_r_bps': tr['up_r'] * 1e4,
            'local_vol_bps': float(pred_df['local_vol'].iloc[si]) * 1e4,
            'mfe_bps': float(mfe_bps),
            'mae_bps': float(mae_bps),
            'p_dir': p_dir,
            'p_opp': p_opp,
            'p_margin': p_dir - p_opp,
            'p_wait': p_wait,
            'u_dir': float(pred_df['u_long'].iloc[si] if is_long else pred_df['u_short'].iloc[si]),
            'reg_trend': float(pred_df['reg_trend'].iloc[si]) if 'reg_trend' in pred_df else 0.0,
            'reg_high_vol': float(pred_df['reg_high_vol'].iloc[si]) if 'reg_high_vol' in pred_df else 0.0,
            'reg_transition': float(pred_df['reg_transition'].iloc[si]) if 'reg_transition' in pred_df else 0.0,
            'jump96_prob': float(pred_df['jump96_prob'].iloc[si]) if 'jump96_prob' in pred_df else 0.0,
            'ret24_q50': float(pred_df['ret24_q50'].iloc[si]) if 'ret24_q50' in pred_df else 0.0,
            'ret96_q50': float(pred_df['ret96_q50'].iloc[si]) if 'ret96_q50' in pred_df else 0.0,
            'mfe96_pred': float(pred_df['mfe96_pred'].iloc[si]) if 'mfe96_pred' in pred_df else 0.0,
            'mae96_pred': float(pred_df['mae96_pred'].iloc[si]) if 'mae96_pred' in pred_df else 0.0,
            'fb96_prob': float(pred_df['fb96_prob'].iloc[si]) if 'fb96_prob' in pred_df else 0.0,
            'rev96_prob': float(pred_df['rev96_prob'].iloc[si]) if 'rev96_prob' in pred_df else 0.0,
            'cont96_prob': float(pred_df['cont96_prob'].iloc[si]) if 'cont96_prob' in pred_df else 0.0,
            'bar96_wait': float(pred_df['bar96_wait'].iloc[si]) if 'bar96_wait' in pred_df else 0.0,
        }
        row['loss_cause'] = _classify_loss_cause(row)
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(title, df_t, group_col):
    print(f'\n=== {title} ===')
    out = {}
    for k, g in df_t.groupby(group_col, observed=False):
        if len(g) == 0:
            continue
        s = _agg_group(g)
        out[str(k)] = s
        print(f"  {str(k):22s} | N={s['N']:4d} | Win={s['Win%']:5.1f}% | PF={s['PF']:5.2f} | Net={s['Net_USC']:+8.2f}c | Avg={s['Avg_USC']:+6.2f}c | Hold={s['Avg_Hold_m']:5.1f}m | MFE={s['Avg_MFE_bp']:+5.1f}bp | MAE={s['Avg_MAE_bp']:+6.1f}bp")
    return out


def analyze_loss_waterfall(df_t):
    losers = df_t[df_t['net_usc'] < 0]
    gross_loss = float(abs(losers['net_usc'].sum()))
    gross_profit = float(df_t.loc[df_t['net_usc'] > 0, 'net_usc'].sum())
    print(f'\n=== LOSS WATERFALL (Gross Profit=+{gross_profit:.2f}c | Gross Loss=-{gross_loss:.2f}c | Net={gross_profit - gross_loss:+.2f}c) ===')
    wf = {}
    for col_name, title in (('loss_cause', 'By Root Cause'), ('exit_reason', 'By Exit Type')):
        print(f'  -- {title} --')
        sub = {}
        grp = losers.groupby(col_name)['net_usc'].agg(['count', 'sum', 'mean']).sort_values('sum')
        for k, r in grp.iterrows():
            share = abs(float(r['sum'])) / max(gross_loss, 1e-9) * 100.0
            sub[str(k)] = {'N': int(r['count']), 'Loss_USC': round(float(r['sum']), 2), 'Share%': round(share, 1), 'Avg_USC': round(float(r['mean']), 2)}
            print(f"    {str(k):24s} | N={int(r['count']):3d} | Loss={r['sum']:+8.2f}c ({share:5.1f}% of all losses) | Avg={r['mean']:+6.2f}c")
        wf[col_name] = sub
    return wf


def analyze_drawdown_episodes(df_t, top_n=3):
    print(f'\n=== TOP {top_n} DRAWDOWN EPISODES ===')
    episodes = []
    peak_bal = df_t['bal_after'].iloc[0] - df_t['net_usc'].iloc[0]
    peak_idx = 0
    cur_trough_dd = 0.0
    trough_idx = 0
    in_dd = False
    for idx, r in df_t.iterrows():
        if r['bal_after'] >= peak_bal:
            if in_dd and cur_trough_dd < -1.0:
                episodes.append((cur_trough_dd, peak_idx, trough_idx, idx))
            peak_bal = r['bal_after']
            peak_idx = idx
            cur_trough_dd = 0.0
            in_dd = False
        else:
            in_dd = True
            dd_now = (r['bal_after'] - peak_bal) / peak_bal * 100.0
            if dd_now < cur_trough_dd:
                cur_trough_dd = dd_now
                trough_idx = idx
    if in_dd and cur_trough_dd < -1.0:
        episodes.append((cur_trough_dd, peak_idx, trough_idx, len(df_t) - 1))
    episodes.sort(key=lambda x: x[0])

    out = []
    for rank, (dd_val, p_i, t_i, rec_i) in enumerate(episodes[:top_n], 1):
        seg = df_t.iloc[p_i:t_i + 1]
        losers = seg[seg['net_usc'] < 0]
        causes = losers['loss_cause'].value_counts().to_dict()
        sides = losers['side'].value_counts().to_dict()
        ep_info = {
            'rank': rank,
            'max_dd_pct': round(dd_val, 2),
            'start': seg['entry_time'].iloc[0][:19],
            'trough': seg['exit_time'].iloc[-1][:19],
            'trades_in_dd': int(len(seg)),
            'losing_trades': int(len(losers)),
            'net_usc': round(float(seg['net_usc'].sum()), 2),
            'sides': sides,
            'causes': causes,
        }
        out.append(ep_info)
        print(f"  #{rank} DD={dd_val:.2f}% | {ep_info['start']} -> {ep_info['trough']} | N={len(seg)} (Losses={len(losers)}, Net={ep_info['net_usc']:+.2f}c) | Sides={sides} | Causes={causes}")
    return out


def analyze_win_vs_loss_features(df_t):
    print('\n=== WIN vs LOSS FEATURE PROFILE ===')
    w = df_t[df_t['net_usc'] > 0]
    l = df_t[df_t['net_usc'] < 0]
    feats = ['p_dir', 'p_margin', 'p_wait', 'up_r_bps', 'local_vol_bps', 'hold_min', 'reg_trend', 'reg_high_vol', 'reg_transition', 'jump96_prob']
    res = {}
    for f in feats:
        wm, lm = float(w[f].mean()), float(l[f].mean())
        sd = float(df_t[f].std())
        d = (wm - lm) / max(sd, 1e-9)
        res[f] = {'win_mean': round(wm, 4), 'loss_mean': round(lm, 4), 'cohens_d': round(d, 3)}
        print(f"  {f:16s} | Win={wm:8.4f} | Loss={lm:8.4f} | Diff={wm - lm:+8.4f} | d={d:+.3f}")
    return res


def analyze_what_if(df_t):
    print('\n=== WHAT-IF FILTER IMPACT (Loss-Cutting Simulations) ===')
    base = _agg_group(df_t)
    scenarios = {
        'Exclude_HighVol_Regime': df_t[df_t['reg_high_vol'] < 0.5],
        'Exclude_p_dir_0.41_0.44': df_t[~((df_t['p_dir'] >= 0.41) & (df_t['p_dir'] < 0.44))],
        'Exclude_AfterSL_Bump': df_t[~df_t['after_sl_bump']],
        'SHORT_Only': df_t[df_t['side'] == 'SHORT'],
        'CalmRange_Regime_Only': df_t[(df_t['reg_high_vol'] < 0.5) & (df_t['reg_trend'] < 0.5) & (df_t['reg_transition'] < 0.5)],
    }
    out = {'baseline': base}
    print(f"  {'Baseline':26s} | N={base['N']:4d} | Win={base['Win%']:5.1f}% | PF={base['PF']:5.2f} | Net={base['Net_USC']:+8.2f}c")
    for name, sub in scenarios.items():
        s = _agg_group(sub)
        out[name] = s
        print(f"  {name:26s} | N={s['N']:4d} | Win={s['Win%']:5.1f}% | PF={s['PF']:5.2f} | Net={s['Net_USC']:+8.2f}c (Delta={s['Net_USC'] - base['Net_USC']:+7.2f}c)")
    return out


def scan_blacklist_patterns(df_t, min_n=15, max_pf=0.80, top_k=15):
    print(f'\n=== FINE-GRAINED BLACKLIST PATTERN SCANNER (N>={min_n}, PF<={max_pf:.2f}) ===')
    num_cols = ['p_dir', 'p_margin', 'p_wait', 'up_r_bps', 'reg_trend', 'reg_high_vol', 'reg_transition',
                'jump96_prob', 'ret24_q50', 'ret96_q50', 'mfe96_pred', 'mae96_pred', 'fb96_prob', 'rev96_prob', 'cont96_prob', 'bar96_wait']
    cat_cols = ['side', 'session', 'dow']
    bins = {}
    for c in cat_cols:
        bins[c] = df_t[c].astype(str)
    for c in num_cols:
        if c not in df_t or df_t[c].nunique() < 4:
            continue
        try:
            q = pd.qcut(df_t[c], q=4, duplicates='drop')
            bins[c] = q.astype(str)
        except ValueError:
            continue

    found = []
    keys = list(bins.keys())
    # 1D scan
    for k in keys:
        for val, g in df_t.groupby(bins[k]):
            if len(g) < min_n:
                continue
            s = _agg_group(g)
            if s['PF'] <= max_pf and s['Net_USC'] < 0:
                found.append({'pattern': f'{k} == {val}', **s})
    # 2D interaction scan
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            comb = bins[k1] + ' & ' + bins[k2]
            for val, g in df_t.groupby(comb):
                if len(g) < min_n:
                    continue
                s = _agg_group(g)
                if s['PF'] <= max_pf and s['Net_USC'] < 0:
                    p1, p2 = val.split(' & ', 1)
                    found.append({'pattern': f'{k1}={p1} AND {k2}={p2}', **s})
    found.sort(key=lambda x: x['Net_USC'])
    for r in found[:top_k]:
        print(f"  N={r['N']:3d} | WR={r['Win%']:5.1f}% | PF={r['PF']:4.2f} | Net={r['Net_USC']:+7.2f}c | Avg={r['Avg_USC']:+6.2f}c | {r['pattern']}")
    return found[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, default='models/v8_btc_dollar/sealed_oos_preds.parquet')
    parser.add_argument('--thresh', type=float, default=0.35)
    parser.add_argument('--min-edge-bps', type=float, default=25.0)
    parser.add_argument('--risk-pct', type=float, default=1.0)
    parser.add_argument('--spread-usd', type=float, default=bt.BROKER_CFG['spread_usd'])
    parser.add_argument('--out-prefix', type=str, default='models/v8_btc_dollar/rca')
    args = parser.parse_args()

    pred_df = pd.read_parquet(args.preds)
    trade_log = []
    bt.run_realistic_backtest(
        pred_df,
        horizon_bars=96,
        spread_usd=args.spread_usd,
        risk_pct=args.risk_pct,
        min_edge_bps=args.min_edge_bps,
        base_thresh=args.thresh,
        sweep=False,
        trade_log_out=trade_log,
    )
    if not trade_log:
        print('No trades executed.')
        return

    df_t = build_trade_df(pred_df, trade_log, args.spread_usd)
    df_t['up_r_bucket'] = pd.cut(df_t['up_r_bps'], bins=[0, 35, 50, 75, 120, 9999], labels=['25-35bp', '35-50bp', '50-75bp', '75-120bp', '>120bp'])
    df_t['p_dir_bucket'] = pd.cut(df_t['p_dir'], bins=[0.30, 0.38, 0.41, 0.44, 1.0], labels=['0.35-0.38', '0.38-0.41', '0.41-0.44', '>=0.44'])
    df_t['regime_dom'] = np.where(
        df_t['reg_high_vol'] >= 0.5, 'HighVol',
        np.where(df_t['reg_trend'] >= 0.5, 'Trend', np.where(df_t['reg_transition'] >= 0.5, 'Transition', 'Calm/Range'))
    )

    report = {
        'overall': _agg_group(df_t),
        'loss_waterfall': analyze_loss_waterfall(df_t),
        'drawdown_episodes': analyze_drawdown_episodes(df_t, top_n=3),
        'win_vs_loss_features': analyze_win_vs_loss_features(df_t),
        'what_if': analyze_what_if(df_t),
        'blacklist_candidates': scan_blacklist_patterns(df_t, min_n=15, max_pf=0.80, top_k=15),
        'by_exit_reason': print_table('BY EXIT REASON', df_t, 'exit_reason'),
        'by_loss_cause': print_table('BY ROOT CAUSE', df_t, 'loss_cause'),
        'by_side': print_table('BY SIDE', df_t, 'side'),
        'by_regime': print_table('BY MODEL REGIME', df_t, 'regime_dom'),
        'by_confidence': print_table('BY MODEL CONFIDENCE (p_dir)', df_t, 'p_dir_bucket'),
        'by_barrier_size': print_table('BY BARRIER SIZE (up_r_bps)', df_t, 'up_r_bucket'),
        'by_session': print_table('BY UTC SESSION (4h)', df_t, 'session'),
        'by_utc_hour': print_table('BY UTC HOUR (00-23)', df_t, 'utc_hour'),
        'by_dow': print_table('BY DAY OF WEEK', df_t, 'dow'),
        'by_month': print_table('BY MONTH', df_t, 'month'),
    }

    out_csv = Path(f'{args.out_prefix}_trades.csv')
    out_json = Path(f'{args.out_prefix}_report.json')
    df_t.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(report, indent=2))
    print(f'\nSaved detailed trade log to {out_csv} and RCA report to {out_json}')


if __name__ == '__main__':
    main()
