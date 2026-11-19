#!/usr/bin/env python3
"""Sealed OOS evaluation and trading backtest for ananta_v8_btc_dollar.pt."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = {
    'horizons': [2, 6, 24, 96],
    'lookback': 128,
    'purge_bars': 256,
    'max_gap_seconds': 3600.0,
    'trailing_feature_window': 200,
    'sealed_months': 3,
    'cal_months': 3,
    'val_months': 6,
    'validation_report_months': 2,
    'path_rank': 4,
    'path_knots': [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96],
    'mtf_timeframes': {'M15': '15min', 'H1': '1h', 'H4': '4h', 'D1': '1D'},
    'mtf_window': 32,
    'mtf_encoder_layers': 1,
    'dropout': 0.25,
    'neutral_move_bps': 5.0,
    'label_vol_multiplier': 0.25,
    'barrier_vol_multiplier': 1.0,
    'reference_cost_bps': 3.0,
    'min_rows': 2000,
}

TF_NAMES = list(CFG['mtf_timeframes'])
FEATURES = [
    'ret1', 'ret3', 'ret12', 'vol12', 'vol48', 'vol_ratio', 'efficiency', 'range_norm',
    'position', 'downside', 'skew24', 'kurt24', 'entropy24',
    'spread_rel', 'spread_present', 'volume_log', 'volume_present',
    'trend_strength_96', 'volatility_har_12', 'volatility_har_48', 'volatility_har_288',
    'hurst_proxy', 'wavelet_band', 'jump_proxy', 'mean_reversion_z', 'autocorr24',
    'taker_dollar_imbalance', 'taker_buy_ratio', 'notional_log', 'trade_count_log',
    'imbalance_ewm', 'flow_shock',
    'log_dt', 'dollar_speed_log', 'dollar_imbalance_per_sec',
    'vpin_24', 'kyle_lambda_24', 'gk_vol_24', 'amihud_24', 'avg_trade_size_log',
]


def load_asset_file(path):
    path = Path(path)
    d = pd.read_parquet(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='raise')
    d = d.sort_values('timestamp').set_index('timestamp')
    for k in ('open', 'high', 'low', 'close', 'bid', 'ask', 'volume', 'notional_usd',
              'signed_notional_usd', 'buy_notional_usd', 'trade_count', 'duration_sec'):
        if k in d:
            d[k] = pd.to_numeric(d[k], errors='coerce')
    return d


def fast_permutation_entropy(log_returns, window=24):
    a = np.asarray(log_returns, dtype=np.float64)
    n = len(a)
    w = window - 2
    out = np.full(n, np.nan, dtype=np.float32)
    if n < window:
        return out
    x = a[:-2]
    y = a[1:-1]
    z = a[2:]
    rank_x = ((y < x).astype('int8') + (z < x).astype('int8'))
    rank_y = ((x <= y).astype('int8') + (z < y).astype('int8'))
    rank_z = 3 - rank_x - rank_y
    patterns = np.full(n, -1, dtype=np.int16)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    patterns[2:][good] = (rank_x[good] * 9 + rank_y[good] * 3 + rank_z[good]).astype(np.int16)
    bad_cs = np.empty(n + 1, dtype=np.int32)
    bad_cs[0] = 0
    np.cumsum(patterns < 0, out=bad_cs[1:])
    valid = (bad_cs[w:] - bad_cs[:-w]) == 0
    ent = np.zeros(n - w + 1, dtype=np.float64)
    cs = np.empty(n + 1, dtype=np.int32)
    cs[0] = 0
    for key in (5, 7, 11, 15, 19, 21):
        np.cumsum(patterns == key, out=cs[1:])
        cnt = cs[w:] - cs[:-w]
        nz = cnt > 0
        pr = cnt[nz] / float(w)
        ent[nz] -= pr * np.log(pr)
    out[w - 1:] = np.where(valid, ent / np.log(6.0), np.nan).astype(np.float32)
    return out


def _causal_features_one_contiguous_segment(d):
    p = d.close
    lr = np.log(p).diff()
    v = lr.rolling(48, min_periods=24).std()
    f = pd.DataFrame(index=d.index)
    f['ret1'] = lr
    f['ret3'] = np.log(p / p.shift(3))
    f['ret12'] = np.log(p / p.shift(12))
    f['vol12'] = lr.rolling(12, min_periods=12).std()
    f['vol48'] = v
    f['vol_ratio'] = f.vol12 / (v + 1e-8)
    f['efficiency'] = p.diff(24).abs() / (p.diff().abs().rolling(24).sum() + 1e-8)
    f['range_norm'] = (d.high - d.low) / (p * v.replace(0, np.nan) + 1e-8)
    f['position'] = (p - p.rolling(24).min()) / (p.rolling(24).max() - p.rolling(24).min() + 1e-8)
    f['downside'] = lr.clip(upper=0).pow(2).rolling(24).mean().pow(0.5)
    f['skew24'] = lr.rolling(24).skew()
    f['kurt24'] = lr.rolling(24).kurt()
    f['entropy24'] = fast_permutation_entropy(lr.to_numpy(), 24)
    if {'bid', 'ask'} <= set(d.columns):
        raw = ((d.ask - d.bid) / p).where((d.ask >= d.bid) & (d.bid > 0))
        f['spread_rel'] = raw
        f['spread_present'] = raw.notna().astype(float)
    else:
        f['spread_rel'] = np.nan
        f['spread_present'] = 0.0
    if 'volume' in d:
        x = np.log1p(d.volume.clip(lower=0))
        f['volume_log'] = x - x.rolling(48, min_periods=24).mean()
        f['volume_present'] = d.volume.notna().astype(float)
    else:
        f['volume_log'] = np.nan
        f['volume_present'] = 0.0
    f['trend_strength_96'] = np.log(p / p.shift(96)) / (lr.rolling(96, min_periods=48).std() * np.sqrt(96) + 1e-9)
    f['volatility_har_12'] = lr.pow(2).rolling(12, min_periods=12).mean()
    f['volatility_har_48'] = lr.pow(2).rolling(48, min_periods=48).mean()
    f['volatility_har_288'] = lr.pow(2).rolling(288, min_periods=144).mean()
    f['hurst_proxy'] = np.log((lr.rolling(48, min_periods=24).sum().abs() + 1e-9) / (lr.abs().rolling(48, min_periods=24).sum() + 1e-9))
    f['wavelet_band'] = lr.rolling(8, min_periods=8).mean() - lr.rolling(48, min_periods=24).mean()
    f['jump_proxy'] = (lr.abs() / (lr.abs().rolling(48, min_periods=24).median() + 1e-9)).clip(upper=50)
    f['mean_reversion_z'] = (np.log(p) - np.log(p).rolling(48, min_periods=24).mean()) / (v * np.sqrt(48) + 1e-8)
    f['autocorr24'] = lr.rolling(24, min_periods=24).corr(lr.shift(1))

    notional = d['notional_usd'] if 'notional_usd' in d else pd.Series(np.nan, index=d.index)
    signed = d['signed_notional_usd'] if 'signed_notional_usd' in d else pd.Series(np.nan, index=d.index)
    buys = d['buy_notional_usd'] if 'buy_notional_usd' in d else pd.Series(np.nan, index=d.index)
    trades = d['trade_count'] if 'trade_count' in d else pd.Series(np.nan, index=d.index)
    imbalance = (signed / (notional + 1e-9)).where(notional > 0)
    f['taker_dollar_imbalance'] = imbalance
    f['taker_buy_ratio'] = (buys / (notional + 1e-9)).where(notional > 0)
    f['notional_log'] = np.log1p(notional.clip(lower=0)) - np.log1p(notional.clip(lower=0)).rolling(48, min_periods=24).mean()
    f['trade_count_log'] = np.log1p(trades.clip(lower=0)) - np.log1p(trades.clip(lower=0)).rolling(48, min_periods=24).mean()
    f['imbalance_ewm'] = imbalance.ewm(span=24, adjust=False, min_periods=12).mean()
    f['flow_shock'] = (signed.abs() / (signed.abs().rolling(48, min_periods=24).median() + 1e-9)).clip(0, 50)

    if 'duration_sec' in d:
        dt = d['duration_sec'].clip(lower=1e-3)
    else:
        dt = pd.Series(d.index.to_series().diff().dt.total_seconds().fillna(1.0), index=d.index).clip(lower=1e-3)
    f['log_dt'] = np.log1p(dt)
    speed = np.log1p((notional.clip(lower=0) / dt).clip(lower=0))
    f['dollar_speed_log'] = speed - speed.rolling(48, min_periods=24).mean()
    f['dollar_imbalance_per_sec'] = np.arcsinh(signed / (dt * 1000.0 + 1e-3))
    f['vpin_24'] = (signed.abs() / (notional + 1e-9)).rolling(24, min_periods=12).mean()
    notional_m = (notional / 1e6).clip(lower=1e-6)
    signed_m = (signed.abs() / 1e6).clip(lower=1e-6)
    f['kyle_lambda_24'] = np.log1p((lr.abs() / signed_m).rolling(24, min_periods=12).mean().clip(lower=0))
    hl = np.log(d.high / d.low)
    co = np.log(d.close / d.open)
    gk = (0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)).clip(lower=0)
    f['gk_vol_24'] = gk.rolling(24, min_periods=12).mean().pow(0.5)
    f['amihud_24'] = np.log1p((lr.abs() / notional_m).rolling(24, min_periods=12).mean().clip(lower=0))
    avg_sz = np.log1p((notional.clip(lower=0) / trades.clip(lower=1.0)).clip(lower=0))
    f['avg_trade_size_log'] = avg_sz - avg_sz.rolling(48, min_periods=24).mean()
    return f.replace([np.inf, -np.inf], np.nan)


def causal_features(d, max_gap_seconds=None):
    if d.empty or len(d) < 2:
        return _causal_features_one_contiguous_segment(d)
    times = d.index.to_numpy(dtype='datetime64[ns]')
    steps = np.diff(times).astype('timedelta64[ns]').astype(np.int64)
    if max_gap_seconds is None:
        max_gap_seconds = float(CFG['max_gap_seconds'])
    gap_ns = int(max_gap_seconds * 1e9)
    boundaries = np.r_[0, np.where(steps > gap_ns)[0] + 1, len(d)]
    if len(boundaries) == 2:
        return _causal_features_one_contiguous_segment(d)
    parts = [_causal_features_one_contiguous_segment(d.iloc[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])]
    return pd.concat(parts).reindex(d.index)


def causal_feature_normalize(frame, center, scale, gap_seconds=None):
    normalized = (frame[FEATURES] - center) / scale
    w = int(CFG['trailing_feature_window'])
    if w > 1 and len(normalized) > 1:
        cols = [c for c in FEATURES if c not in ('spread_present', 'volume_present')]
        ts = frame.index.to_numpy(dtype='datetime64[ns]')
        steps = np.diff(ts).astype('timedelta64[ns]').astype(np.int64)
        gap_ns = int((gap_seconds if gap_seconds is not None else float(CFG['max_gap_seconds'])) * 1e9)
        groups = np.r_[0, np.cumsum(steps > gap_ns)]
        vals = normalized[cols]
        rolled = vals.groupby(groups, sort=False).rolling(w, min_periods=min(w, 48))
        mu = rolled.mean().reset_index(level=0, drop=True)
        sigma = rolled.std().reset_index(level=0, drop=True).clip(lower=0.5)
        transformed = (vals - mu) / sigma
        normalized.loc[:, cols] = transformed.where(mu.notna(), vals)
    return normalized.clip(-12, 12).fillna(0.0).to_numpy('float32')


def build_closed_timeframe_features(d, freq):
    freq_delta = pd.Timedelta(freq)
    o = d[['open', 'high', 'low', 'close']].resample(freq, closed='right', label='right', origin='epoch').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    )
    for field in ('volume', 'notional_usd', 'signed_notional_usd', 'buy_notional_usd', 'trade_count', 'duration_sec'):
        if field in d.columns:
            o[field] = d[field].resample(freq, closed='right', label='right', origin='epoch').sum(min_count=1)
    o = o.dropna(subset=['open', 'high', 'low', 'close']).copy()
    tf = causal_features(o, max_gap_seconds=max(float(CFG['max_gap_seconds']), freq_delta.total_seconds() * 2.0))
    return tf


class CausalDiagonalStateSpace(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decay_logits = nn.Parameter(torch.linspace(0.5, 5.0, dim))
        self.in_projection = nn.Linear(dim, dim)
        self.out_gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        B, T, D = h.shape
        alpha = torch.sigmoid(self.decay_logits).clamp(0.05, 0.999)
        lags = torch.arange(T, device=h.device, dtype=h.dtype)
        impulse = (1 - alpha[:, None]) * alpha[:, None].pow(lags[None, :])
        states = F.conv1d(self.in_projection(h).transpose(1, 2),
                          impulse.flip(-1).unsqueeze(1), padding=T - 1, groups=D)[..., :T]
        return self.norm(states.transpose(1, 2) * torch.sigmoid(self.out_gate(h)))


class MultiTimeframeFusion(nn.Module):
    def __init__(self, features, hidden, heads, layers):
        super().__init__()
        drop = float(CFG.get('dropout', 0.25))
        self.encoders = nn.ModuleList([nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=drop, batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False) for _ in TF_NAMES])
        self.input = nn.ModuleList([nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden)) for _ in TF_NAMES])
        self.tokens = nn.Parameter(torch.randn(len(TF_NAMES), hidden) * 0.02)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden))
        self.norm = nn.LayerNorm(hidden)
        self.scale = hidden ** -0.5
        self.register_buffer('causal_mask', torch.triu(torch.ones(CFG['mtf_window'], CFG['mtf_window'], dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, query, mtf, mtf_mask):
        B, K, W, Fdim = mtf.shape
        tokens = []
        present = []
        causal = self.causal_mask[:W, :W]
        for j in range(K):
            good = mtf_mask[:, j, :].bool()
            h = self.input[j](mtf[:, j]) + self.tokens[j]
            h = self.encoders[j](h, mask=causal)
            tokens.append(h[:, -1])
            present.append(good.any(1))
        t = torch.stack(tokens, dim=1)
        available = torch.stack(present, dim=1)
        scores = (self.query(query)[:, None, :] * self.key(t)).sum(-1) * self.scale
        scores = scores.masked_fill(~available, -1e9)
        alpha = F.softmax(scores, dim=-1) * available.float()
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp(min=1e-8)
        fused = (alpha.unsqueeze(-1) * self.value(t)).sum(1)
        blended = self.output(torch.cat([query, fused], dim=-1))
        return torch.where(available.any(1, keepdim=True), blended, query), alpha


class UniversalQuantModel(nn.Module):
    def __init__(self, features, hidden, nexperts, topk, nh,
                 transformer_layers, attention_heads, gru_layers, expert_width, ret_scale, rv_scale):
        super().__init__()
        self.ret_scale = ret_scale
        self.rv_scale = rv_scale
        drop = float(CFG.get('dropout', 0.25))
        self.nh = nh
        self.topk = topk
        self.npath = len(CFG['path_knots'])
        self.rank = CFG['path_rank']
        self.input = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden))
        self.conv = nn.Conv1d(hidden, hidden, 5, padding=4, groups=hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, attention_heads, hidden * 4, dropout=drop, batch_first=True, norm_first=True),
            transformer_layers, enable_nested_tensor=False
        )
        self.longrange = nn.GRU(hidden, hidden, num_layers=gru_layers, batch_first=True, dropout=drop if gru_layers > 1 else 0.0)
        self.ssm = CausalDiagonalStateSpace(hidden)
        self.backbone_fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden))
        self.mtf = MultiTimeframeFusion(features, hidden, attention_heads, CFG['mtf_encoder_layers'])
        self.ssl_predict = nn.Linear(hidden, features)
        self.regime = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 7))
        self.router = nn.Sequential(nn.Linear(hidden + 7, hidden), nn.GELU(), nn.Linear(hidden, nexperts))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_width), nn.GELU(), nn.Dropout(drop),
                nn.Linear(expert_width, expert_width), nn.GELU(), nn.Dropout(drop),
                nn.Linear(expert_width, hidden)
            ) for _ in range(nexperts)
        ])
        self.forecast_fusion = nn.Sequential(nn.Linear(hidden * 2 + 7, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden))
        self.direction = nn.Linear(hidden, nh * 3)
        self.quantiles = nn.Linear(hidden, nh * 3)
        self.rv = nn.Linear(hidden, nh)
        self.excursions = nn.Linear(hidden, nh * 2)
        self.barrier = nn.Linear(hidden, nh * 3)
        self.pattern = nn.Linear(hidden, nh)
        self.pattern_detail = nn.Linear(hidden, nh * 3)
        self.jump = nn.Linear(hidden, nh)
        self.path = nn.Linear(hidden, nh)
        self.path_distribution = nn.Linear(hidden, self.npath * (2 + self.rank))
        self.decision_quantiles = nn.Linear(hidden + nh * 3, 3 * 3)
        self.decision = nn.Sequential(nn.Linear(hidden + nh * 3 + 9, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 3))
        self.register_buffer('causal_mask', torch.triu(torch.ones(CFG['lookback'], CFG['lookback'], dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, x, mtf=None, mtf_mask=None):
        B, T, _ = x.shape
        h = self.input(x)
        h = h + self.conv(h.transpose(1, 2))[..., :T].transpose(1, 2)
        causal = self.causal_mask[:T, :T]
        temporal = self.encoder(h, mask=causal)[:, -1]
        recurrent = self.longrange(h)[0][:, -1]
        state_space = self.ssm(h)[:, -1]
        z0 = self.backbone_fusion(torch.cat([temporal, recurrent, state_space], -1))
        if mtf is not None:
            z0, mtf_attention = self.mtf(z0, mtf, mtf_mask)
        else:
            mtf_attention = torch.zeros(B, len(TF_NAMES), device=x.device, dtype=x.dtype)
        regime = self.regime(z0)
        route = self.router(torch.cat([z0, torch.sigmoid(regime)], -1))
        top = route.topk(min(self.topk, route.shape[-1]), dim=-1)
        gate = torch.zeros_like(route).scatter(1, top.indices, F.softmax(top.values.float(), dim=-1).to(route.dtype))
        mixture = torch.zeros_like(z0)
        for j, expert in enumerate(self.experts):
            selected = torch.nonzero(gate[:, j] > 0, as_tuple=True)[0]
            if selected.numel():
                expert_contribution = expert(z0.index_select(0, selected)) * gate[selected, j, None]
                mixture = mixture.index_add(0, selected, expert_contribution.to(dtype=mixture.dtype))
        z = z0 + mixture
        ctx = z + self.forecast_fusion(torch.cat([z, z0, torch.sigmoid(regime)], dim=-1))
        direction = self.direction(ctx).reshape(B, self.nh, 3)
        raw = self.quantiles(ctx).reshape(B, self.nh, 3)
        median = raw[..., 1] * self.ret_scale
        quantiles = torch.stack([median - F.softplus(raw[..., 0]) * self.ret_scale, median, median + F.softplus(raw[..., 2]) * self.ret_scale], dim=-1)
        raw_path = self.path_distribution(ctx).reshape(B, self.npath, 2 + self.rank)
        path_mu = raw_path[..., 0] * self.ret_scale
        path_diag = (F.softplus(raw_path[..., 1]).clamp(max=10.) + .05) * self.ret_scale
        path_factors = torch.tanh(raw_path[..., 2:]) * self.ret_scale
        direction_probs = F.softmax(direction, dim=-1).reshape(B, -1)
        dq = self.decision_quantiles(torch.cat([ctx, direction_probs], dim=-1)).reshape(B, 3, 3)
        utility_median = dq[..., 1]
        action_quantiles = torch.stack([utility_median - F.softplus(dq[..., 0]), utility_median, utility_median + F.softplus(dq[..., 2])], dim=-1) * self.ret_scale
        decision = self.decision(torch.cat([ctx, direction_probs, action_quantiles.reshape(B, -1) / self.ret_scale], -1))
        return {
            'direction': direction,
            'quantiles': quantiles,
            'rv': F.softplus(self.rv(ctx)) * self.rv_scale,
            'excursions': self.excursions(ctx).reshape(-1, self.nh, 2) * self.ret_scale,
            'barrier': self.barrier(ctx).reshape(-1, self.nh, 3),
            'pattern': self.pattern(ctx),
            'pattern_detail': self.pattern_detail(ctx).reshape(-1, self.nh, 3),
            'jump': self.jump(ctx),
            'path_mu': path_mu,
            'path_diag': path_diag,
            'path_factors': path_factors,
            'action_quantiles': action_quantiles,
            'decision': decision,
            'regime': regime,
            'mtf_attention': mtf_attention,
        }


def run_nonoverlapping_backtest(df_bt, horizon_bars=96, fee_rt=0.0008, margin=0.05, label='V8_BTC_DOLLAR'):
    c = df_bt['close'].to_numpy(dtype=np.float64)
    ps = df_bt['p_short'].to_numpy(dtype=np.float64)
    pl = df_bt['p_long'].to_numpy(dtype=np.float64)
    ul = df_bt['u_long'].to_numpy(dtype=np.float64)
    us = df_bt['u_short'].to_numpy(dtype=np.float64)
    results = []
    for thresh in (0.35, 0.40, 0.45, 0.50, 0.55):
        eq = 1.0
        peak = 1.0
        max_dd = 0.0
        trades = []
        l_tr = []
        s_tr = []
        i = 0
        n_rep = len(c)
        while i + horizon_bars < n_rep:
            if pl[i] >= thresh and (pl[i] - ps[i]) >= margin and ul[i] >= us[i]:
                ret = (c[i + horizon_bars] / c[i]) - 1.0 - fee_rt
                trades.append(ret)
                l_tr.append(ret)
                eq *= (1.0 + ret)
                peak = max(peak, eq)
                max_dd = min(max_dd, (eq - peak) / peak)
                i += horizon_bars
            elif ps[i] >= thresh and (ps[i] - pl[i]) >= margin and us[i] >= ul[i]:
                ret = 1.0 - (c[i + horizon_bars] / c[i]) - fee_rt
                trades.append(ret)
                s_tr.append(ret)
                eq *= (1.0 + ret)
                peak = max(peak, eq)
                max_dd = min(max_dd, (eq - peak) / peak)
                i += horizon_bars
            else:
                i += 1
        arr = np.asarray(trades, dtype=np.float64)
        la = np.asarray(l_tr, dtype=np.float64)
        sa = np.asarray(s_tr, dtype=np.float64)
        win = float((arr > 0).mean()) if len(arr) else 0.0
        l_win = float((la > 0).mean()) if len(la) else 0.0
        s_win = float((sa > 0).mean()) if len(sa) else 0.0
        pf = float(arr[arr > 0].sum() / max(abs(arr[arr < 0].sum()), 1e-12)) if len(arr) else 0.0
        sh = float(arr.mean() / max(arr.std(), 1e-12)) if len(arr) > 1 else 0.0
        res = {
            'type': 'non_overlapping', 'horizon_bars': horizon_bars, 'thresh': thresh,
            'trades': len(arr), 'longs': len(la), 'long_win': l_win, 'shorts': len(sa), 'short_win': s_win,
            'win_rate': win, 'profit_factor': pf, 'per_trade_sharpe': sh,
            'total_return_pct': (eq - 1) * 100, 'max_drawdown_pct': max_dd * 100
        }
        results.append(res)
        print(f'[{label}|h={horizon_bars}] Thresh={thresh:.2f} | N={len(arr)} (L={len(la)}/{l_win:.2f}, S={len(sa)}/{s_win:.2f}) | win={win:.3f} | PF={pf:.2f} | sr={sh:.2f} | ret={(eq-1)*100:.2f}% | dd={max_dd*100:.2f}%')
    return results


def run_triple_barrier_backtest(df_bt, horizon_bars=96, fee_rt=0.0008, margin=0.05, label='V8_BTC_DOLLAR'):
    c = df_bt['close'].to_numpy(dtype=np.float64)
    hi = df_bt['high'].to_numpy(dtype=np.float64)
    lo = df_bt['low'].to_numpy(dtype=np.float64)
    lv = df_bt['local_vol'].to_numpy(dtype=np.float64)
    ps = df_bt['p_short'].to_numpy(dtype=np.float64)
    pl = df_bt['p_long'].to_numpy(dtype=np.float64)
    ul = df_bt['u_long'].to_numpy(dtype=np.float64)
    us = df_bt['u_short'].to_numpy(dtype=np.float64)
    mult = float(CFG['barrier_vol_multiplier'])
    min_mv = float(CFG['neutral_move_bps']) * 1e-4
    results = []
    for min_edge in (0.0, 0.0025):
        for thresh in (0.35, 0.40, 0.45, 0.50, 0.55):
            eq = 1.0
            peak = 1.0
            max_dd = 0.0
            trades = []
            l_tr = []
            s_tr = []
            i = 0
            n_rep = len(c)
            while i + horizon_bars < n_rep:
                up_r = max(mult * lv[i] * np.sqrt(horizon_bars), min_mv)
                go_l = up_r >= min_edge and pl[i] >= thresh and (pl[i] - ps[i]) >= margin and ul[i] >= us[i]
                go_s = up_r >= min_edge and ps[i] >= thresh and (ps[i] - pl[i]) >= margin and us[i] >= ul[i]
                if not (go_l or go_s):
                    i += 1
                    continue
                tp = c[i] * np.exp(up_r) if go_l else c[i] * np.exp(-up_r)
                sl = c[i] * np.exp(-up_r) if go_l else c[i] * np.exp(up_r)
                k_exit = horizon_bars
                exit_p = c[i + horizon_bars]
                for k in range(1, horizon_bars + 1):
                    if go_l:
                        hit_sl = lo[i + k] <= sl
                        hit_tp = hi[i + k] >= tp
                        if hit_sl:
                            k_exit = k; exit_p = sl; break
                        if hit_tp:
                            k_exit = k; exit_p = tp; break
                    else:
                        hit_sl = hi[i + k] >= sl
                        hit_tp = lo[i + k] <= tp
                        if hit_sl:
                            k_exit = k; exit_p = sl; break
                        if hit_tp:
                            k_exit = k; exit_p = tp; break
                ret = ((exit_p / c[i]) - 1.0 if go_l else 1.0 - (exit_p / c[i])) - fee_rt
                conf_sz = float(np.clip(((pl[i] - ps[i]) if go_l else (ps[i] - pl[i])) / 0.25, 0.35, 1.0))
                trades.append(ret * conf_sz)
                (l_tr if go_l else s_tr).append(ret * conf_sz)
                eq *= (1.0 + ret * conf_sz)
                peak = max(peak, eq)
                max_dd = min(max_dd, (eq - peak) / peak)
                i += k_exit
            arr = np.asarray(trades, dtype=np.float64)
            la = np.asarray(l_tr, dtype=np.float64)
            sa = np.asarray(s_tr, dtype=np.float64)
            win = float((arr > 0).mean()) if len(arr) else 0.0
            pf = float(arr[arr > 0].sum() / max(abs(arr[arr < 0].sum()), 1e-12)) if len(arr) else 0.0
            sh = float(arr.mean() / max(arr.std(), 1e-12)) if len(arr) > 1 else 0.0
            l_w = float((la > 0).mean()) if len(la) else 0.0
            s_w = float((sa > 0).mean()) if len(sa) else 0.0
            res = {
                'type': 'triple_barrier', 'horizon_bars': horizon_bars, 'min_edge_bps': min_edge * 1e4, 'thresh': thresh,
                'trades': len(arr), 'longs': len(la), 'long_win': l_w, 'shorts': len(sa), 'short_win': s_w,
                'win_rate': win, 'profit_factor': pf, 'per_trade_sharpe': sh,
                'total_return_pct': (eq - 1) * 100, 'max_drawdown_pct': max_dd * 100
            }
            results.append(res)
            print(f'[TB|{label}|h={horizon_bars}|tp>={min_edge*1e4:.0f}bp] T={thresh:.2f} | N={len(arr)} (L={len(la)}/{l_w:.2f}, S={len(sa)}/{s_w:.2f}) | win={win:.3f} | PF={pf:.2f} | sr={sh:.2f} | ret={(eq-1)*100:.2f}% | dd={max_dd*100:.2f}%')
    return results


def run_realistic_backtest(df_bt, horizon_bars=96, spread_usd=0.8, ping_ms=60.0, slippage_bps=0.3, margin=0.05, label='REALISTIC'):
    o = df_bt['open'].to_numpy(dtype=np.float64)
    c = df_bt['close'].to_numpy(dtype=np.float64)
    hi = df_bt['high'].to_numpy(dtype=np.float64)
    lo = df_bt['low'].to_numpy(dtype=np.float64)
    lv = df_bt['local_vol'].to_numpy(dtype=np.float64)
    dur = df_bt['duration_sec'].to_numpy(dtype=np.float64)
    ps = df_bt['p_short'].to_numpy(dtype=np.float64)
    pl = df_bt['p_long'].to_numpy(dtype=np.float64)
    ul = df_bt['u_long'].to_numpy(dtype=np.float64)
    us = df_bt['u_short'].to_numpy(dtype=np.float64)

    ping_s = ping_ms / 1000.0
    half_spread = spread_usd * 0.5
    slip_frac = slippage_bps * 1e-4
    mult = float(CFG['barrier_vol_multiplier'])
    min_mv = float(CFG['neutral_move_bps']) * 1e-4
    results = []

    def exec_entry(idx, is_long):
        # Simulate 60ms network ping from bar idx close into subsequent bar(s)
        rem = ping_s
        k = idx + 1
        n_rep = len(c)
        while k < n_rep - 1 and dur[k] < rem:
            rem -= dur[k]
            k += 1
        frac = min(rem / max(dur[k], 1e-3), 1.0)
        mid_entry = o[k] + frac * (c[k] - o[k])
        if is_long:
            fill = mid_entry + half_spread + mid_entry * slip_frac
        else:
            fill = mid_entry - half_spread - mid_entry * slip_frac
        return k, fill

    # 1. Fixed-horizon realistic execution
    for hb in (24, 96):
        for thresh in (0.35, 0.40, 0.45):
            eq = 1.0; peak = 1.0; max_dd = 0.0; trades = []; l_tr = []; s_tr = []
            i = 0; n_rep = len(c); skipped_fast = 0
            while i + hb + 5 < n_rep:
                go_l = pl[i] >= thresh and (pl[i] - ps[i]) >= margin and ul[i] >= us[i]
                go_s = ps[i] >= thresh and (ps[i] - pl[i]) >= margin and us[i] >= ul[i]
                if not (go_l or go_s):
                    i += 1; continue
                ent_k, entry_p = exec_entry(i, go_l)
                if ent_k > i + 1:
                    skipped_fast += 1
                exit_k = min(ent_k + hb, n_rep - 1)
                # Exit also incurs 60ms ping into exit_k + 1
                _, exit_raw = exec_entry(exit_k - 1, not go_l)
                ret = ((exit_raw / entry_p) - 1.0) if go_l else (1.0 - (exit_raw / entry_p))
                trades.append(ret); (l_tr if go_l else s_tr).append(ret)
                eq *= (1.0 + ret); peak = max(peak, eq); max_dd = min(max_dd, (eq - peak) / peak)
                i = exit_k
            arr = np.asarray(trades, dtype=np.float64)
            la = np.asarray(l_tr, dtype=np.float64); sa = np.asarray(s_tr, dtype=np.float64)
            win = float((arr > 0).mean()) if len(arr) else 0.0
            pf = float(arr[arr > 0].sum() / max(abs(arr[arr < 0].sum()), 1e-12)) if len(arr) else 0.0
            sh = float(arr.mean() / max(arr.std(), 1e-12)) if len(arr) > 1 else 0.0
            l_w = float((la > 0).mean()) if len(la) else 0.0; s_w = float((sa > 0).mean()) if len(sa) else 0.0
            res = {
                'type': 'realistic_fixed', 'horizon_bars': hb, 'thresh': thresh,
                'trades': len(arr), 'longs': len(la), 'long_win': l_w, 'shorts': len(sa), 'short_win': s_w,
                'win_rate': win, 'profit_factor': pf, 'per_trade_sharpe': sh,
                'total_return_pct': (eq - 1) * 100, 'max_drawdown_pct': max_dd * 100,
                'ping_multi_bar_fills': skipped_fast,
            }
            results.append(res)
            print(f'[REAL-FIX|{label}|h={hb}] T={thresh:.2f} | N={len(arr)} (L={len(la)}/{l_w:.2f}, S={len(sa)}/{s_w:.2f}) | win={win:.3f} | PF={pf:.2f} | sr={sh:.2f} | ret={(eq-1)*100:.2f}% | dd={max_dd*100:.2f}%')

    # 2. Triple-Barrier realistic execution (TP = limit fill no slip, SL/Timeout = market fill w/ spread+slip+ping)
    for min_edge in (0.0, 0.0025):
        for thresh in (0.35, 0.40, 0.45):
            eq = 1.0; peak = 1.0; max_dd = 0.0; trades = []; l_tr = []; s_tr = []
            i = 0; n_rep = len(c)
            while i + horizon_bars + 5 < n_rep:
                up_r = max(mult * lv[i] * np.sqrt(horizon_bars), min_mv)
                go_l = up_r >= min_edge and pl[i] >= thresh and (pl[i] - ps[i]) >= margin and ul[i] >= us[i]
                go_s = up_r >= min_edge and ps[i] >= thresh and (ps[i] - pl[i]) >= margin and us[i] >= ul[i]
                if not (go_l or go_s):
                    i += 1; continue
                ent_k, entry_p = exec_entry(i, go_l)
                tp = entry_p * np.exp(up_r) if go_l else entry_p * np.exp(-up_r)
                sl = entry_p * np.exp(-up_r) if go_l else entry_p * np.exp(up_r)
                k_exit = horizon_bars
                exit_p = c[min(ent_k + horizon_bars, n_rep - 1)]
                hit_type = 'timeout'
                for k in range(1, horizon_bars + 1):
                    idx_k = ent_k + k
                    if idx_k >= n_rep:
                        break
                    if go_l:
                        if lo[idx_k] <= sl:
                            k_exit = k; exit_p = sl - half_spread - sl * slip_frac; hit_type = 'sl'; break
                        if hi[idx_k] >= tp:
                            k_exit = k; exit_p = tp; hit_type = 'tp'; break
                    else:
                        if hi[idx_k] >= sl:
                            k_exit = k; exit_p = sl + half_spread + sl * slip_frac; hit_type = 'sl'; break
                        if lo[idx_k] <= tp:
                            k_exit = k; exit_p = tp; hit_type = 'tp'; break
                if hit_type == 'timeout':
                    _, exit_p = exec_entry(min(ent_k + k_exit - 1, n_rep - 2), not go_l)
                ret = ((exit_p / entry_p) - 1.0) if go_l else (1.0 - (exit_p / entry_p))
                conf_sz = float(np.clip(((pl[i] - ps[i]) if go_l else (ps[i] - pl[i])) / 0.25, 0.35, 1.0))
                trades.append(ret * conf_sz)
                (l_tr if go_l else s_tr).append(ret * conf_sz)
                eq *= (1.0 + ret * conf_sz)
                peak = max(peak, eq)
                max_dd = min(max_dd, (eq - peak) / peak)
                i = ent_k + k_exit
            arr = np.asarray(trades, dtype=np.float64)
            la = np.asarray(l_tr, dtype=np.float64); sa = np.asarray(s_tr, dtype=np.float64)
            win = float((arr > 0).mean()) if len(arr) else 0.0
            pf = float(arr[arr > 0].sum() / max(abs(arr[arr < 0].sum()), 1e-12)) if len(arr) else 0.0
            sh = float(arr.mean() / max(arr.std(), 1e-12)) if len(arr) > 1 else 0.0
            l_w = float((la > 0).mean()) if len(la) else 0.0; s_w = float((sa > 0).mean()) if len(sa) else 0.0
            res = {
                'type': 'realistic_tb', 'horizon_bars': horizon_bars, 'min_edge_bps': min_edge * 1e4, 'thresh': thresh,
                'trades': len(arr), 'longs': len(la), 'long_win': l_w, 'shorts': len(sa), 'short_win': s_w,
                'win_rate': win, 'profit_factor': pf, 'per_trade_sharpe': sh,
                'total_return_pct': (eq - 1) * 100, 'max_drawdown_pct': max_dd * 100
            }
            results.append(res)
            print(f'[REAL-TB|{label}|h={horizon_bars}|tp>={min_edge*1e4:.0f}bp] T={thresh:.2f} | N={len(arr)} (L={len(la)}/{l_w:.2f}, S={len(sa)}/{s_w:.2f}) | win={win:.3f} | PF={pf:.2f} | sr={sh:.2f} | ret={(eq-1)*100:.2f}% | dd={max_dd*100:.2f}%')
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, choices=[0, 1, 2], default=0,
                        help='0=all, 1=first half (save cache), 2=second half + backtest')
    parser.add_argument('--from-preds', type=str, default=None,
                        help='Skip inference and run backtests from saved predictions parquet')
    args = parser.parse_args()

    t0 = time.time()
    preds_cache_path = Path('models/v8_btc_dollar/sealed_oos_preds.parquet')

    if args.from_preds:
        print('Loading cached predictions:', args.from_preds)
        pred_df = pd.read_parquet(args.from_preds)
        probs_all = pred_df[['p_short', 'p_wait', 'p_long']].to_numpy()
    else:
        torch.set_num_threads(4)
        ckpt_path = Path('models/v8_btc_dollar/ananta_v8_btc_dollar.pt')
        parquet_path = Path('/tmp/BTCUSD_DOLLAR_500K.parquet')

        print('Loading checkpoint:', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        mc = ckpt['model_config']
        temp = float(ckpt['calibration_temperature'])
        med = pd.Series(ckpt['med'])
        iqr = pd.Series(ckpt['iqr'])
        mtf_scalers = {k: (pd.Series(v['med']), pd.Series(v['iqr'])) for k, v in ckpt['mtf_scalers'].items()}

        model = UniversalQuantModel(
            features=mc['features'], hidden=mc['hidden'], nexperts=mc['nexperts'],
            topk=mc['topk'], nh=mc['nh'], transformer_layers=mc['layers'],
            attention_heads=mc['heads'], gru_layers=mc['gru_layers'],
            expert_width=mc['expert_width'], ret_scale=float(ckpt['ret_scale']),
            rv_scale=float(ckpt['rv_scale']),
        )
        model.load_state_dict(ckpt['model_state'], strict=True)
        model.eval()

        print('Loading parquet:', parquet_path)
        d = load_asset_file(parquet_path)
        end = d.index.max()
        sealed_start = end - pd.DateOffset(months=int(CFG['sealed_months']))
        warmup_start = sealed_start - pd.DateOffset(days=40)
        print(f'Data end: {end} | Sealed OOS start: {sealed_start} | Warmup start: {warmup_start}')

        d_slice = d.loc[d.index >= warmup_start].copy()
        print(f'Sliced bars (with warmup): {len(d_slice)}')

        print('Computing causal features...')
        f = causal_features(d_slice)
        z = causal_feature_normalize(f, med, iqr)
        dates = d_slice.index.to_numpy(dtype='datetime64[ns]')

        print('Building MTF closed candle features...')
        mtf_data = {}
        for tf_name, tf_freq in CFG['mtf_timeframes'].items():
            tf = build_closed_timeframe_features(d_slice, tf_freq)
            tf_med, tf_iqr = mtf_scalers[tf_name]
            tf_z = causal_feature_normalize(tf, tf_med, tf_iqr, gap_seconds=max(float(CFG['max_gap_seconds']), pd.Timedelta(tf_freq).total_seconds() * 2.0))
            tf_valid = np.isfinite(tf[['ret1', 'vol12', 'vol48']].to_numpy()).all(1).astype('bool')
            tf_times = tf.index.to_numpy('datetime64[ns]')
            stops = np.searchsorted(tf_times, dates, side='right').astype('int32')
            mtf_data[tf_name] = (tf_z, tf_valid, stops)

        lb = CFG['lookback']
        w = CFG['mtf_window']
        sealed_np = sealed_start.tz_localize(None).to_datetime64()
        core_ok = np.isfinite(f[['ret1', 'vol12', 'vol48']].to_numpy()).all(1)
        run_bad = np.convolve((~core_ok).astype('int32'), np.ones(lb, dtype='int32'), mode='full')[:len(dates)]

        valid_idx = np.where((dates >= sealed_np) & (np.arange(len(dates)) >= lb - 1) & (run_bad == 0))[0]
        mid = len(valid_idx) // 2
        if args.part == 1:
            run_idx = valid_idx[:mid]
        elif args.part == 2:
            run_idx = valid_idx[mid:]
        else:
            run_idx = valid_idx
        print(f'Sealed OOS inference samples (part={args.part}): {len(run_idx)} / {len(valid_idx)} (elapsed {time.time() - t0:.1f}s)')

        batch_size = 512
        collect = {k: [] for k in (
            'probs', 'uq', 'uq_q10', 'uq_q90', 'dir24', 'dir96',
            'ret24_q', 'ret96_q', 'rv', 'exc96', 'bar96', 'jump96',
            'pat96', 'regime', 'path96_q'
        )}
        t_inf = time.time()
        with torch.no_grad():
            for start_i in range(0, len(run_idx), batch_size):
                batch_ids = run_idx[start_i:start_i + batch_size]
                B = len(batch_ids)
                x_b = np.stack([z[idx - lb + 1:idx + 1] for idx in batch_ids], axis=0)
                mtf_b = np.zeros((B, len(TF_NAMES), w, mc['features']), dtype=np.float32)
                mtf_m = np.zeros((B, len(TF_NAMES), w), dtype=bool)
                for j, tf_name in enumerate(TF_NAMES):
                    tf_z, tf_valid, stops = mtf_data[tf_name]
                    for bi, idx in enumerate(batch_ids):
                        stop = int(stops[idx])
                        s_j = max(0, stop - w)
                        ln = stop - s_j
                        if ln > 0:
                            mtf_b[bi, j, -ln:] = tf_z[s_j:stop]
                            mtf_m[bi, j, -ln:] = tf_valid[s_j:stop]
                out = model(
                    torch.from_numpy(x_b),
                    mtf=torch.from_numpy(mtf_b),
                    mtf_mask=torch.from_numpy(mtf_m),
                )
                collect['probs'].append(F.softmax(out['decision'].float() / temp, dim=-1).numpy())
                aq = out['action_quantiles'].float().numpy()
                collect['uq'].append(aq[..., 1])
                collect['uq_q10'].append(aq[..., 0])
                collect['uq_q90'].append(aq[..., 2])
                dp = F.softmax(out['direction'].float(), dim=-1).numpy()
                collect['dir24'].append(dp[:, 2, :])
                collect['dir96'].append(dp[:, 3, :])
                q_all = out['quantiles'].float().numpy()
                collect['ret24_q'].append(q_all[:, 2, :])
                collect['ret96_q'].append(q_all[:, 3, :])
                collect['rv'].append(out['rv'].float()[:, 2:].numpy())
                collect['exc96'].append(out['excursions'].float()[:, 3, :].numpy())
                collect['bar96'].append(F.softmax(out['barrier'].float()[:, 3, :], dim=-1).numpy())
                collect['jump96'].append(torch.sigmoid(out['jump'].float()[:, 3]).numpy())
                collect['pat96'].append(torch.sigmoid(out['pattern_detail'].float()[:, 3, :]).numpy())
                collect['regime'].append(torch.sigmoid(out['regime'].float()[:, :3]).numpy())
                p_mu = out['path_mu'].float().cumsum(-1)[:, -1]
                p_var = out['path_diag'].float().square().cumsum(-1)[:, -1] + out['path_factors'].float().cumsum(1)[:, -1].square().sum(-1)
                p_sd = p_var.clamp(min=1e-15).sqrt()
                collect['path96_q'].append(torch.stack([p_mu - 1.28155 * p_sd, p_mu, p_mu + 1.28155 * p_sd], dim=-1).numpy())
                if (start_i // batch_size) % 30 == 0:
                    print(f'  Inference {start_i + B}/{len(run_idx)} ({time.time() - t_inf:.1f}s)', flush=True)

        part_dict = {k: np.concatenate(v, axis=0) for k, v in collect.items()}
        part1_cache = Path('/tmp/v8_oos_part1.npz')
        if args.part == 1:
            np.savez(part1_cache, **part_dict)
            print(f'Saved part 1 cache to {part1_cache} ({time.time() - t0:.1f}s)')
            return
        elif args.part == 2:
            p1 = np.load(part1_cache)
            full = {k: np.concatenate([p1[k], part_dict[k]], axis=0) for k in part_dict}
        else:
            full = part_dict

        probs_all = full['probs']
        uq_all = full['uq']
        sel_df = d_slice.iloc[valid_idx]
        lr_slice = np.log(d_slice['close']).diff()
        vol_slice = lr_slice.rolling(48, min_periods=24).std().clip(lower=1e-8).iloc[valid_idx].fillna(1e-4)
        dur_col = sel_df['duration_sec'].fillna(4.34).to_numpy(dtype=np.float64) if 'duration_sec' in sel_df else np.full(len(sel_df), 4.34)

        pred_df = pd.DataFrame({
            'timestamp': sel_df.index,
            'open': sel_df['open'].to_numpy(dtype=np.float64),
            'close': sel_df['close'].to_numpy(dtype=np.float64),
            'high': sel_df['high'].to_numpy(dtype=np.float64),
            'low': sel_df['low'].to_numpy(dtype=np.float64),
            'duration_sec': dur_col,
            'local_vol': vol_slice.to_numpy(dtype=np.float64),
            'p_short': probs_all[:, 0], 'p_wait': probs_all[:, 1], 'p_long': probs_all[:, 2],
            'u_short': uq_all[:, 0], 'u_wait': uq_all[:, 1], 'u_long': uq_all[:, 2],
            'u_short_q10': full['uq_q10'][:, 0], 'u_short_q90': full['uq_q90'][:, 0],
            'u_long_q10': full['uq_q10'][:, 2], 'u_long_q90': full['uq_q90'][:, 2],
            'dir24_dn': full['dir24'][:, 0], 'dir24_up': full['dir24'][:, 2],
            'dir96_dn': full['dir96'][:, 0], 'dir96_up': full['dir96'][:, 2],
            'ret24_q10': full['ret24_q'][:, 0], 'ret24_q50': full['ret24_q'][:, 1], 'ret24_q90': full['ret24_q'][:, 2],
            'ret96_q10': full['ret96_q'][:, 0], 'ret96_q50': full['ret96_q'][:, 1], 'ret96_q90': full['ret96_q'][:, 2],
            'rv24_pred': full['rv'][:, 0], 'rv96_pred': full['rv'][:, 1],
            'mfe96_pred': full['exc96'][:, 0], 'mae96_pred': full['exc96'][:, 1],
            'bar96_short': full['bar96'][:, 0], 'bar96_wait': full['bar96'][:, 1], 'bar96_long': full['bar96'][:, 2],
            'jump96_prob': full['jump96'],
            'fb96_prob': full['pat96'][:, 0], 'rev96_prob': full['pat96'][:, 1], 'cont96_prob': full['pat96'][:, 2],
            'reg_trend': full['regime'][:, 0], 'reg_high_vol': full['regime'][:, 1], 'reg_transition': full['regime'][:, 2],
            'path96_q10': full['path96_q'][:, 0], 'path96_q50': full['path96_q'][:, 1], 'path96_q90': full['path96_q'][:, 2],
        })
        pred_df.to_parquet(preds_cache_path, index=False)
        print(f'Saved FULL OOS predictions cache ({len(pred_df.columns)} cols) to {preds_cache_path}')

    # Buy & Hold benchmark
    bh_ret = (pred_df['close'].iloc[-1] / pred_df['close'].iloc[0] - 1.0) * 100.0
    print(f'\n=== SEALED OOS SUMMARY ({pred_df["timestamp"].iloc[0]} -> {pred_df["timestamp"].iloc[-1]}) ===')
    print(f'Samples: {len(pred_df)} | Buy & Hold Return: {bh_ret:.2f}%')
    print(f'Prob quantiles (p_short / p_wait / p_long) mean: {probs_all.mean(0).round(4).tolist()}')
    print(f'Prob max: short={probs_all[:,0].max():.4f}, wait={probs_all[:,1].max():.4f}, long={probs_all[:,2].max():.4f}')

    print('\n--- Realistic Execution Backtest (Fee=0, Spread=$0.80, Ping=60ms, Slippage=0.3bps/side) ---')
    real_results = run_realistic_backtest(pred_df, horizon_bars=96, spread_usd=0.8, ping_ms=60.0, slippage_bps=0.3)

    out_json = {
        'cohort': 'sealed_oos_realistic',
        'start': str(pred_df['timestamp'].iloc[0]),
        'end': str(pred_df['timestamp'].iloc[-1]),
        'samples': len(pred_df),
        'buy_and_hold_return_pct': bh_ret,
        'execution_params': {'fee': 0.0, 'spread_usd': 0.8, 'ping_ms': 60.0, 'slippage_bps_per_side': 0.3},
        'backtests': real_results,
    }
    out_path = Path('models/v8_btc_dollar/sealed_oos_realistic_backtest.json')
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f'\nSaved realistic backtest results to {out_path} (total {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
