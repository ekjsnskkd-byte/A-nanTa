# A-nanTa — BTC Dollar-Bar Multi-Head Quantitative Model (V8)

A-nanTa is a causal, event-time quantitative research and backtesting framework for **BTCUSD** using **$500K Dollar Bars** (`BTCUSD_DOLLAR_500K.parquet`) and a multi-head hybrid architecture (`UniversalQuantModel`).

---

## Architecture (`UniversalQuantModel`)

- **Backbone**: Causal 1D Conv + Causal Transformer (3 layers, 4 heads) + 2-layer GRU + Causal Diagonal State-Space Model (SSM)
- **Multi-Timeframe Fusion (MTF)**: Closed-candle causal cross-attention over `M15`, `H1`, `H4`, `D1`
- **Sparse MoE**: 6 experts (`top_k = 2`) conditioned on latent state + 7-dim regime logits
- **Multi-Head Outputs** (`horizons = [2, 6, 24, 96]` dollar bars, `2,422,077` active parameters):
  - `decision` (`Short / Wait / Long`) & `action_quantiles` (`q10, q50, q90`)
  - `direction` & `quantiles` (`q10, q50, q90` returns per horizon)
  - `path_mu`, `path_diag`, `path_factors` (12-knot Low-Rank Multivariate Normal price path distribution)
  - `rv` (Realized Volatility), `excursions` (`MFE / MAE`), `barrier` (Triple-Barrier probs), `jump`, `pattern_detail` (`false_breakout / reversal / continuation`), `regime`

---

## Repository Structure

```text
├── README.md
├── models/
│   └── v8_btc_dollar/
│       ├── ananta_v8_btc_dollar.pt               # Frozen V8 BTC Dollar-Bar checkpoint (9.4 MB)
│       ├── calibration.json                      # Temperature scaling (T = 1.3820, N = 113,270)
│       ├── data_coverage.json                    # Dataset stats (11,804,813 bars, 2020-10 to 2026-10)
│       ├── training_history.json                 # 7-stage progressive training logs
│       ├── training_quality_diagnostics.json     # Validation cohort diagnostics (N = 109,323)
│       ├── sealed_oos_backtest.json              # Sealed OOS backtest (8 bps RT fee)
│       ├── sealed_oos_realistic_backtest.json    # Sealed OOS realistic execution backtest
│       └── sealed_oos_preds.parquet              # Cached Sealed OOS predictions (all 44 columns)
├── notebooks/
│   ├── training/
│   │   ├── A-nanTa_V7314_TRAINING_QUALITY_GATES.ipynb
│   │   └── A-nanTa_V8_FULLPOWER_ENSEMBLE.ipynb
│   └── tests/
│       ├── A-nanTa_V7314_DOWNLOAD_AUDIT.ipynb
│       ├── A-nanTa_V7314_L4_SMOKE_TEST.ipynb
│       └── A-nanTa_V731_ISOLATED_L4_DIAGNOSTIC.ipynb
└── scripts/
    ├── backtest_v8_btc_dollar.py                 # Standalone OOS inference & realistic backtest
    ├── sim_cent_2000.py                          # 2,000 USC Cent account (1:2000 lev) simulator
    └── sim_multihead_cent.py                     # Multi-head consensus + risk filter + dynamic TP/SL
```

---

## Training & Sealed OOS Evaluation Summary

- **Dataset**: `BTCUSD_DOLLAR_500K.parquet` (`11,804,813` bars, `2020-10-01` to `2026-10-31`, median bar duration `4.342s`)
- **7-Stage Progressive Training**: `A_SSL` (5 ep) → `B_REGIME` (3 ep) → `D_FORECAST` (10 ep) → `E_PATH` (8 ep) → `F_EXPERTS` (7 ep) → `G_DECISION` (5 ep) → `H_JOINT` (5 ep)
- **Sealed OOS Cohort**: `2026-08-01` to `2026-10-31` (`124,978` bars, Buy & Hold `+23.31%`)

### Realistic Execution Backtest (`0` Fee, `\$0.80` Spread, `60 ms` Ping, `0.3 bps/side` Slippage)

| Strategy | Threshold | Trades (L / S) | Win Rate | Profit Factor | Return | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fixed-Horizon (`h=24`) | `0.35` | 443 (274 / 169) | 51.7% | 1.16 | `+5.73%` | -2.00% |
| Fixed-Horizon (`h=96`) | `0.35` | 294 (179 / 115) | 51.4% | 1.18 | `+9.14%` | -5.46% |
| Triple-Barrier (`h=96, tp>=25bp`) | `0.45` | 30 (27 / 3) | 63.3% | 1.72 | `+0.81%` | -0.34% |

### Multi-Head Strategy on Cent Account (`2,000 USC`, `1:2000` Max Leverage, `20%` Stop-Out)

Combining `decision`, `action_quantiles`, `barrier`, `direction`, `quantiles`, `path_quantiles` consensus, `jump`/`false_breakout`/`regime_transition` risk filters, and dynamic `MFE/MAE/RV` TP/SL:

| Configuration | Trades | Win Rate | PF | Final Balance (USC) | Return | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (`Decision` only, `T=0.40`, 5% risk) | 233 | 49.8% | 0.98 | `1,362.59` | `-31.87%` | -70.54% |
| `+Consensus +RiskFilters` (`T=0.40`, 5% risk) | 101 | 52.5% | 1.20 | `2,708.70` | `+35.44%` | `-21.89%` |
| **`+ALL HEADS (+Dynamic TP/SL)` (`T=0.35`, 5% risk)** | **183** | **55.7%** | **1.47** | **`7,418.10`** | **`+270.90%`** | **`-31.14%`** |
| `+ALL HEADS (+Dynamic TP/SL)` (`T=0.35`, 10% risk) | 183 | 55.7% | 1.47 | `18,420.07` | `+821.00%` | `-56.62%` |

---

## Quick Start

Run simulations directly from the cached Sealed OOS predictions (`models/v8_btc_dollar/sealed_oos_preds.parquet`):

```bash
# Multi-head strategy ablation on 2,000 USC Cent account (1:2000 leverage)
python scripts/sim_multihead_cent.py

# Leverage & risk sweep on 2,000 USC Cent account
python scripts/sim_cent_2000.py

# Re-run realistic backtest from cached predictions
python scripts/backtest_v8_btc_dollar.py --from-preds models/v8_btc_dollar/sealed_oos_preds.parquet
```
