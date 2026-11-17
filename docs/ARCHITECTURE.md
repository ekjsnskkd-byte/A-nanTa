# Architecture — A-nanTa Super Ensemble

## Design principles

1. **Composition over reinvention:** import third-party model packages and checkpoints via adapters; do not fork or vendor large upstream repositories by default.
2. **Contracts before models:** all outputs align on symbol, timeframe, observation cutoff, target horizon and units. Reject mismatches.
3. **Prediction is not policy:** forecasts, meta-learner, regime router, risk sizing and execution are separate modules.
4. **Offline evidence before deployment:** out-of-fold temporal predictions feed the stacker; only walk-forward net results qualify models for promotion.
5. **Asset-aware:** shared experts may be reused across assets but symbol-specific calibration, specs and costs remain separate.
6. **Least privilege:** training jobs never receive broker credentials; live sending requires a separate deployable service and explicit enablement.

## Layers

| Layer | Contract | Choices | v0.1.0 state |
|---|---|---|---|
| Acquisition | timestamped source-tagged quotes/candles | MT5, HistData, TrueFX, FXCM, CryptoHFTData, OKX/Binance | Planned |
| Canonical data | UTC, typed Candle, provenance & quality logs | Python / Parquet / Arrow | CSV contract implemented |
| Base experts | Forecast(expected_return, horizon, observed_at) | EXAONE, Kronos, Chronos, TimesFM, Time-MoE, Toto, TiRex | 2 baselines; optional Chronos adapter |
| Regime experts | probabilities over regimes with uncertainty | HMM, jump models, classifiers | Deterministic heuristic only |
| Meta-ensemble | OOF forecasts -> calibrated predictive distribution | AutoGluon / custom stacking / MoE gate | Static weights prototype |
| Decision | order intent + abstention + cost/risk checks | threshold, meta-labeling, portfolio risk | Research signal only |
| Backtest | replay+fills+cash/margin accounting | NautilusTrader | Split/cost helpers only |
| Deployment | paper trading then guarded broker adapter | MT5 / CCXT | Not implemented |
| Audit | model / data / code snapshot + metrics | MLflow, Optuna, GitHub Actions | Basic CI only |

## Required canonical data

- **UTC timestamps with timezone information**; market venue's local trading calendar stored separately.
- Symbol and broker/exchange/instrument metadata, including contract size, quote currency, margin rules and precision.
- Bid/Ask ticks whenever available; OHLCV candles can train models but **cannot substitute for tick-level fill data**.
- Source identifiers, timestamps of availability, revision provenance, missing-data and duplication reports.
- Strictly forward time: available_at <= inference_cutoff, and model targets start after the cutoff. Closed candles only.

`src/ananta/contracts.py` is a minimal application contract, not yet the full Tick/OrderBook schema.

## Forecast contract

Each prediction MUST identify:
- expert_id (stable); symbol; timeframe; observed_at (UTC aware); horizon (bars).
- expected_return in **fractional return units**, e.g. 0.001 = 10 bps.
- Future extensions: quantiles, direction probabilities, uncertainty, latency, model hash, training cutoff, currency normalization.

Do not average raw price forecasts across different underlying quotes or horizons. Adapters must normalize to the contract before ensembling.

## Meta-learning design

1. Generate **out-of-fold time-ordered forecasts** for every expert on identical timestamps.
2. Validate each expert independently for calibration, cost-adjusted utility and complementary error structure.
3. Fit a stacking model on historical OOF forecasts plus only available-at-decision market state.
4. Dynamic gate may condition on regime probabilities, uncertainty, recent drift and spread; never on realized outcomes of the target horizon.
5. Forecast aggregate must pass abstention, transaction cost, exposure limits and market-state checks.

Static regime weights in `src/ananta/ensemble.py` demonstrate routing only. They are **not learned and not calibrated**. Fit/transform split and temporal embargo are mandatory before adding trained gating.

## Research-to-production promotion

Candidate -> reproducible dataset snapshot -> train -> walk-forward OOF -> holdout -> simulation with broker-specific costs -> paper shadow -> risk-controlled live promotion. Every stage records Git SHA, data hash, checkpoint SHA, license approval and metrics. Rollback is mandatory.

## Performance constraints

Large pretrained models should run in **batched asynchronous inference** at appropriate candle horizons. Do not block sub-second order execution waiting for a multi-billion parameter model on each tick. Cache predictions by (model hash, instrument, horizon, asof) and set maximum prediction age.

Separate forecast-serving GPU workers from deterministic risk/execution CPU workers.

## Architecture boundaries

- `experts/`: immutable forecasters; zero broker calls.
- `data.py`: offline CSV adapter with validation.
- `regime.py`: baseline state detector.
- `ensemble.py`: transparent weighting and candidate decision, no execution.
- `evaluation.py`: preliminary walk-forward and round-trip fill cost utilities.
- Future `backtest/`, `risk/`, `execution/`, `serving/` will have their own versioned APIs and independent tests.
