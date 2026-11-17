# A-nanTa — Technical Architecture

## Design decision

Build one maintainable multi-asset Super Ensemble from separately hosted pretrained models.
Reuse upstream open-source engines, not source-code copies. No large-model variants or
separate challenger roster. Distinguish **selected upstream** from **integrated dependency**
and from **calibrated production model**.

## Planned processing graph

```text
Historically licensed quotes / bars / genuine exchange L2
  -> source provenance + availability timestamp + closed-bar controls
  -> per-market feature adapters
      |-- foundation experts: 7 fixed checkpoints (config/model_registry.yaml)
      |-- regime: jumpmodels [UPSTREAM SELECTED]
      |-- microstructure: broker L1 metrics / TLOB on genuine exchange L2 only
  -> aligned, point-in-time OOF predictions
      |-- quantowl labeling + purged CV [UPSTREAM SELECTED]
      |-- AutoGluon tabular stacker + learned regime gating [UPSTREAM SELECTED]
  -> calibrated aggregate prediction
  -> proposal + abstain / cost filter (research-only code exists)
  -> deterministic independent risk gate + portfolio sizing [PLANNED]
  -> NautilusTrader event-driven simulation [UPSTREAM SELECTED]
  -> paper trading -> separate guarded execution [PLANNED]
```

## Contract boundaries

| Layer | Selected implementation | Current code state |
|---|---|---|
| Source / schema | Broker tick / exchange archives / UTC normalization | CSV reader and Candle contract only |
| Foundation experts | EXAONE, Reverso, TimeH2O, Timestral, Chronos-2, TimesFM 3, TiRex-2 | Synthetic baseline experts; optional untested Chronos wrapper |
| Regime | jumpmodels: discrete / continuous / sparse statistical jump | Heuristic test control only |
| Microstructure | LeonardoBerti00/TLOB when genuine L2 exists | Tested L1 Quote and L2 Snapshot validation/imbalance code |
| Labeling | quantowl | Not integrated |
| Stacking | AutoGluon | Static weighted prototype, no learned stacker |
| Decision / policy | FinRL-X design, quantowl meta-labeling | Fail-closed `assess_forecast` signal proposal, no order sizing |
| Sim / execution | NautilusTrader | Basic cost return and walk-forward functions only |
| Audit | MLflow proposed later | CI tests and Git commit provenance |

## Canonical signals

`Forecast(expert_id, symbol, timeframe, observed_at, horizon, expected_return)`
returns a **fractional price return** over a fixed horizon; not a predicted bid/ask
execution PnL. Before stacking, normalize all model outputs to a consistent target,
target horizon, cutoff, closed-bar convention, and quote denomination. Never average
an asset price forecast with a return forecast or mixing timeframes.

`Quote`: UTC-aware symbol, bid, ask, spread. `L2Snapshot`: exchange,
timestamp and actual bid/ask price/size levels, sorted by price. The presence of
a quote never implies an actual consolidated order book.

## Causal training

- Fit preprocessing and regime features *inside* each walk-forward fold.
- Use purging/embargo when forward targets overlap fold boundaries.
- Construct stacked training rows only from out-of-fold base predictions.
- Feed the learned gating network only inputs available at `observed_at`.
- Cost and execution assumptions are broker/venue-specific and immutable per run.
- Benchmark each model's marginal contribution to after-cost portfolio results.

## Production boundary

The research signal policy is deliberately **not** execution code. A separate
deterministic risk service must control exposure, free margin, drawdown,
stop-out estimates, stale quotes, missing models, kill switch and reconciliation.
No RL or foundation model bypasses this service. Real trading requires operator
approval and validated vendor permissions.

## Operational design

Heavy models run in isolated environments; external weights are version-pinned
with checksums in private artifact storage. CPU-based risk and order execution do
not wait for GPU inference on each tick. Predictions are cached by model hash,
symbol, horizon and observed timestamp. A missing/late prediction fails closed.

## File boundaries

- `src/ananta/contracts.py`: Candle and Forecast typed contracts
- `src/ananta/microstructure.py`: L1/L2 validators and actual L2 imbalance
- `src/ananta/registry.py`: read-only selection validation, no auto-installs
- `src/ananta/decision.py`: research-only proposal with abstention
- `src/ananta/regime.py`: deterministic QA heuristic
- `src/ananta/ensemble.py`: static QA ensemble
- `src/ananta/evaluation.py`: preliminary expanding split and quote cost utilities
- `src/ananta/experts/`: deterministic baselines, optional Chronos wrapper

## Scope note

Upstream TLOB, FinRL-X, jumpmodels, quantowl, AutoGluon stacking and
NautilusTrader are **not yet vendored, installed or exercised** in current tests.
Their upstream licenses and data requirements remain independent obligations.
