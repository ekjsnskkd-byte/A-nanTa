# A-nanTa — canonical architecture

**Source of truth:** [QUANT_MODEL_PLAN.md](QUANT_MODEL_PLAN.md), [model_registry.yaml](../config/model_registry.yaml), and [component_registry.yaml](../config/component_registry.yaml).

## Approved flow

```text
Licensed, point-in-time market bars / bid-ask / real venue ticks
  -> market-specific features, validation, cutoffs
  -> Quant prediction: MS-AR / Kalman / EGARCH
  -> Quant intelligence: Statistical Jump Model / BOCPD
  -> Pattern expert: past-only matching (+ optional MOMENT embeddings)
  -> optional independent pretrained forecasts (8 selected; none required)
  -> outcome probabilities and calibration using chronological OOF data
  -> First-Passage + Optimal Stopping and executable quote costs
  -> deterministic Buy/Sell/Wait/Close and hard risk checks
  -> research-only walk-forward simulation
```

**No foundation expert is mandatory.** Do not interpret pretrained price forecast as an executable order price. Align horizons, timestamps, symbols, and the return convention before using any forecast.

## Actual code now

- `contracts.py`, `data.py`: typed candle/forecast contracts and CSV input
- `microstructure.py`: quote/L2 data validation utilities (not a TLOB model)
- `decision.py`: research-only deterministic proposal, not broker execution
- `evaluation.py`: preliminary walk-forward and return helpers
- `registry.py`: validates exact approved 8 + 7 selection
- `experts/autogluon_chronos.py`: optional untested Chronos-2 wrapper, **not** an approved standalone AutoGluon ensemble

All listed Quant estimators, pattern retrieval, calibration and entry/exit solvers remain planned; pretrained weights remain uninstalled within this repository. No synthetic demo model or legacy ensemble is part of the approved architecture.

## Safety and evidence

Fit within chronological windows; never leak smoothed online regimes, future pattern outcomes or future bid/ask. Account for spreads, slippage, financing, margin constraints and rejected/missed fills. Keep credentials, checkpoint weights, market data and permission letters out of Git. No live execution until separately authorized and audited.
