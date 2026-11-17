# Research & Validation Protocol

## Non-negotiable anti-leakage rules

1. Fix an inference cutoff; every feature and checkpoint training cutoff must precede its targets.
2. Train/validation/test are **chronological**; never random-split overlapping market episodes.
3. Use purging for labels whose forecast horizons overlap boundaries, and embargo across train/test transitions.
4. Fit normalization, imputers, volatility estimates, regime detectors and meta-learners **inside each training fold**.
5. Construct stacking features from **out-of-fold** predictions only. No in-sample base-model predictions for training a meta-model.
6. Use an untouched final holdout once after model selection. Prevent repeated re-optimization on the same holdout.

`walk_forward_splits` is a minimal expanding splitter with **index embargo only**. It does not yet implement label-horizon purging, CPCV, multiple-testing correction or broker calendars.

## Metrics

**Forecast:** directional accuracy relative to class balance, WQL/CRPS where probabilistic, MAE/MSE on returns, calibration, coverage and by-regime drift.

**Trading:** cumulative and annualized net return, PF, expectancy, trade count/turnover, maximum equity drawdown, worst stop-out, tail losses, Sharpe/Sortino with appropriate sampling, exposure, margin utilization, and bootstrap uncertainty.

**Operational:** latency p50/p95/p99, missing predictions, stale model availability, model-failure fallback, order rejection, fill-price drift, reproducibility and GPU memory.

## Frictions

Replay using broker-appropriate bid/ask, commissions, financing/swaps, slippage and market impact hypotheses, trading sessions, order types, stops, partial fills, contract units, FX conversion, leverage, liquidation/stop-out and funding for perpetuals.

Do not claim backtested alpha if any material friction, symbol spec or fill mechanism is unknown. Mark estimates and sensitivity tests explicitly.

## Ablation

Run each individual model, naive controls, supervised tabular controls and the ensemble on **the same** symbols, timestamps, horizons and cost assumptions. Evaluate incremental value when a model is added/removed. Size of neural network does not establish trading edge.

## Traceability

Every table/chart must store `commit_sha`, `dataset_hash`, `symbol_spec_version`, `training_cutoff`, `checkpoint_hash`, `experiment_config`, `seed`, `evaluation_code_version` and `cost_model_id`. A result without provenance is not decision-grade evidence.
