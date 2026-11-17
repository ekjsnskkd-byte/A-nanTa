# Roadmap — A-nanTa selected full-stack ensemble

A milestone is complete **only** when code, reproducible tests and an artifact
prove it. No unrequested model variants or challenger inventories.

| Stage | Work | Acceptance gate | Status |
|---|---|---|---|
| P0 | Typed core, synthetic tests, documentation and GitHub Actions | Lint, unit and smoke tests succeed | Implemented; check current CI |
| P1 | Licensed market-data import, point-in-time provenance | UTC, symbol spec, quote gap & duplicate reports | Not started |
| P2 | Download & individually integrate all 7 selected checkpoints | Real inference, timestamp/horizon checks, resource logs | Chronos adapter source only |
| P3 | Regime and microstructure | Fit regime from training fold; L1 FX/CFDs; trained TLOB only where true L2 | L1/L2 contract added; upstream not integrated |
| P4 | QuantOWL labels, OOF table and AutoGluon stacking | Purged time splits, no target leakage, calibrated error | Static-weight prototype only |
| P5 | Decision, risk and portfolio policy | After-cost abstention, bounded exposure, deterministic hard risk | Research-only signal filter added |
| P6 | NautilusTrader quote/tick/book simulation | Ledger, fills, margin, swap/funding, slippage validation | Not started |
| P7 | Joint benchmark and ablation | Beats best single model net of costs on untouched holdout | Not started |
| P8 | Paper/shadow then controlled live gate | Operator sign-off, kill switch, complete audit trail | Not started |

## Implementation order

1. Install and benchmark individual foundation checkpoints with frozen weight
   revisions. Keep each runtime isolated and source weights outside GitHub.
2. Build market feed import paths (MT5 broker bid/ask ticks and independent FX,
   metals and crypto sources; **no Dukascopy**).
3. Fit the selected jumpmodels library within chronological folds.
4. Train TLOB on venue L2 data where actually supplied; OTC feed uses only real
   quote/spread metrics, never synthetic levels.
5. Use quantowl to generate triple-barrier/meta labels and purged split logic.
6. Train AutoGluon tabular stacker on out-of-fold forecasts across the seven
   models; validate independent learned gating by regime.
7. Add FinRL-X policy only when a broker-cost simulation and fixed risk gate exist.
8. Implement NautilusTrader event-driven portfolio and execution backtest.
9. Compare individual vs ensemble net results, uncertainty, latency and resource
   use; only then promote to paper operation.

## Per-stage completion requirements

- Commit SHA, checkpoint hash, dataset manifest and exact experiment config
- Dataset temporal cutoff and target availability proof
- Unit tests and replayable metrics
- Documented upstream license, rights to weights and data
- No real broker credentials or order calls in training/CI
- Untouched test episodes reserved for selection-grade evidence

No profit factor, accuracy or expected return is asserted from this scaffold.
