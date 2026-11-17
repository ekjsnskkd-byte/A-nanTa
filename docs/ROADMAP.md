# Delivery Roadmap — gates, not promises

The repository is a research scaffold. A milestone is complete ONLY when its acceptance criteria and test artifact exist.

| Phase | Deliverable | Exit gate | Current |
|---|---|---|---|
| P0 Foundations | Package, interfaces, docs, synthetic fixtures, CI | `pip install`; smoke CLI; unit tests; static lint | **Implemented; CI verification pending** |
| P1 Real market data | Broker quote/tick collectors and separate licensed sources | Per-asset UTC, gap, duplicate, symbol mapping and provenance report | Not started |
| P2 Model integrations | Seven checkpoint adapters with pinned versions and resource limits | Identical cutoff/horizon test + independent inference smoke on real data | Chronos adapter code only |
| P3 Forecast benchmark | Zero-shot benchmark + naive & supervised controls | Walk-forward predictions, cost/latency/VRAM, run manifest | Not started |
| P4 Trained ensemble | OOF meta-learner, regime gate, uncertainty calibration | Gate beats best single baseline on untouched periods; no leakage | Static prototype only |
| P5 Fine-tune only winners | Model-specific LoRA/full or AutoML training | Net marginal gain under same evaluation protocol | Not started |
| P6 Realistic backtest | Event-driven broker-specific margin/spread/swap/latency | Reproducible fills and cash ledger vs reference checks | Not started |
| P7 Paper deployment | Shadow inference, audit trail, replay & kill switch | Operator review; 0 unauthorized orders; correct rollback | Not started |
| P8 Guarded live | Explicitly approved accounts + exposure guardrails | Separate live enablement, loss limits, telemetry and runbook | Not started |

## Working order

### First development sprint
- Review P0 package and decide target first market, timeframe, instrument catalog.
- Implement ingest for MT5 broker-exported tick files and one independent source per asset class.
- Create data quality reports, checksum dataset versions and immutable splits.
- Add Chronos-2 pretrained inference integration test on a non-synthetic fixture.
- Measure **naive baseline vs zero-shot foundation** using identical dates and net costs.

### Second sprint
- Add Kronos and EXAONE wrappers as independent model workers.
- Add TimesFM 3.0 only after the specific checkpoint permissions are verified for the intended educational/live demo.
- Build and persist OOF forecast table keyed by (symbol,timeframe,cutoff,horizon,model_id).
- Compare error correlations before constructing the meta-learner.

### Third sprint
- Integrate the selected Time-MoE 453M-total and Toto 2.0 313M checkpoints alongside TiRex if they show complementary value.
- Integrate walk-forward stacker and learned gate, calibrated abstention.
- Add broker fill simulation and run at least one untouched holdout.

## Definition of done (every PR)

1. Focused change and architecture contract documented.
2. Unit tests for success, malformed data, boundary and leakage paths.
3. Reproducible command, logs, version identifiers and deterministic seeds where possible.
4. Benchmark claim linked to exact dataset split and artifact.
5. Permission/ToS audit for any new checkpoint or market data provider.
6. No live credentials, raw proprietary data, model weights or private emails committed.
7. CI green; reviewer sign-off before deployment.

## Explicitly out of scope for P0

No downloads of multi-billion parameter checkpoints, no backtest profitability claims, no RL agent training, no broker execution, no production UI.
