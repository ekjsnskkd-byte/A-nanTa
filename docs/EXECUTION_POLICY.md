# Execution separation & safe deployment

## Current status

There is **no live execution code** and no broker login path in v0.1.0. CLI signals are illustrative and use synthetic input. `ANANTA_LIVE_TRADING_ENABLED=false` documents this policy; it is not a switch that magically enables live trading.

## Production service boundaries (future)

- **Model serving:** consumes read-only market snapshots and returns timestamped forecasts.
- **Policy:** turns forecasts into proposed position changes using exposure, confidence and expected net edge.
- **Risk:** independently checks capital, free margin, symbol caps, drawdown, stale prices, feed health and daily loss limits. Never delegate hard limits to an LLM or RL policy.
- **Execution:** a separate process with broker/exchange credentials, idempotent orders, acknowledgement/reconciliation, cancellation, retry policy and immutable audit log.
- **Kill switch:** explicit operator control, server-side and independent of model predictions.

## Promotion gates

Offline tests -> reproducible broker-cost simulation -> paper shadow -> controlled live demonstration. A live classroom demonstration is still **real trading** and must be scoped under specific permissions and broker rules.

No unattended live order placing, no secret storage in GitHub Actions, no broker credentials in training workers, no unsafe default order sizing. Require operator approval and recorded environment configuration for all live activity.
