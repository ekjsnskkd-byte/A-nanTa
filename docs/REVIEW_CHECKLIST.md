# Independent Review Checklist — 2026-11-17

## Present in repository

- [x] Public source scaffold with Python packaging.
- [x] UTC-aware candle checks, quote validation, monotonic series contract.
- [x] Two deterministic expert controls and transparent weighted router.
- [x] Simple regime heuristic, NO_TRADE cost threshold and spread-aware return math.
- [x] Basic chronological walk-forward splitter with configurable embargo.
- [x] Local CSV command line demo and synthetic fixture.
- [x] Optional AutoGluon/Chronos-2 adapter source, without bundled checkpoint.
- [x] Unit tests, static-lint workflow configuration and documented run commands.
- [x] Architecture, data/validation policies, model catalog, delivery gates.

## Not completed / must not be claimed

- [ ] Official model weights downloaded and hashes recorded.
- [ ] EXAONE, Kronos, TimesFM, Time-MoE, Toto and TiRex adapters runnable.
- [ ] All model outputs share verified equivalent target definitions.
- [ ] MT5 / HistData / crypto exchange historical collectors implemented.
- [ ] Complete broker-specific tick execution, margin, stop-out and financing.
- [ ] Proper purged cross-validation and out-of-fold meta-learner trained.
- [ ] Learned dynamic gating & calibrated NO_TRADE threshold.
- [ ] Independent out-of-sample multi-asset benchmark with costs.
- [ ] Drift detection, MLflow registry and recovery tested.
- [ ] Paper/live execution service, risk limits and kill switch.
- [ ] Vendor licenses / signed permissions recorded in private approval registry.
- [ ] GitHub Actions run successful on the current commit (verify from Actions UI).

## Reviewer focus

Prioritize data leakage, source provenance, alignment of forecasting horizon, net transaction costs, model diversity, and provenance of commercial-use permission. All placeholders must be recognizable as such.
