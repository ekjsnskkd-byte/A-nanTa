# A-nanTa Review Checklist — 2026-11-17

## Implemented and reviewable

- [x] Importable Python package, synthetic fixture and research-only CLI
- [x] UTC/quote/candle/forecast contracts and validations
- [x] Deterministic QA experts and static regime weighting (not learned)
- [x] Preliminary walk-forward split and spread-aware return function
- [x] One selected seven-checkpoint registry: EXAONE, Reverso Small, TimeH2O,
      Timestral 123M, Chronos-2, TimesFM 3, TiRex-2
- [x] Six selected upstream components, no extra model size tiers
- [x] L1 quote spread and genuine L2 imbalance primitives with tests
- [x] Fail-closed research-only forecast/cost policy with tests
- [x] Optional Chronos-2 adapter code (unverified pretrained inference)
- [x] Source and license review notes, roadmap and CI configuration

## NOT DONE: do not claim otherwise

- [ ] Downloaded/pinned/hashed any of the seven pretrained checkpoints
- [ ] Real inference adapters validated for all seven models
- [ ] Real broker and crypto data ingestion and quality metrics
- [ ] Regime HMM/HSMM/Jump Model trained and integrated
- [ ] TLOB installed and trained on verified exchange L2 data
- [ ] quantowl meta-labeling and purged cross-validation integrated
- [ ] AutoGluon trained seven-forecast OOF meta-learner/gating
- [ ] FinRL-X position/portfolio policy trained and verified
- [ ] NautilusTrader live-quality multi-venue backtest
- [ ] Net edge demonstrated on untouched data after actual costs
- [ ] Independent risk kernel, live account approval or kill switch
- [ ] Vendor permissions inspected and approved for intended uses

## Review priorities

1. **Missing is not installed:** selected upstream != imported model code.
2. Time leakage, as-of alignment and label horizon overlap.
3. Broker-specific spread, swap, margin and liquidity realism.
4. L1 vs L2 separation; no fake FX order-book data.
5. Net incremental benefit of the selected ensemble, not parameter totals.
6. GitHub Actions must pass for the latest commit before signing off.
