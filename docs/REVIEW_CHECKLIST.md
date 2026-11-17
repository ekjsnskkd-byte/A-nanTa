# A-nanTa review checklist — canonical 8 + 7 plan

## Existing scaffold

- [x] Typed data/forecast contracts and offline market CSV validation
- [x] Selected 8 pretrained checkpoint manifest; no weights checked in
- [x] Selected 7 Quant methods manifest; no estimators falsely marked integrated
- [x] L1/L2 quote validation utilities and preliminary evaluation tools
- [x] Research-only deterministic decision guard, no broker order path

## Not implemented / not validated

- [ ] Fitted MS-AR, Kalman, EGARCH, SJM, BOCPD or historical pattern matching
- [ ] First-Passage / Optimal Stopping solver, calibrated probabilities, cost-aware entry/exit replay
- [ ] End-to-end inference from pretrained weights with real financial data
- [ ] Point-in-time broker data validation, complete purged chronological evaluation
- [ ] Evidence of net incremental edge on a truly untouched sample
- [ ] Separate independently reviewed broker risk/execution boundary
- [ ] Verified current weight licenses and external dependency terms

Do not confuse selected, implemented and benchmarked. Core code tests are not market edge tests.
