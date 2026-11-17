# ADR-0001 — Prefer modular adapters to a monolithic fork

- **Status:** Accepted for bootstrap
- **Date:** 2026-11-17
- **Problem:** A-nanTa must combine a heterogeneous set of pretrained time-series models and later validate strategy performance across OTC FX, CFDs, metal feeds and crypto markets without recreating each architecture.
- **Decision:** Use a small core with stable `Candle` / `Forecast` contracts, independent third-party adapters, and separately deployable offline evaluation, model serving, execution and risk modules. Begin with cheap deterministic baselines and one optional real model integration. No massive upstream code vendoring.
- **Consequences:** Some integration adapters still need implementation and tests, but upstream upgrades, license boundaries, checkpoint provenance and resource isolation stay manageable.
- **Alternatives rejected:** One giant finetuned model from scratch; equal-weight voting without OOF calibration; a premature production trading bot; using synthetic data as performance evidence.
- **Revisit when:** A uniform pretrained forecaster API becomes more mature or measured latency/resource profile justifies another layer.
