# A-nanTa

**Canonical research plan:** [Quant prediction, regime and entry/exit](docs/QUANT_MODEL_PLAN.md).

A-nanTa is a multi-asset quantitative trading **research scaffold** for FX/CFDs, metals and crypto. It is **not** a live-trading system or a proven profitable ensemble.

## Approved models — no alternative roster

- **8 selected pretrained experts:** EXAONE Finance 1.0, Reverso Small, TimeH2O-v1, Timestral-123M, Chronos-2, TimesFM 3.0, TiRex-2, and MOMENT-1 Small. These are optional independent experts, not a mandatory EXAONE-centered system. Full identifiers: [model registry](config/model_registry.yaml).
- **7 selected Quant models/methods:** MS-AR, Kalman State-Space, EGARCH, Statistical Jump Model, BOCPD, Historical Pattern Matching, and First-Passage. Full sources: [quant registry](config/component_registry.yaml).
- **Support methods, not extra trading experts:** Stochastic Optimal Stopping, probability calibration, broker-aware entry/exit costs, deterministic risk logic, data validation and walk-forward tests. See [plan](docs/QUANT_MODEL_PLAN.md).

Selection **does not mean installed, fitted, tested or approved**. No checkpoint is committed. The repositories' licensing and weight usage restrictions still need review.

## Current implementation boundary

The repository retains typed market data and forecast contracts, offline CSV validation, bid/ask and genuine L2 data validation, a conservative research-only signal policy, preliminary evaluation helpers, and an **untested optional Chronos-2 adapter**. It has **no running quant prediction models, no installed pretrained checkpoints, no proven net edge and no live order pathway**. Legacy simulated momentum/mean-reversion ensembles have been removed to avoid confusing demos with approved experts.

## Offline commands

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
ananta stack-status
ananta validate-data --csv examples/synthetic_m1.csv
```

These commands **do not download or run models**.

## Current build order

1. Point-in-time market data, quotes and labels; broker-specific symbol conventions.
2. Fit the seven approved Quant methods using strictly historical windows, and integrate only the approved eight pretrained checkpoints when separately authorized.
3. Evaluate direction, trend, regime, volatility and pattern outcome probability with calibrated horizons.
4. Compute first-passage / optimal stopping decisions with executable bid/ask costs.
5. Run chronological walk-forward, ablations and untouched holdout; publish net-cost evidence and model-specific limitations.

No RL portfolio controller, TLOB model, additional upstream ensemble stacker, or separate model candidate roster is approved by this plan.

See [Architecture](docs/ARCHITECTURE.md), [Roadmap](docs/ROADMAP.md), [Model Catalog](docs/MODEL_CATALOG.md), [Validation](docs/VALIDATION.md), [Third-party terms](THIRD_PARTY.md).
