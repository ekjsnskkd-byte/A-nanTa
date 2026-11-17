# A-nanTa

**Modular Super Ensemble AI Trading Research** for Forex, CFDs, cryptocurrency and metals.

**Project maturity: integration scaffold (not a working 7-model ensemble, not approved for live orders).**
This repository contains a synthetic-data prototype, strict interfaces and selected upstream
model/component manifests. No external checkpoint is committed or downloaded by CI.

## One selected stack, no shadow roster

### Selected foundation checkpoints (7)

| Expert | Approx. parameters | Why selected | Installation |
|---|---:|---|---|
| [EXAONE Finance 1.0](https://github.com/LGAI-Research/EXAONE-Forecast) | 202M | Specialized financial forecasting | Not installed |
| [Reverso Small](https://github.com/shinfxh/reverso) | 0.55M | Tiny convolution/recurrent expert | Not installed |
| [TimeH2O-v1](https://huggingface.co/h2oai/timeh2o-v1) | 108M | Probabilistic generalist | Not installed |
| [Timestral 123M](https://github.com/mistralai/timestral) | 123M | Multivariate probabilistic covariates | Not installed |
| [Chronos-2](https://github.com/amazon-science/chronos-forecasting) | 120M | General distribution forecasting | Optional adapter code, not tested with weights |
| [TimesFM 3.0](https://github.com/google-research/timesfm) | 330M | Multivariate long-context forecasting | Not installed |
| [TiRex-2](https://github.com/NX-AI/tirex) | 82.5M total | Recurrent/xLSTM diversity | Not installed |

Approximate total: **966.05 million parameters** (sum of separately hosted checkpoints; not one giant network). See [model catalog](docs/MODEL_CATALOG.md) and [model registry](config/model_registry.yaml). These are **the entire selected foundation roster**; previous candidate sizes are not part of the plan.

### Selected domain components (6)

| Capability | Upstream | What we get |
|---|---|---|
| Regime detection | [regime-detection](https://github.com/Nabel7/regime-detection) | HMM / HSMM / sparse jump models |
| Exchange L2 microstructure | [TLOB](https://github.com/LeonardoBerti00/TLOB) | Limit-order-book learning where genuine L2 is available |
| Meta-labeling / purged CV | [quantowl](https://github.com/landtml/quantowl) | Triple barrier, meta-labeling, purging, diagnostics |
| Master ensemble | [AutoGluon](https://github.com/autogluon/autogluon) | Stacking / automated model combination |
| Trading and portfolio policy | [FinRL-X](https://github.com/AI4Finance-Foundation/FinRL-Trading) | Modular policy and allocation infrastructure |
| Backtest / execution engine | [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | Event-driven multi-asset simulation and execution architecture |

These packages are **selected dependencies, not installed or integrated** simply because they appear in the manifest. Read the [component registry](config/component_registry.yaml) and [integration plan](docs/ARCHITECTURE.md).

**Market limitation:** L2/TLOB only for actual order-book venues, e.g. crypto exchanges. OTC FX/MT5 CFDs generally expose broker quotes rather than a centralized order book. Never synthesize book-depth inputs from candles.

## Fast offline checks

Python 3.11+:

```bash
python -m venv .venv
# Activate your environment
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
ananta stack-status
ananta validate-data --csv examples/synthetic_m1.csv
ananta demo --csv examples/synthetic_m1.csv --round-trip-cost-bps 8
```

- `stack-status` reads the selected manifest; **does not download or execute checkpoints**.
- `demo` uses two deterministic baselines on **synthetic** market data, not the 7 foundation models; these signals have no demonstrated trading edge.
- `src/ananta/microstructure.py` validates real L1 quotes / L2 books; `src/ananta/decision.py` provides a **research-only, fail-closed signal filter** without orders.
- `src/ananta/experts/autogluon_chronos.py` is an optional, unverified Chronos-2 wrapper, not part of the core demo.
- No broker secrets, live-order path, or pretrained weights in GitHub.

## Delivery stages

1. Normalize and validate real broker / exchange tick and L2 data, without Dukascopy.
2. Add true inference adapters and versioned checkpoints for the 7 selected models.
3. Run single-model, same-horizon walk-forward comparisons and OOF predictions.
4. Use quantowl labels + AutoGluon stacking and a trained regime gate; use TLOB only when L2 exists.
5. Validate **incremental net edge** vs individual models, after execution costs, on unseen markets and dates.
6. Reproduce order-level fills and portfolio accounting with NautilusTrader.
7. Paper / shadow test, independently approve limits and licenses, only then consider a separate guarded live system.

## Engineering and review

- [Architecture](docs/ARCHITECTURE.md)
- [Model checkpoints and selection rationale](docs/MODEL_CATALOG.md)
- [Delivery gates](docs/ROADMAP.md)
- [Data sources and restrictions](docs/DATA_SOURCES.md)
- [Validation protocol](docs/VALIDATION.md)
- [Risk and execution boundaries](docs/EXECUTION_POLICY.md)
- [Development](docs/DEVELOPMENT.md)
- [Review checklist](docs/REVIEW_CHECKLIST.md)
- [Dependency license notes](THIRD_PARTY.md)

**License governance:** Public model weight licenses may restrict commercial use or demonstrations involving real orders. User-provided authorizations must be checked by an operator for exact covered parties, weights and uses, and stored privately, not in this public repository.

A-nanTa is research software; no accuracy, trade expectancy or profitability is claimed.
