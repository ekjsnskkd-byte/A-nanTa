# A-nanTa

**Research-first, multi-asset Super Ensemble AI infrastructure** for Forex, broker CFDs, metals and cryptocurrency.

> **Project state: Bootstrap v0.1.0.** The Python research scaffold and a deterministic baseline ensemble are implemented. The seven foundation-model candidates are **NOT downloaded, benchmarked, or active as a production ensemble**. The only external-model adapter currently in code is an **optional** AutoGluon Chronos-2 path. Live broker execution is **not implemented or enabled**.

## Project mission

Reuse reputable open-source frameworks and pretrained checkpoints, connect diverse forecasting architectures behind a unified interface, and validate whether their combined predictions yield **incremental net trading edge**. Do not train a giant foundation model from scratch, and do not confuse model size with trading profitability.

## System at a glance

```text
Historical market data (Forex / CFDs / Crypto / Metals)
            |
 Quality, provenance, UTC normalization, leakage checks
            |
    Unified Candle / Forecast contracts
            |
  +---------+----------+----------+----------+
  | Financial / OHLCV  | TS foundation models |
  | EXAONE / Kronos    | Chronos / TimesFM    |
  | Time-MoE / Toto    | TiRex                |
  +---------+----------+----------+----------+
            |
 Regime detector + out-of-fold predictions
            |
 Calibrated meta-learner / dynamic gating [PLANNED]
            |
   Costs / risk / NO_TRADE / decision policy
            |
 Event-driven backtest > paper > guarded live [PLANNED]
```

The v0.1.0 implementation runs two *deterministic baseline experts* and a heuristic regime router to verify interfaces and tests. These are **engineering controls, not demonstrated trading alpha**.

## Quickstart (Python 3.11+)

```bash
git clone https://github.com/ekjsnskkd-byte/A-nanTa.git
cd A-nanTa
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
ananta validate-data --csv examples/synthetic_m1.csv
ananta demo --csv examples/synthetic_m1.csv --round-trip-cost-bps 8
```

The example file is **synthetic** and the CLI prints a **research-only illustrative signal**. It does not access a broker or place an order.

### Optional pretrained experiment: Chronos-2

```bash
python -m pip install -e ".[forecast]"
```

`src/ananta/experts/autogluon_chronos.py` implements a lazy, optional AutoGluon Chronos-2 adapter. Its `fit(train_history)` method may download the official pretrained weights. **Training/validation windows and sufficient RAM/VRAM remain the operator's responsibility.** The adapter is not used in the default demo or CI.

## Candidate foundation model roster

| Candidate | Primary strength | Integration state |
|---|---|---|
| EXAONE Finance 1.0 | Financial distribution specialist | Research candidate |
| Kronos-base | OHLCV/candlestick specialist | Research candidate |
| Chronos-2 | Probabilistic time-series forecasting | Optional AutoGluon adapter |
| TimesFM 3.0 | Multivariate generalist | Research candidate, license gated |
| Time-MoE (453M total; TimeMoE-200M) | Sparse MoE time-series model | Selected checkpoint candidate |
| Toto 2.0 (313M) | Scalable probabilistic TS forecasts | Selected checkpoint candidate |
| TiRex-2 | Recurrent/xLSTM diversity expert | Research candidate |

**Size selection:** Time-MoE 453M total and Toto 2.0 313M are the only selected sizes; the plan does not include larger size alternatives. **No "best trading model" has yet been proven.** Pretrained checkpoint weight availability and permissible uses must be reviewed independently of code licenses. Private email authorizations are not stored in this public repository.

## Documents for technical review

- [Architecture and component boundaries](docs/ARCHITECTURE.md)
- [Implementation roadmap and acceptance gates](docs/ROADMAP.md)
- [Model registry and licensing](docs/MODEL_CATALOG.md)
- [Data sources and input contracts](docs/DATA_SOURCES.md)
- [Evaluation and anti-leakage protocol](docs/VALIDATION.md)
- [Execution safety and promotion criteria](docs/EXECUTION_POLICY.md)
- [Integration and local development](docs/DEVELOPMENT.md)
- [Current review checklist](docs/REVIEW_CHECKLIST.md)
- [Architecture decision record](docs/adr/0001-modular-ensemble.md)
- [Third-party dependency notes](THIRD_PARTY.md)

## Current scope and explicit exclusions

Implemented: importable Python package; strict timestamp/quote contracts; CSV validator; two baseline experts; simple regime heuristic; cost-aware NO_TRADE prototype; basic walk-forward split utility; spread-aware trade-return function; optional Chronos-2 adapter; tests and CI.

Not yet implemented: external market-data collectors; live trained gating/meta-learner; regime-specific trained models; full multi-model foundation inference; true multi-asset portfolio simulator; broker execution; realistic order book / margin / swap; production model serving; profitable strategy.

**This repo does not include third-party pretrained weights, proprietary market data, secrets, or commercial permissions.** A-nanTa source licensing has not been chosen; do not assume blanket reuse rights for the original repository code.
