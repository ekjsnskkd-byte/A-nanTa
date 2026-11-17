# Model catalog — candidates, not installed models

This is a shortlist, not an accuracy ranking or a claim of trading edge. Review **model card, checkpoint version and weight license** before use. Private permissions from vendors are owned and stored by the operator outside this public repo. They have not been independently inspected here.

| Candidate | Official project | Primary signal | Current integration | License workflow |
|---|---|---|---|---|
| EXAONE Finance 1.0 | https://github.com/LGAI-Research/EXAONE-Forecast | Financial-specialist forecasting | Planned | Weight license NC; verify written exception |
| Kronos-base | https://github.com/shiyu-coder/Kronos | OHLCV/candle sequence | Planned | Verify selected official checkpoint model card |
| Chronos-2 | https://github.com/amazon-science/chronos-forecasting | Quantile/probabilistic forecasting | Optional AutoGluon adapter | Apache 2.0 checkpoint terms; record hash |
| TimesFM 3.0 | https://github.com/google-research/timesfm | Native multivariate forecasting | Planned | Self-hosted pretrained weights NC/non-production by default; verify authorized exception |
| Time-MoE | https://github.com/Time-MoE/Time-MoE | Sparse time-series MoE | Planned | Verify checkpoint terms |
| Toto 2.0 | https://github.com/DataDog/toto | General probabilistic forecasting | Planned | Verify checkpoint terms |
| TiRex-2 | https://github.com/NX-AI/tirex | xLSTM/recurrent diversity | Planned | Verify checkpoint terms |

**Important:** Licensing of GitHub source code != licensing of downloaded model weights != licensing of training data. An email permission can be valid for specific named uses, versions and entities only; check the stated scope before a live classroom demonstration.

## Selection criteria

- Does this expert improve **net ensemble expectancy** on untouched walk-forward periods?
- Does it contribute error diversity beyond current members?
- Is its predicted target aligned to our common forecast contract?
- Is it reliable across Forex, broker CFDs, metals and crypto instruments?
- Is CPU/GPU memory and 99th percentile prediction latency viable for the chosen timeframe?
- Are checkpoint provenance, license and usable dataset clear?

## Installation governance

- No implicit model downloads during `pip install`, `pytest` or `ananta demo`.
- Separate deployment environment per heavy framework to avoid Torch/JAX dependency collisions.
- Pin upstream versions and checkpoint hashes **after** validated experiments.
- Keep pretrained weights in local object storage/cache ignored by Git. Do not vendor models into `src`.
- Track model eligibility as: `candidate` -> `research_allowed` -> `benchmark_approved` -> `production_approved`.
- Store any email grants privately; commit only a nonsensitive approval ID and reviewed status.

## Registered baseline experts

`momentum` and `mean_reversion` in `experts/baselines.py` are low-cost deterministic controls for integration/QA. Their apparent PnL must never be used as evidence of an edge without proper backtesting.
