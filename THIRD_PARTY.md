# Selected third-party projects, weights & licenses

**No selected upstream source code, model weights, market data or permission
letters are copied into A-nanTa.** These are external integration targets.

## Seven selected pretrained weights

| Model | Upstream | License-specific attention |
|---|---|---|
| EXAONE Finance | https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0 | Non-commercial default weights; written operator grant must cover actual use |
| Reverso Small | https://huggingface.co/shinfxh/reverso-small | Verify checkpoint terms and CUDA dependencies |
| TimeH2O-v1 | https://huggingface.co/h2oai/timeh2o-v1 | H2O non-commercial weight license; verify written exception |
| Timestral-123M | https://huggingface.co/mistralai/Timestral-123M | Modified MIT model license, separate from Apache code |
| Chronos-2 | https://huggingface.co/amazon/chronos-2 | Verify selected checkpoint revision/license |
| TimesFM 3.0 | https://huggingface.co/google/timesfm-3.0-pytorch | Non-commercial model weights by default; verify written exception |
| TiRex-2 | https://huggingface.co/NX-AI/TiRex-2 | Verify selected checkpoint revision/license |

## Six chosen software components

| Component | Upstream | Reported source license |
|---|---|---|
| Jump model regime detection | https://github.com/Yizhan-Oliver-Shu/jump-models | Apache 2.0 |
| L2 Microstructure | https://github.com/LeonardoBerti00/TLOB | MIT |
| Meta-labeling/CV | https://github.com/landtml/quantowl | MIT |
| Ensemble | https://github.com/autogluon/autogluon | Apache 2.0 |
| Trading policy | https://github.com/AI4Finance-Foundation/FinRL-Trading | Apache 2.0 |
| Backtest | https://github.com/nautechsystems/nautilus_trader | LGPL 3.0 |

Public licenses and developer docs should be rechecked at exact integration
commit, not treated as perpetual or automatically sublicensed rights. Imported
software and pretrained weights can have independent licenses. The absence of
a `LICENSE` file for A-nanTa itself is not a blanket permission to redistribute it.

Historical trading data is independently licensed and may prohibit resale or
redistribution even where model weights or source code are free.

Private user-reported email grants have not been inspected. Record exact
granted scope in a **private** compliance register, never Git or CI logs.
