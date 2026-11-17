# Approved pretrained model catalog — 2026-11-17

**Eight selected, not installed or validated.** None is required as the central model; all may contribute only if separately supported by out-of-sample evidence.

| Model | Checkpoint | Role |
|---|---|---|
| EXAONE Finance 1.0 | https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0 | Financial price forecasting |
| Reverso Small | https://huggingface.co/shinfxh/reverso-small | Lightweight forecasting |
| TimeH2O-v1 | https://huggingface.co/h2oai/timeh2o-v1 | Probabilistic forecasting |
| Timestral-123M | https://huggingface.co/mistralai/Timestral-123M | Multivariate probabilistic forecasting |
| Chronos-2 | https://huggingface.co/amazon/chronos-2 | General probabilistic forecasting |
| TimesFM 3.0 | https://huggingface.co/google/timesfm-3.0-pytorch | Multivariate forecasting |
| TiRex-2 | https://huggingface.co/NX-AI/TiRex-2 | Recurrent forecasting |
| MOMENT-1 Small | https://huggingface.co/AutonLab/MOMENT-1-small | Time-series representation and pattern similarity |

These weights have different usage rights. Verify exact model-card terms, revisions and any private permissions before use. MOMENT embeddings are not calibrated P(up) probabilities. Model weight availability does not prove trading edge.

**Quant:** see [QUANT_MODEL_PLAN.md](QUANT_MODEL_PLAN.md) and [component_registry.yaml](../config/component_registry.yaml) for exactly seven selected statistical/algorithmic methods. Calibration, optimal stopping and transaction-cost accounting are required supporting techniques, **not additional pretrained models**.
