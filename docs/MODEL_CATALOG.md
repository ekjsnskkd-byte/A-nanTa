# Selected Model Catalog — 2026-11-17

**Seven selected pretrained forecasting checkpoints, no alternative variants.**
All are **selected but not installed**. A GitHub model card / source alone cannot
prove net-trading performance. The selection is guided by published forecasting
results, architecture diversity and approximate deployment cost.

| Model | Checkpoint | Parameters | Role | Integration |
|---|---|---:|---|---|
| EXAONE Finance 1.0 | https://huggingface.co/LG-AI-Research/EXAONE-Finance-1.0 | 202M | Financial distribution specialist | Planned |
| Reverso Small | https://huggingface.co/shinfxh/reverso-small | 0.55M | Long-convolution/recurrent | Planned; requires CUDA and special kernels upstream |
| TimeH2O-v1 | https://huggingface.co/h2oai/timeh2o-v1 | 108M | General probabilistic | Planned |
| Timestral-123M | https://huggingface.co/mistralai/Timestral-123M | 123M | Multivariate probabilistic | Planned |
| Chronos-2 | https://huggingface.co/amazon/chronos-2 | 120M | Distribution / covariates | Optional adapter code only |
| TimesFM 3.0 | https://huggingface.co/google/timesfm-3.0-pytorch | 330M | General multivariate | Planned |
| TiRex-2 | https://huggingface.co/NX-AI/TiRex-2 | 82.5M total | Recurrent/xLSTM | Planned |

**Total:** about 966.05M parameters across seven independent checkpoints. No
corresponding hardware claim follows from weight count alone; Reverso requires
CUDA-specific kernels and models have different dependencies, caches and contexts.

## Selection caveats

- Some general-purpose benchmark leaders have not been compared against all
  these models on *the same* financial datasets.
- Published FinVerse evidence supports EXAONE Finance's financial applicability;
  it is not an order-fill simulation for real FX/CFD accounts.
- Reverso Small is 550K parameters, with public checkpoint. Upstream indicates
  the larger Reverso checkpoints are not released; do not silently swap.
- A forecasting model is not a calibrated trading policy or an execution model.
- No pretrained weight, vendor permission email, or private market data is kept in Git.

## Licenses

EXAONE Finance, TimeH2O-v1 and TimesFM 3.0 carry checkpoint-specific restrictions
under default public terms. User-reported permission letters are not independently
verified and must be reviewed for the precise deployment, including educational
demonstrations with real money. Timestral weight terms differ from code's Apache 2.0.

Record approved entitlement IDs and exact weight revisions **privately**. A license
for inference code does not grant rights to a checkpoint or training dataset.

## Required benchmarks to certify suitability

Same time ranges, symbols, forecast horizon, sampling, broker costs, and as-of
data for each individual model; evaluate marginal ensemble gain and error
correlation. Distinguish outright zero-shot, optional fine-tuning and stacked
models. All promotion requires untouched time holdouts and net PnL.
