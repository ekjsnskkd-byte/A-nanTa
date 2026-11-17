# Development & Integration

Python 3.11+; clean environment.

```bash
python -m venv .venv
# activate the venv
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
ananta stack-status
ananta validate-data --csv examples/synthetic_m1.csv
ananta demo --csv examples/synthetic_m1.csv --round-trip-cost-bps 8
```

The CLI is **research-only**. `stack-status` reads the selected registries but
does not download/install models. Integration tests for pretrained models must
run separately with explicit operator consent for large downloads.

## Layout

```text
config/
  model_registry.yaml     # only the 7 selected pretrained checkpoints
  component_registry.yaml # one selected upstream for each missing capability
src/ananta/
  registry.py             # validates both manifest files
  contracts.py            # point-in-time candle and forecast contracts
  data.py                 # offline CSV import (not a data lake)
  microstructure.py       # tested L1 quote and L2 snapshot helpers
  decision.py             # research-only, no order sending
  regime.py               # heuristic placeholder, not fitted HMM/HSMM
  ensemble.py             # static baseline weights only
  evaluation.py           # basic splits and return metric
  experts/                # 2 baseline experts, optional Chronos-2 code
tests/                    # offline / no model downloads
docs/                     # research and operational controls
```

## Wiring third-party projects

1. Use `config/model_registry.yaml` and `config/component_registry.yaml`
   as the **only selected** roster. No shadow list of older models.
2. Install each heavy upstream in its own environment, pin repository revision
   and exact checkpoint hash, and store licensing permissions privately.
3. Implement each Expert adapter to produce a point-in-time
   `Forecast` with an identical horizon and normalized fractional return.
4. Add negative tests for stale quotes, symbols, invalid/missing weights,
   unsupported frequencies, and training cutoffs.
5. Persist out-of-fold predictions before quantowl / AutoGluon training.
6. Use an independent deterministic risk gate for every trade intent.
7. Do not merge performance claims without source data and full net-cost replay.

### Microstructure requirements

For broker FX/CFDs use actual L1 Bid/Ask; `Quote` provides spread only.
For exchange crypto with genuine size/depth levels, use `L2Snapshot` and
then an adapted TLOB pipeline. No candle-to-book construction.

### Reverso performance note

Its released 550K checkpoint requires CUDA-compatible GPU and specialized
FlashFFTConv dependencies, despite its tiny parameter count. Record actual
environment/memory requirements rather than inferring them from weight size.
