# Development

Canonical plan: [QUANT_MODEL_PLAN.md](QUANT_MODEL_PLAN.md). Selection: exactly **8 pretrained + 7 Quant**; supporting methods are not extra experts.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
ananta stack-status
ananta validate-data --csv examples/synthetic_m1.csv
```

The CLI is research-only and does not download model weights. Maintain strict point-in-time input and forecast contracts. Keep broker credentials, private checkpoint weights, vendor permissions and market feeds out of Git. A checkpoint is only integrated when real inference with versioned weights and repeatable tests succeeds.

Implemented utility files include `contracts.py`, `data.py`, `microstructure.py`, `decision.py`, `evaluation.py`, `registry.py`; `experts/autogluon_chronos.py` remains untested Chronos-2-specific code, **not** a general ensemble selection.

Fit Quant methods within chronological training folds; compare each with no-model and no-trade baselines under identical quote costs. Use only observed-history patterns and online regime scores. Do not introduce unapproved components or trading policies.
