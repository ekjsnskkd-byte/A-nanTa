# Development guide

## Prerequisites

Python 3.11+, Git, isolated virtualenv. The base package intentionally installs **no heavy dependencies**.

```bash
python -m venv .venv
# source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate          # PowerShell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
ananta validate-data --csv examples/synthetic_m1.csv
ananta demo --csv examples/synthetic_m1.csv --round-trip-cost-bps 8
```

## Source layout

```text
src/ananta/
  contracts.py              # typed, timezone-safe Candle/Forecast interfaces
  data.py                   # strict local CSV adapter
  regime.py                 # heuristic QA detector (replace with trained)
  ensemble.py               # simple weighted router & NO_TRADE prototype
  evaluation.py             # basic walk-forward / spread-aware returns
  experts/
    base.py                 # Expert Protocol
    baselines.py            # deterministic controls
    autogluon_chronos.py    # optional pretrained Chronos-2 integration
  cli.py                    # research-only commands
tests/                      # fast offline tests
examples/                   # synthetic fixture, not market evidence
config/                     # manifests and eligibility, human reviewed
docs/                       # architecture, research policy and roadmap
```

## Pretrained model experiments

Optionally install `pip install -e ".[forecast]"` in a separate environment. Model downloads, inference and GPU checks are **opt-in** and never performed by CI. For Chronos-2, first build a time-ordered historical `train_history`; initialize and call `fit(train_history)`, then pass later point-in-time `history` to `forecast(history, horizon)`. Check memory, frequency/calendar and training cutoff yourself before treating results as valid.

## Contributing a new expert

1. Implement `expert_id` and `forecast(history, horizon) -> Forecast`.
2. Never consume future observations, hidden broker data or testing labels.
3. Return fractional **price return**, not an unlabelled price or probability.
4. Add unit tests: stale cutoff, symbol mismatch, horizon, NaNs, missing weights, missing dependencies.
5. Document official checkpoint, license/permission scope, dependency versions and resource footprint.
6. Write a separate offline benchmark before claiming any benefit.

## Branching and review

Use small `feature/...` branches and focused PRs; run all tests and lint. Do not merge unverified experiments to main. Keep raw datasets, tokens, proprietary checkpoints and license emails outside the public repo.
