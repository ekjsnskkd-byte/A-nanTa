# Contributing

A-nanTa aims to reuse existing models responsibly and prioritize reproducible quant research.

- Create focused feature branches and PRs.
- Run `python -m pytest -q` and `python -m ruff check src tests` before review.
- Declare every external dependency, training data source, artifact checksum and time cutoff.
- Add negative tests for leakage, invalid quotes, missing model weights and horizon mismatches.
- Keep pretrained weights and market datasets out of Git.
- Do not claim live readiness, accuracy or profitability from synthetic fixtures.
- Do not add order-placement functionality to the research CLI.
- Do not merge unrequested changes or change baseline performance criteria without approval.

Coding conventions: Python 3.11+, type annotations for public APIs, standard-library core, deterministic fixtures, UTC.
