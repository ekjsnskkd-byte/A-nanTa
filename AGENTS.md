# A-nanTa agent instructions

**Single research scope:** [docs/QUANT_MODEL_PLAN.md](docs/QUANT_MODEL_PLAN.md). Use exactly the 8 selected pretrained experts in [config/model_registry.yaml](config/model_registry.yaml) and 7 Quant methods in [config/component_registry.yaml](config/component_registry.yaml), without extra rosters.

1. Do only what the user requested; inspect files before modifications.
2. Do not add RL portfolio policies, unrelated model candidates or autonomous trading systems.
3. Distinguish selected, installed, fitted, tested and production-ready states.
4. No future data in features, regime probabilities, historical neighbors, calibration or out-of-sample tests.
5. Evaluate market predictions and entry/exit after broker-specific bid/ask costs and realistic fill assumptions.
6. Do not claim profit or accuracy from synthetic fixtures or source publications.
7. Keep market data, model weights, permissions and credentials outside Git.
8. No live broker execution without separate authorization and reviewed risk controls.
9. Run tests where tooling permits; disclose unverified checks.
