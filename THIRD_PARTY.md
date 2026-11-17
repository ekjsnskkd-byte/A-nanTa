# Third-party model and software use

No external model weights, third-party source code, private permissions or market data are committed to A-nanTa.

**Approved pretrained checkpoints (8):** EXAONE Finance 1.0, Reverso Small, TimeH2O-v1, Timestral-123M, Chronos-2, TimesFM 3.0, TiRex-2, MOMENT-1 Small. Refer to [config/model_registry.yaml](config/model_registry.yaml) for exact upstream URLs and per-checkpoint rights-review flags.

**Approved Quant model/method sources (7):** statsmodels MS-AR; statsmodels Kalman State-Space; arch EGARCH; jumpmodels; Bayesian Online Change Point Detection; STUMPY/tslearn-style historical matching; SciPy-assisted First-Passage implementation. Refer to [config/component_registry.yaml](config/component_registry.yaml) for source links.

These selections are research targets, not vendored or installed upstream packages. Review current license terms for both source libraries **and model weights** at the exact version to be used; private permissions require separate verification. Market feed licenses are independent. Software code, checkpoint weights, model cards, training data and distribution rights can have distinct terms.
