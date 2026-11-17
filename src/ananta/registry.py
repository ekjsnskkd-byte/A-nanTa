"""Validate the exact approved research roster; selection does not install models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PRETRAINED_IDS = frozenset({
    "exaone_finance_1", "reverso_small", "timeh2o_v1", "timestral_123m",
    "chronos2", "timesfm3", "tirex2", "moment_1_small",
})
QUANT_IDS = frozenset({
    "ms_ar", "kalman_trend", "egarch", "statistical_jump_model",
    "bocpd", "historical_pattern_matching", "first_passage",
})


@dataclass(frozen=True)
class Selection:
    models: tuple[dict, ...]
    components: tuple[dict, ...]


def _read(path: str | Path, key: str) -> tuple[dict, ...]:
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get(key), list):
        raise ValueError(f"{key}: malformed registry")
    records = document[key]
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"{key}: invalid record")
        name = record["id"]
        if not name or name in ids:
            raise ValueError(f"{key}: blank or duplicate ID {name}")
        ids.add(name)
    return tuple(records)


def load_selections(model_path: str | Path, component_path: str | Path) -> Selection:
    """Read only the approved selection; never download or execute weights."""
    models = _read(model_path, "models")
    components = _read(component_path, "components")
    if {m["id"] for m in models} != PRETRAINED_IDS:
        raise ValueError("registry must contain exactly 8 approved pretrained models")
    if {c["id"] for c in components} != QUANT_IDS:
        raise ValueError("registry must contain exactly 7 approved quant methods")
    for model in models:
        if model.get("status") != "selected_not_installed" or not model.get("checkpoint"):
            raise ValueError("pretrained models must be selected but not installed")
    for component in components:
        if component.get("status") != "selected_not_integrated" or not component.get("upstream"):
            raise ValueError("quant methods must be selected but not integrated")
    return Selection(models, components)
