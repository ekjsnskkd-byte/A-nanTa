"""Validate selected upstream components; configuration is not an installation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    """Read the active selection only. No downloads, imports or model execution."""
    models = _read(model_path, "models")
    components = _read(component_path, "components")
    if len(models) != 7 or len(components) != 6:
        raise ValueError("expected 7 selected foundation models and 6 selected components")
    for model in models:
        if model.get("status") != "selected_not_installed" or not model.get("checkpoint"):
            raise ValueError("models must be uninstalled candidates with checkpoint IDs")
        if not isinstance(model.get("parameters_m"), (int, float)):
            raise ValueError("parameter count missing")
        if model["parameters_m"] <= 0:
            raise ValueError("parameter count must be positive")
    for component in components:
        if not str(component.get("status", "")).startswith("selected_"):
            raise ValueError("unapproved component state")
        if not component.get("upstream"):
            raise ValueError("missing upstream project URL")
    return Selection(models, components)
