from pathlib import Path

import pytest

from ananta.registry import PRETRAINED_IDS, QUANT_IDS, load_selections

ROOT = Path(__file__).resolve().parents[1]


def test_selected_stack_exact_and_uninstalled():
    result = load_selections(
        ROOT / "config/model_registry.yaml",
        ROOT / "config/component_registry.yaml",
    )
    assert {model["id"] for model in result.models} == PRETRAINED_IDS
    assert {component["id"] for component in result.components} == QUANT_IDS
    assert len(result.models) == 8
    assert len(result.components) == 7
    assert all(model["status"] == "selected_not_installed" for model in result.models)
    assert all(c["status"] == "selected_not_integrated" for c in result.components)


def test_duplicate_model_not_accepted(tmp_path):
    model = tmp_path / "models.yml"
    model.write_text(
        "models:\n"
        "  - id: x\n    status: selected_not_installed\n    checkpoint: x\n"
        "  - id: x\n    status: selected_not_installed\n    checkpoint: x\n"
    )
    with pytest.raises(ValueError, match="duplicate ID"):
        load_selections(model, ROOT / "config/component_registry.yaml")
