from pathlib import Path

import pytest

from ananta.registry import load_selections

ROOT = Path(__file__).resolve().parents[1]


def test_selected_stack_exact_and_uninstalled():
    result = load_selections(ROOT / "config/model_registry.yaml",
                             ROOT / "config/component_registry.yaml")
    ids = {model["id"] for model in result.models}
    assert ids == {
        "exaone_finance_1", "reverso_small", "timeh2o_v1",
        "timestral_123m", "chronos2", "timesfm3", "tirex2",
    }
    assert len(result.components) == 6
    assert all(model["status"] == "selected_not_installed" for model in result.models)
    assert sum(model["parameters_m"] for model in result.models) == pytest.approx(966.05)


def test_duplicate_model_not_accepted(tmp_path):
    m = tmp_path / "models.yml"
    c = ROOT / "config/component_registry.yaml"
    m.write_text(
        "models:\n"
        "  - id: x\n    status: selected_not_installed\n    checkpoint: x\n"
        "  - id: x\n    status: selected_not_installed\n    checkpoint: x\n"
    )
    with pytest.raises(ValueError, match="duplicate ID"):
        load_selections(m, c)
