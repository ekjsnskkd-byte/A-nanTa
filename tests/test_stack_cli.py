import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stack_status_reads_selected_roster():
    proc = subprocess.run(
        [sys.executable, "-m", "ananta.cli", "stack-status",
         "--models", str(ROOT / "config/model_registry.yaml"),
         "--components", str(ROOT / "config/component_registry.yaml")],
        text=True, capture_output=True, check=True,
    )
    status = json.loads(proc.stdout)
    assert status["pretrained_models_selected"] == 8
    assert status["quant_methods_selected"] == 7
    assert status["checkpoint_weights_installed_by_this_command"] == 0
    assert status["research_only"] is True
