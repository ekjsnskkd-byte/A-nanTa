import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validate_command():
    proc = subprocess.run(
        [sys.executable, "-m", "ananta.cli", "validate-data",
         "--csv", str(ROOT / "examples" / "synthetic_m1.csv")],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "valid"
    assert payload["bars"] == 15


def test_demo_is_research_only():
    proc = subprocess.run(
        [sys.executable, "-m", "ananta.cli", "demo",
         "--csv", str(ROOT / "examples" / "synthetic_m1.csv")],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["research_only"] is True
    assert payload["action"] in {"BUY", "SELL", "NO_TRADE"}
