"""Research-only CLI: inspect chosen models and validate offline data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ananta.data import read_ohlcv_csv
from ananta.registry import load_selections


def main() -> None:
    parser = argparse.ArgumentParser(description="A-nanTa research CLI (no live trading)")
    subs = parser.add_subparsers(dest="command", required=True)
    validate = subs.add_parser("validate-data")
    validate.add_argument("--csv", type=Path, required=True)
    selected = subs.add_parser("stack-status")
    selected.add_argument("--models", type=Path, default=Path("config/model_registry.yaml"))
    selected.add_argument("--components", type=Path,
                          default=Path("config/component_registry.yaml"))
    args = parser.parse_args()

    if args.command == "stack-status":
        selection = load_selections(args.models, args.components)
        print(json.dumps({
            "pretrained_models_selected": len(selection.models),
            "quant_methods_selected": len(selection.components),
            "checkpoint_weights_installed_by_this_command": 0,
            "model_ids": [m["id"] for m in selection.models],
            "quant_ids": [q["id"] for q in selection.components],
            "research_only": True,
        }, indent=2))
        return

    rows = read_ohlcv_csv(args.csv)
    print(json.dumps({
        "status": "valid", "bars": len(rows),
        "symbol": rows[-1].symbol, "timeframe": rows[-1].timeframe,
    }))


if __name__ == "__main__":
    main()
