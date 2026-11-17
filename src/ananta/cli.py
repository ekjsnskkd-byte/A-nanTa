"""Safe research-only command line entrypoint. No live broker order path."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ananta.data import read_ohlcv_csv
from ananta.ensemble import WeightedEnsemble
from ananta.experts.baselines import MeanReversionBaseline, MomentumBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="A-nanTa research CLI (no live trading)")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "demo"):
        command = subs.add_parser(name)
        command.add_argument("--csv", type=Path, required=True)
        if name == "demo":
            command.add_argument("--round-trip-cost-bps", type=float, default=8.0)
            command.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    rows = read_ohlcv_csv(args.csv)
    if args.command == "validate-data":
        print(json.dumps({"status": "valid", "bars": len(rows),
                          "symbol": rows[-1].symbol, "timeframe": rows[-1].timeframe}))
        return
    model = WeightedEnsemble([MomentumBaseline(), MeanReversionBaseline()])
    result = model.forecast(rows, horizon=args.horizon,
                            round_trip_cost_bps=args.round_trip_cost_bps)
    payload = asdict(result)
    payload["forecasts"] = [
        {**asdict(forecast), "observed_at": forecast.observed_at.isoformat()}
        for forecast in result.forecasts
    ]
    payload["research_only"] = True
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
