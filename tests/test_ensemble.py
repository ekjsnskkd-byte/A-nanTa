from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ananta.contracts import Candle, Forecast
from ananta.data import read_ohlcv_csv
from ananta.ensemble import WeightedEnsemble
from ananta.experts.baselines import MeanReversionBaseline, MomentumBaseline

ROOT = Path(__file__).resolve().parents[1]


def test_demo_produces_research_signal():
    history = read_ohlcv_csv(ROOT / "examples" / "synthetic_m1.csv")
    engine = WeightedEnsemble([MomentumBaseline(), MeanReversionBaseline()])
    result = engine.forecast(history, round_trip_cost_bps=10)
    assert result.action in {"BUY", "SELL", "NO_TRADE"}
    assert sum(result.normalized_weights.values()) == pytest.approx(1.0)
    assert len(result.forecasts) == 2


def test_high_cost_forces_no_trade():
    history = read_ohlcv_csv(ROOT / "examples" / "synthetic_m1.csv")
    engine = WeightedEnsemble([MomentumBaseline(), MeanReversionBaseline()])
    result = engine.forecast(history, round_trip_cost_bps=100_000)
    assert result.action == "NO_TRADE"


def test_duplicate_experts_rejected():
    with pytest.raises(ValueError, match="unique"):
        WeightedEnsemble([MomentumBaseline(), MomentumBaseline()])


def test_misaligned_model_is_rejected():
    class WrongTimestamp:
        expert_id = "wrong"

        def forecast(self, history, horizon):
            bar = history[-1]
            return Forecast(self.expert_id, bar.symbol, bar.timeframe,
                            bar.timestamp + timedelta(minutes=1), horizon, 0.01)

    history = read_ohlcv_csv(ROOT / "examples" / "synthetic_m1.csv")
    weights = {"trend": {"wrong": 1}, "range": {"wrong": 1},
               "volatile": {"wrong": 1}, "unknown": {"wrong": 1}}
    engine = WeightedEnsemble([WrongTimestamp()], weights_by_regime=weights)
    with pytest.raises(ValueError, match="unaligned"):
        engine.forecast(history)
