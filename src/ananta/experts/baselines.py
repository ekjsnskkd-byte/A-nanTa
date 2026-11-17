"""Deterministic research controls, not validated trading alpha."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from ananta.contracts import Candle, Forecast, validate_history


@dataclass(slots=True)
class MomentumBaseline:
    lookback: int = 5
    expert_id: str = "momentum"

    def forecast(self, history: Sequence[Candle], horizon: int) -> Forecast:
        if self.lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        validate_history(history, self.lookback + 1)
        latest = history[-1]
        anchor = history[-self.lookback - 1].close
        estimate = (latest.close / anchor - 1.0) * horizon / self.lookback
        return Forecast(self.expert_id, latest.symbol, latest.timeframe,
                        latest.timestamp, horizon, estimate)


@dataclass(slots=True)
class MeanReversionBaseline:
    lookback: int = 5
    strength: float = 1.0
    expert_id: str = "mean_reversion"

    def forecast(self, history: Sequence[Candle], horizon: int) -> Forecast:
        if self.lookback < 2 or horizon < 1 or not (0.0 <= self.strength <= 1.0):
            raise ValueError("invalid lookback, horizon or strength")
        validate_history(history, self.lookback)
        latest = history[-1]
        average = mean(bar.close for bar in history[-self.lookback:])
        estimate = (average / latest.close - 1.0) * self.strength
        return Forecast(self.expert_id, latest.symbol, latest.timeframe,
                        latest.timestamp, horizon, estimate)
