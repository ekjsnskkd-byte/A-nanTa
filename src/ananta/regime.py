"""A transparent baseline regime classifier; replace with trained detector later."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ananta.contracts import Candle, validate_history

Regime = Literal["trend", "range", "volatile", "unknown"]


@dataclass(slots=True)
class RegimeHeuristic:
    lookback: int = 6
    trend_threshold: float = 0.65
    volatility_threshold: float = 0.015

    def detect(self, history: Sequence[Candle]) -> Regime:
        validate_history(history)
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if len(history) < self.lookback + 1:
            return "unknown"
        closes = [bar.close for bar in history[-self.lookback - 1:]]
        moves = [abs(a - b) for a, b in zip(closes[1:], closes[:-1])]
        path = sum(moves)
        if path == 0:
            return "range"
        mean_absolute_move = path / self.lookback / closes[0]
        if mean_absolute_move > self.volatility_threshold:
            return "volatile"
        efficiency = abs(closes[-1] - closes[0]) / path
        return "trend" if efficiency >= self.trend_threshold else "range"
