"""Shared, leak-resistant contracts for market inputs and model outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        if not _aware(self.timestamp):
            raise ValueError("timestamp must be timezone-aware")
        if not self.symbol or not self.timeframe:
            raise ValueError("symbol and timeframe are required")
        values = (self.open, self.high, self.low, self.close)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("OHLC prices must be finite and positive")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low exceeds an OHLC value")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is below an OHLC value")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ValueError("volume must be finite and nonnegative")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be provided together")
        if self.bid is not None:
            if not (math.isfinite(self.bid) and math.isfinite(self.ask)):
                raise ValueError("bid/ask must be finite")
            if self.bid <= 0 or self.ask < self.bid:
                raise ValueError("bid/ask must be positive and non-crossed")


@dataclass(frozen=True, slots=True)
class Forecast:
    expert_id: str
    symbol: str
    timeframe: str
    observed_at: datetime
    horizon: int
    expected_return: float

    def __post_init__(self) -> None:
        if not self.expert_id or not self.symbol or not self.timeframe:
            raise ValueError("expert_id, symbol and timeframe are required")
        if not _aware(self.observed_at):
            raise ValueError("observed_at must be timezone-aware")
        if self.horizon < 1 or not math.isfinite(self.expected_return):
            raise ValueError("invalid horizon or expected return")


def validate_history(history: Sequence[Candle], min_bars: int = 2) -> None:
    if len(history) < min_bars:
        raise ValueError(f"at least {min_bars} bars are required")
    first = history[0]
    previous = first.timestamp
    for i, bar in enumerate(history):
        if bar.symbol != first.symbol or bar.timeframe != first.timeframe:
            raise ValueError("mixed symbol/timeframe in a history window")
        if i > 0 and bar.timestamp <= previous:
            raise ValueError("history must have strictly increasing unique timestamps")
        previous = bar.timestamp
