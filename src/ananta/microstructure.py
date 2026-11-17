"""Strict L1 quote metrics and *real* L2 order book primitives.

This module does NOT infer order flow or liquidity from OHLC candles.
TLOB integration requires actual exchange L2 snapshots and its own training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


def _valid_time(time: datetime) -> bool:
    return time.tzinfo is not None and time.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class Quote:
    timestamp: datetime
    symbol: str
    bid: float
    ask: float

    def __post_init__(self) -> None:
        if not _valid_time(self.timestamp) or not self.symbol:
            raise ValueError("quote requires timezone-aware timestamp and symbol")
        if not (math.isfinite(self.bid) and math.isfinite(self.ask)):
            raise ValueError("quotes must be finite")
        if self.bid <= 0 or self.ask < self.bid:
            raise ValueError("quote prices must be positive and non-crossed")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 10_000.0


@dataclass(frozen=True, slots=True)
class Level:
    price: float
    size: float

    def __post_init__(self) -> None:
        if (not math.isfinite(self.price) or self.price <= 0
                or not math.isfinite(self.size) or self.size <= 0):
            raise ValueError("LOB price and size must be positive finite values")


@dataclass(frozen=True, slots=True)
class L2Snapshot:
    timestamp: datetime
    symbol: str
    exchange: str
    bids: tuple[Level, ...]  # sorted best to worst, descending
    asks: tuple[Level, ...]  # sorted best to worst, ascending

    def __post_init__(self) -> None:
        if not _valid_time(self.timestamp) or not self.symbol or not self.exchange:
            raise ValueError("LOB requires timestamp, symbol and exchange")
        if not self.bids or not self.asks:
            raise ValueError("both sides of L2 book required")
        if self.bids[0].price >= self.asks[0].price:
            raise ValueError("crossed or locked book")
        if any(left.price <= right.price for left, right in
               zip(self.bids, self.bids[1:])):
            raise ValueError("bid levels must descend")
        if any(left.price >= right.price for left, right in
               zip(self.asks, self.asks[1:])):
            raise ValueError("ask levels must ascend")

    def imbalance(self, depth: int = 5) -> float:
        """Observed L2 size imbalance in [-1,+1]; not an inferred market-wide flow."""
        if depth < 1:
            raise ValueError("depth must be positive")
        bid_size = sum(level.size for level in self.bids[:depth])
        ask_size = sum(level.size for level in self.asks[:depth])
        return (bid_size - ask_size) / (bid_size + ask_size)
