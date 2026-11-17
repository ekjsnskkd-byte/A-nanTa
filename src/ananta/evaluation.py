"""Leak-conscious splitting and spread-aware single-trade returns."""

from __future__ import annotations

from typing import Iterator


def walk_forward_splits(
    samples: int, train_bars: int, test_bars: int, embargo_bars: int = 0,
    step_bars: int | None = None,
) -> Iterator[tuple[range, range]]:
    """Yield expanding training and strictly later testing windows (end-exclusive)."""
    if min(samples, train_bars, test_bars) <= 0 or embargo_bars < 0:
        raise ValueError("invalid split dimensions")
    step = test_bars if step_bars is None else step_bars
    if step <= 0:
        raise ValueError("step_bars must be positive")
    train_end = train_bars
    while train_end + embargo_bars + test_bars <= samples:
        test_start = train_end + embargo_bars
        yield range(0, train_end), range(test_start, test_start + test_bars)
        train_end += step


def realized_net_return(
    direction: str, entry_bid: float, entry_ask: float, exit_bid: float,
    exit_ask: float, commission_bps: float = 0.0, slippage_bps: float = 0.0,
) -> float:
    """Single round trip relative return: long enters at ask, exits at bid."""
    if direction not in {"BUY", "SELL"}:
        raise ValueError("direction must be BUY or SELL")
    if min(entry_bid, entry_ask, exit_bid, exit_ask) <= 0:
        raise ValueError("quotes must be positive")
    if entry_bid > entry_ask or exit_bid > exit_ask:
        raise ValueError("crossed quote")
    if min(commission_bps, slippage_bps) < 0:
        raise ValueError("cost must be nonnegative")
    if direction == "BUY":
        gross = exit_bid / entry_ask - 1.0
    else:
        gross = (entry_bid - exit_ask) / entry_bid
    return gross - (commission_bps + slippage_bps) / 10_000.0
