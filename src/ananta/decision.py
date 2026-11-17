"""Fail-closed signal policy for research. It cannot place an order."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ananta.contracts import Forecast
from ananta.microstructure import Quote


@dataclass(frozen=True, slots=True)
class PolicyLimits:
    min_edge_bps: float = 3.0
    max_spread_bps: float = 35.0
    max_forecast_age_seconds: float = 600.0

    def __post_init__(self) -> None:
        values = (self.min_edge_bps, self.max_spread_bps, self.max_forecast_age_seconds)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("policy limits must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ProposedSignal:
    action: str
    reason: str
    net_edge_bps: float
    research_only: bool = True


def assess_forecast(
    forecast: Forecast,
    quote: Quote,
    limits: PolicyLimits | None = None,
    round_trip_extra_cost_bps: float = 0.0,
) -> ProposedSignal:
    """Candidate direction after approximate L1 costs; no position sizing or execution.

    The spread is paid once for a round-trip under a constant-spread approximation.
    Live-fill models must replace this estimate before any trading decision.
    """
    config = limits or PolicyLimits()
    if forecast.symbol != quote.symbol:
        raise ValueError("forecast/quote symbol mismatch")
    if not math.isfinite(round_trip_extra_cost_bps) or round_trip_extra_cost_bps < 0:
        raise ValueError("extra costs must be nonnegative finite")
    age_seconds = (quote.timestamp - forecast.observed_at).total_seconds()
    if age_seconds < 0:
        return ProposedSignal("NO_TRADE", "QUOTE_BEFORE_FORECAST", float("-inf"))
    if age_seconds > config.max_forecast_age_seconds:
        return ProposedSignal("NO_TRADE", "STALE_FORECAST", float("-inf"))
    spread = quote.spread_bps
    if spread > config.max_spread_bps:
        return ProposedSignal("NO_TRADE", "EXCESSIVE_SPREAD", float("-inf"))
    edge = abs(forecast.expected_return) * 10_000.0 - spread - round_trip_extra_cost_bps
    if edge <= config.min_edge_bps:
        return ProposedSignal("NO_TRADE", "INSUFFICIENT_NET_EDGE", edge)
    return ProposedSignal("BUY" if forecast.expected_return > 0 else "SELL",
                          "RESEARCH_CANDIDATE_ONLY", edge)
